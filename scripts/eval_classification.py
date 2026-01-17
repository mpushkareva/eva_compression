import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import pipeline
from tqdm import tqdm

from utils.dataloaders.image_dataloaders import get_imagenet_dataloaders, get_cifar_dataloaders


def accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5)
) -> List[float]:
    """Compute top‑k accuracy from raw logits."""
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # (B, maxk)
    pred = pred.t()  # (maxk, B)
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k * (100.0 / batch_size)).item())
    return res
def _get_hf_token(token: str = None) -> str:
    """Get HuggingFace token from argument, environment, or cache."""
    import os
    if token:
        return token
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except Exception:
        return None


def _load_quantized_model(quantized_path: Path, device: str):
    """Load quantized model directly from file."""
    wrapper_path = quantized_path.with_suffix('.wrapper.pth')
    if wrapper_path.exists():
        print(f"Loading quantized wrapper model from: {wrapper_path}")
        state_dict = torch.load(wrapper_path, map_location=device)
        print("Quantized wrapper model loaded successfully!")
        return state_dict
    
    if not quantized_path.exists():
        raise FileNotFoundError(f"Quantized model file not found: {quantized_path}")
    
    print(f"Loading quantized model state dict from: {quantized_path}")
    state_dict = torch.load(quantized_path, map_location=device)
    print("Quantized model state dict loaded!")
    return state_dict


def build_hf_pipeline(
    model_name_or_path: str, 
    device: str, 
    num_labels: int, 
    token: str = None,
    quantized_model_path: str = None
):
    """
    Build HF image-classification pipeline.

    model_name_or_path:
        - 'timm/eva02_base_patch14_224.mim_in22k'  (EVA feature extractor -> you need head)
        - 'your-org/quantized-eva-checkpoint'
        - local path to base model (only used if quantized_model_path is not provided)
    
    quantized_model_path:
        - Path to quantized model state dict (.pth file) or wrapper (.wrapper.pth file)
        - If provided, will load the quantized model directly without matching with base model
    """
    import os
    from pathlib import Path
    from transformers import AutoImageProcessor, TimmWrapperForImageClassification

    token = _get_hf_token(token)
    device_index = 0 if device == "cuda" and torch.cuda.is_available() else -1

    if quantized_model_path:
        # Load quantized model directly
        quantized_path = Path(quantized_model_path)
        print(f"Loading quantized model directly from: {quantized_path}")
        # quantized_state_dict = _load_quantized_model(quantized_path, device)
        
        image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, token=token)
        model = TimmWrapperForImageClassification.from_pretrained(
            quantized_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            token=token,
        )
    else:
        # Load base model normally
        image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, token=token)
        model = TimmWrapperForImageClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            token=token,
        )

    pipe = pipeline(
        task="image-classification",
        model=model,
        image_processor=image_processor,
        device=device_index,
    )
    return pipe

def _get_model_label_order(model, num_classes: int):
    """Extract label order from model config."""
    if not (hasattr(model, 'config') and hasattr(model.config, 'id2label')):
        return None
    
    id2label = model.config.id2label
    model_label_order = []
    for i in range(num_classes):
        label = id2label.get(i, id2label.get(str(i), f"LABEL_{i}"))
        model_label_order.append(label)
    return model_label_order


def _is_generic_label_format(labels):
    """Check if labels are generic LABEL_X format."""
    return labels and all(label.startswith("LABEL_") for label in labels[:10])


def _create_label_mapping(dataset, dataset_synsets, target_order, num_classes):
    """Create mapping from dataset indices to target indices."""
    dataset_class_to_target_idx = {}
    for target_idx, synset in enumerate(target_order):
        if synset in dataset_synsets:
            dataset_idx = dataset_synsets.index(synset)
            dataset_class_to_target_idx[dataset_idx] = target_idx
    
    if len(dataset_class_to_target_idx) != num_classes:
        print(f"WARNING: Only {len(dataset_class_to_target_idx)}/{num_classes} classes mapped!")
        return None
    
    idx_mapping = torch.zeros(len(dataset.classes), dtype=torch.long)
    for dataset_idx, target_idx in dataset_class_to_target_idx.items():
        idx_mapping[dataset_idx] = target_idx
    return idx_mapping


def _get_label_mapping(dataset, model, num_classes: int):
    """Get label mapping between dataset and model output order."""
    if not (hasattr(dataset, 'classes') and num_classes == 1000):
        return None
    
    standard_synsets = get_standard_imagenet_synsets()
    if not standard_synsets:
        return None
    
    model_label_order = _get_model_label_order(model, num_classes)
    dataset_synsets = getattr(dataset, 'wnids', dataset.classes)
    
    # Use model order if available and not generic, otherwise use standard
    if model_label_order and not _is_generic_label_format(model_label_order):
        target_order = model_label_order
    else:
        if _is_generic_label_format(model_label_order):
            print("WARNING: Model has generic LABEL_X format, using standard ImageNet order")
        target_order = standard_synsets
    
    return _create_label_mapping(dataset, dataset_synsets, target_order, num_classes)


def evaluate_with_pipeline(pipe, val_loader: DataLoader, num_classes: int) -> Tuple[float, float, int]:
    """Evaluate model wrapped in HF pipeline on a classification dataset.
    
    Returns:
        Tuple of (top1_accuracy, top5_accuracy, n_samples)
    """
    import torchvision.transforms as T
    
    top1_sum = 0.0
    top5_sum = 0.0
    n_samples = 0

    model = pipe.model
    dataset = val_loader.dataset
    idx_mapping = _get_label_mapping(dataset, model, num_classes)
    
    to_pil = T.ToPILImage()
    image_processor = pipe.image_processor
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader, desc="Evaluating", unit="batch")):
            batch_size = targets.size(0)
            
            # Process images through image processor
            processed_images = []
            for img in images:
                img_pil = to_pil(img.cpu())
                inputs = image_processor(img_pil, return_tensors="pt")
                pixel_values = inputs.pixel_values if hasattr(inputs, 'pixel_values') else inputs['pixel_values']
                processed_images.append(pixel_values)
            
            pixel_values_batch = torch.cat(processed_images, dim=0).to(device)
            
            outputs = model(pixel_values_batch)
            logits_batch = outputs.logits.cpu() if hasattr(outputs, 'logits') else outputs.cpu()
            
            # Apply label mapping if available
            if idx_mapping is not None:
                targets = idx_mapping[targets]

            # Debug first batch
            if batch_idx == 0:
                _, pred_top1 = logits_batch.topk(1, dim=1)
                matches = (pred_top1.squeeze() == targets).sum().item()
                print(f"\nFirst batch: {matches}/{batch_size} correct")
                print(f"  Targets (first 5): {targets[:5].tolist()}")
                print(f"  Predictions (first 5): {pred_top1[:5].squeeze().tolist()}")

            top1, top5 = accuracy_from_logits(logits_batch, targets, topk=(1, min(5, num_classes)))
            top1_sum += top1 * batch_size
            top5_sum += top5 * batch_size
            n_samples += batch_size

    top1_avg = top1_sum / n_samples
    top5_avg = top5_sum / n_samples
    print(f"Eval: Top-1 = {top1_avg:.2f}%, Top-5 = {top5_avg:.2f}% (N={n_samples})")
    return top1_avg, top5_avg, n_samples


def get_standard_imagenet_synsets():
    """Get standard ImageNet-1K synset list in canonical order."""
    try:
        # Try to load from create_synsets_from_standard.py
        from utils.create_synsets_from_standard import STANDARD_IMAGENET_SYNSETS
        return STANDARD_IMAGENET_SYNSETS
    except ImportError:
        # Fallback: try to read from synsets.txt in devkit
        synsets_file = Path("data/imagenet/ILSVRC2012_devkit_t12/data/synsets.txt")
        if synsets_file.exists():
            with open(synsets_file, "r") as f:
                return [line.strip().split()[0] for line in f.readlines()]
    return None

def build_val_loader(args, normalize: bool = True) -> Tuple[DataLoader, int]:
    """Build val loader (ImageNet or CIFAR) and infer num_classes.
    
    Args:
        args: Parsed arguments
        normalize: If False, skip normalization (for pipeline evaluation where pipeline handles it)
    """
    if args.dataset in ("cifar10", "cifar100"):
        _, val_loader = get_cifar_dataloaders(
            root=args.val_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            use_randaug=False,
            cifar_version=args.dataset,
            use_imagenet_labels=args.use_imagenet_labels,
            normalize=normalize,
        )
        num_classes = 1000 if (args.dataset == "cifar10" and args.use_imagenet_labels) else len(val_loader.dataset.classes)
    else:
        _, val_loader = get_imagenet_dataloaders(
            val_dir=args.val_dir,
            train_dir=args.train_dir if args.train_dir else None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            eval_resize=args.eval_resize,
            use_randaug=False,
            normalize=normalize,
        )
        if not hasattr(val_loader.dataset, "classes"):
            raise RuntimeError("Cannot infer num_classes from dataset.")
        num_classes = len(val_loader.dataset.classes)
    
    return val_loader, num_classes

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate EVA or quantized HF image classification models."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HF model id or local path (EVA or quantized version).",
    )
    parser.add_argument("--train-dir", type=str, default="")
    parser.add_argument("--val-dir", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--eval-resize", type=int, default=256)
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        choices=["imagenet", "cifar10", "cifar100"],
        help="Which dataset to build val loader for.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (optional, will try env vars and HF cache if not provided).",
    )
    parser.add_argument(
        "--use-imagenet-labels",
        action="store_true",
        help="For CIFAR-10: map CIFAR-10 labels to ImageNet class indices. "
             "This allows using ImageNet-trained models directly on CIFAR-10 data. "
             "The model will output 1000 classes (ImageNet size) instead of 10.",
    )
    parser.add_argument(
        "--quantized-model",
        type=str,
        default=None,
        help="Path to quantized model state dict (.pth file) or wrapper (.wrapper.pth file). "
             "If provided, will load the base model from --model and then load quantized weights.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # For pipeline evaluation, skip normalization in dataloader (pipeline handles it)
    val_loader, num_classes = build_val_loader(args, normalize=False)
    pipe = build_hf_pipeline(
        args.model, 
        device=args.device, 
        num_labels=num_classes,
        quantized_model_path=args.quantized_model
    )
    evaluate_with_pipeline(pipe, val_loader, num_classes=num_classes)


if __name__ == "__main__":
    main()