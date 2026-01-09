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
        - local path to base model (for loading quantized weights)
    
    quantized_model_path:
        - Path to quantized model state dict (.pth file)
        - If provided, will load the base model and then load quantized weights
    """
    import os
    from pathlib import Path
    from transformers import AutoImageProcessor, TimmWrapperForImageClassification

    if token is None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token is None:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            pass
    device_index = 0 if device == "cuda" and torch.cuda.is_available() else -1

    # Get token from environment or HuggingFace cache
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    # If still None, try to read from HF cache location
    if token is None:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            pass

    # Load image processor
    image_processor = AutoImageProcessor.from_pretrained(
        model_name_or_path,
        token=token,
    )

    # Create classification wrapper with a fresh head of size `num_labels`
    model = TimmWrapperForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,  # needed because original checkpoint has no head
        token=token,
    )
    
    # Load quantized model if path is provided
    if quantized_model_path is not None:
        quantized_path = Path(quantized_model_path)
        if quantized_path.exists():
            print(f"Loading quantized model from: {quantized_path}")
            try:
                # Try loading wrapper state dict first (preferred)
                wrapper_path = quantized_path.with_suffix('.wrapper.pth')
                if wrapper_path.exists():
                    print(f"Loading quantized wrapper model from: {wrapper_path}")
                    state_dict = torch.load(wrapper_path, map_location=device)
                    model.load_state_dict(state_dict, strict=False)
                    print("Quantized wrapper model loaded successfully!")
                else:
                    # Fallback: load quantized model state dict and replace in wrapper
                    print(f"Loading quantized model state dict from: {quantized_path}")
                    quantized_state_dict = torch.load(quantized_path, map_location=device)
                    
                    # Replace the underlying model's weights with quantized weights
                    if hasattr(model, 'model'):
                        # Get the underlying timm model
                        underlying_model = model.model
                        # Load quantized weights into underlying model
                        underlying_model.load_state_dict(quantized_state_dict, strict=False)
                        print("Quantized model weights loaded into wrapper!")
                    else:
                        # Try loading directly into model
                        model.load_state_dict(quantized_state_dict, strict=False)
                        print("Quantized model weights loaded!")
            except Exception as e:
                print(f"Warning: Could not load quantized model: {e}")
                print("Continuing with non-quantized model...")
                import traceback
                traceback.print_exc()
        else:
            print(f"Warning: Quantized model path does not exist: {quantized_path}")
            print("Continuing with non-quantized model...")
    
    # Check if head is randomly initialized (move this check AFTER model is created)
    if hasattr(model, 'classifier'):
        head_params = list(model.classifier.parameters())
        if len(head_params) > 0:
            # Check if weights are close to zero (random init) or have meaningful values
            weight_norm = head_params[0].data.norm().item()
            print(f"Classification head weight norm: {weight_norm:.6f}")
            if weight_norm < 0.01:
                print("WARNING: Classification head appears to be randomly initialized!")
                print("The model may need fine-tuning or a checkpoint with a trained head.")

    pipe = pipeline(
        task="image-classification",
        model=model,
        image_processor=image_processor,
        device=device_index,
    )
    return pipe

def evaluate_with_pipeline(pipe, val_loader: DataLoader, num_classes: int) -> Tuple[float, float, int]:
    """Evaluate model wrapped in HF pipeline on a classification dataset.
    
    Returns:
        Tuple of (top1_accuracy, top5_accuracy, n_samples)
    """
    from PIL import Image
    import torchvision.transforms as T
    
    top1_sum = 0.0
    top5_sum = 0.0
    n_samples = 0

    # Get label mapping from model config
    model = pipe.model
    model_label_order = None
    if hasattr(model, 'config') and hasattr(model.config, 'id2label'):
        id2label = model.config.id2label
        label2id = {}
        for idx, label in id2label.items():
            label2id[label] = int(idx) if isinstance(idx, str) else idx
        print(f"Model has id2label mapping with {len(id2label)} classes")
        # Print first few labels for debugging
        sample_labels = list(id2label.items())[:5]
        print(f"Sample model labels: {sample_labels}")
        
        # Extract model's label order (synset IDs)
        # Handle both string and integer keys in id2label
        model_label_order = []
        for i in range(num_classes):
            # Try both string and integer keys
            label = id2label.get(i, id2label.get(str(i), f"LABEL_{i}"))
            model_label_order.append(label)
        print(f"Model label order (first 5): {model_label_order[:5]}")
        print(f"Model label order (last 5): {model_label_order[-5:]}")
    else:
        label2id = {f'LABEL_{i}': i for i in range(num_classes)}
        print(f"Warning: Model has no id2label, using default LABEL_0 to LABEL_{num_classes-1}")
    
    # CRITICAL FIX: Map dataset's alphabetical indices to model's output order
    dataset = val_loader.dataset
    idx_mapping = None
    standard_synsets = None
    if hasattr(dataset, 'classes') and num_classes == 1000:
        # Get standard ImageNet synset order
        standard_synsets = get_standard_imagenet_synsets()
        
        # Use wnids (synset IDs) if available, otherwise fall back to classes
        # For torchvision ImageNet, wnids contains synset IDs like "n01440764"
        dataset_synsets = getattr(dataset, 'wnids', None)
        if dataset_synsets is None:
            # Fallback: if no wnids, try to extract from classes (shouldn't happen for ImageNet)
            dataset_synsets = dataset.classes
        
        if standard_synsets:
            print(f"Found {len(standard_synsets)} standard ImageNet synsets")
            print(f"Dataset has {len(dataset.classes)} classes")
            if hasattr(dataset, 'wnids'):
                print(f"Dataset wnids (synset IDs) available: {len(dataset.wnids)}")
                print(f"Dataset wnids (first 5): {dataset.wnids[:5]}")
                print(f"Dataset wnids (last 5): {dataset.wnids[-5:]}")
            print(f"Dataset classes (first 5): {dataset.classes[:5]}")
            print(f"Dataset classes (last 5): {dataset.classes[-5:]}")
            
            # Determine target order: use model's label order if available, otherwise standard order
            target_order = model_label_order if model_label_order else standard_synsets
            print(f"Target label order (first 5): {target_order[:5]}")
            print(f"Target label order (last 5): {target_order[-5:]}")
            
            # Check if model labels are generic "LABEL_X" format (no synset info)
            if model_label_order and all(label.startswith("LABEL_") for label in model_label_order[:10]):
                print("WARNING: Model has generic LABEL_X format, cannot determine synset mapping!")
                print("Assuming model outputs are in the same order as dataset (alphabetical)")
                print("If model was trained with standard ImageNet order, accuracy will be incorrect!")
                # Don't create mapping - assume identity (alphabetical = alphabetical)
                idx_mapping = None
            else:
                # Verify target_order matches standard order if using standard
                if not model_label_order:
                    if target_order == standard_synsets:
                        print("Using standard ImageNet synset order")
                    else:
                        print("WARNING: Target order does not match standard synsets!")
                
                # Create mapping: dataset_class_index -> target_index (model or standard)
                # Use dataset_synsets (wnids) for matching, not dataset.classes
                dataset_class_to_target_idx = {}
                for target_idx, synset in enumerate(target_order):
                    if synset in dataset_synsets:
                        dataset_idx = dataset_synsets.index(synset)
                        dataset_class_to_target_idx[dataset_idx] = target_idx
                    else:
                        print(f"WARNING: Synset {synset} at target_idx {target_idx} not found in dataset synsets!")
                
                print(f"Successfully mapped {len(dataset_class_to_target_idx)} classes")
                
                # Verify mapping completeness
                if len(dataset_class_to_target_idx) != 1000:
                    print(f"ERROR: Only {len(dataset_class_to_target_idx)}/1000 classes mapped!")
                    print("This will cause incorrect label mapping!")
                    # Show some examples of missing mappings
                    missing_in_target = [c for c in dataset_synsets if c not in target_order]
                    if missing_in_target:
                        print(f"Dataset synsets not in target order (first 10): {missing_in_target[:10]}")
                
                # Create tensor mapping for efficient conversion
                if len(dataset_class_to_target_idx) == 1000:
                    idx_mapping = torch.zeros(len(dataset.classes), dtype=torch.long)
                    for dataset_idx, target_idx in dataset_class_to_target_idx.items():
                        idx_mapping[dataset_idx] = target_idx
                    print("Label mapping tensor created successfully")
                    # Show more sample mappings
                    print(f"Sample mappings:")
                    for i in [0, 1, 100, 500, 999]:
                        if i < len(dataset.classes):
                            wnid = dataset_synsets[i] if i < len(dataset_synsets) else "N/A"
                            print(f"  dataset_idx {i} (wnid={wnid}) -> target_idx {idx_mapping[i].item()}")
                else:
                    print(f"ERROR: Cannot create mapping tensor - only {len(dataset_class_to_target_idx)}/1000 classes mapped")
        else:
            print("Warning: Could not load standard ImageNet synsets")
            if model_label_order and not all(label.startswith("LABEL_") for label in model_label_order[:10]):
                print("Using model's label order for mapping")
                dataset_synsets = getattr(dataset, 'wnids', dataset.classes)
                dataset_class_to_target_idx = {}
                for target_idx, synset in enumerate(model_label_order):
                    if synset in dataset_synsets:
                        dataset_idx = dataset_synsets.index(synset)
                        dataset_class_to_target_idx[dataset_idx] = target_idx
                if len(dataset_class_to_target_idx) == 1000:
                    idx_mapping = torch.zeros(len(dataset.classes), dtype=torch.long)
                    for dataset_idx, target_idx in dataset_class_to_target_idx.items():
                        idx_mapping[dataset_idx] = target_idx
                    print("Label mapping created from model's label order")
                else:
                    print(f"ERROR: Only {len(dataset_class_to_target_idx)}/1000 classes mapped from model labels")
    
    # IMPORTANT: Use model directly to get logits instead of pipeline scores
    # Pipeline returns probabilities, but we need raw logits for proper comparison
    to_pil = T.ToPILImage()
    image_processor = pipe.image_processor
    device = next(model.parameters()).device

    i = 0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating", unit="batch"):
            batch_size = targets.size(0)
            
            # Process batch of images through image processor
            # Convert tensors to PIL images and process them
            processed_images = []
            for img in images:
                img_cpu = img.cpu()
                img_pil = to_pil(img_cpu)
                inputs = image_processor(img_pil, return_tensors="pt")
                if hasattr(inputs, 'pixel_values'):
                    pixel_values = inputs.pixel_values
                else:
                    pixel_values = inputs['pixel_values']
                processed_images.append(pixel_values)
            
            # Stack all processed images into a batch
            pixel_values_batch = torch.cat(processed_images, dim=0).to(device)  # (B, C, H, W)
            
            # Get raw logits from model in one forward pass
            outputs = model(pixel_values_batch)
            if hasattr(outputs, 'logits'):
                logits_batch = outputs.logits.cpu()  # (B, num_classes)
            elif isinstance(outputs, torch.Tensor):
                logits_batch = outputs.cpu()
            else:
                # Fallback: process one by one (slower but works)
                logits_list = []
                for img in images:
                    img_cpu = img.cpu()
                    img_pil = to_pil(img_cpu)
                    out = pipe(img_pil, top_k=num_classes)
                    logits = torch.empty(num_classes, dtype=torch.float32)
                    for item in out:
                        label_str = item["label"]
                        if label_str in label2id:
                            idx = label2id[label_str]
                            score = item["score"]
                            logits[idx] = torch.log(torch.tensor(score) / (1 - score + 1e-8))
                        else:
                            try:
                                idx = int(label_str.split("_")[-1])
                                if 0 <= idx < num_classes:
                                    score = item["score"]
                                    logits[idx] = torch.log(torch.tensor(score) / (1 - score + 1e-8))
                            except (ValueError, IndexError):
                                pass
                    logits_list.append(logits.unsqueeze(0))
                logits_batch = torch.cat(logits_list, dim=0)
            
            # Reorder logits if model's output order doesn't match standard ImageNet order
            if idx_mapping is not None and model_label_order is not None and standard_synsets is not None:
                # Check if model order matches standard order
                if model_label_order != standard_synsets:
                    print("WARNING: Model label order does not match standard ImageNet order!")
                    print("Creating logit reordering mapping...")
                    # Create mapping: model_output_idx -> standard_idx
                    logit_reorder = torch.zeros(num_classes, dtype=torch.long)
                    for standard_idx, synset in enumerate(standard_synsets):
                        if synset in model_label_order:
                            model_idx = model_label_order.index(synset)
                            logit_reorder[model_idx] = standard_idx
                        else:
                            print(f"WARNING: Synset {synset} not in model label order!")
                    # Reorder logits: logits_batch[:, model_idx] -> logits_batch[:, standard_idx]
                    logits_batch = logits_batch[:, logit_reorder]
                    print("Logits reordered to match standard ImageNet order")
            
            # Apply label mapping if available (map dataset indices to standard ImageNet indices)
            if idx_mapping is not None:
                targets = idx_mapping[targets]

            # Debug: print first batch predictions vs targets
            if i == 0:
                _, pred_top1 = logits_batch.topk(1, dim=1)
                print(f"\nDebug - First batch (batch_size={batch_size}):")
                print(f"  Targets (first 5, after mapping): {targets[:5].tolist()}")
                print(f"  Predictions (first 5): {pred_top1[:5].squeeze().tolist()}")
                print(f"  Logits shape: {logits_batch.shape}")
                print(f"  Logits range: [{logits_batch.min().item():.2f}, {logits_batch.max().item():.2f}]")
                print(f"  Logits mean: {logits_batch.mean().item():.2f}, std: {logits_batch.std().item():.2f}")
                # Check if any predictions match targets
                matches = (pred_top1.squeeze() == targets).sum().item()
                print(f"  Correct predictions in first batch: {matches}/{batch_size}")

            top1, top5 = accuracy_from_logits(
                logits_batch,
                targets,
                topk=(1, min(5, num_classes))
            )
            top1_sum += top1 * batch_size
            top5_sum += top5 * batch_size
            n_samples += batch_size
            
            i += 1

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
        # CIFAR uses a single root; we use val_dir as that root here.
        # If use_imagenet_labels is True, labels will be mapped to ImageNet indices
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
    else:
        # Default: ImageNet-style folders
        # train_dir is optional - pass None if empty string
        train_dir = args.train_dir if args.train_dir else None
        _, val_loader = get_imagenet_dataloaders(
            val_dir=args.val_dir,
            train_dir=train_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            eval_resize=args.eval_resize,
            use_randaug=False,
            normalize=normalize,
        )

    ds = val_loader.dataset
    
    # If using ImageNet labels for CIFAR-10, we need 1000 classes (ImageNet size)
    if args.dataset == "cifar10" and args.use_imagenet_labels:
        num_classes = 1000  # ImageNet has 1000 classes
    elif hasattr(ds, "classes"):
        num_classes = len(ds.classes)
    else:
        raise RuntimeError("Cannot infer num_classes from dataset.")
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