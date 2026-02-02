import argparse
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor

from scripts.quantize_torch import load_eva_model, quantize_torch
from utils.dataloaders.image_dataloaders import get_imagenet_dataloaders


def accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5)
) -> list[float]:
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


def _get_hf_token(token: str = None) -> Optional[str]:
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


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    num_classes: int = 1000,
    image_processor: Optional[AutoImageProcessor] = None
) -> Tuple[float, float, int]:
    top1_sum = 0.0
    top5_sum = 0.0
    n_samples = 0

    model.eval()
    
    # Use image processor if provided, otherwise use direct tensor input
    # use_image_processor = image_processor is not None
    # if use_image_processor:
    #     import torchvision.transforms as T
    #     to_pil = T.ToPILImage()
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader, desc="Evaluating", unit="batch")):
            batch_size = targets.size(0)
            
            # if use_image_processor:
            # Process images through image processor
            processed_images = []
            for img in images:
                # img_pil = to_pil(img.cpu())
                # inputs = image_processor(img_pil, return_tensors="pt")
                inputs = image_processor(img, return_tensors="pt")
                pixel_values = inputs['pixel_values']
                processed_images.append(pixel_values)
            
            images = torch.cat(processed_images, dim=0).to(device)
            # else:
            #     images = images.to(device)
            
            targets = targets.to(device)
            logits = model(images)
            logits = logits.cpu()
            targets = targets.cpu()
            
            if batch_idx == 0:
                _, pred_top1 = logits.topk(1, dim=1)
                matches = (pred_top1.squeeze() == targets).sum().item()
                print(f"\nFirst batch: {matches}/{batch_size} correct")
                print(f"  Targets (first 5): {targets[:5].tolist()}")
                print(f"  Predictions (first 5): {pred_top1[:5].squeeze().tolist()}")
                print(f"  Logits shape: {logits.shape}")
            
            top1, top5 = accuracy_from_logits(logits, targets, topk=(1, min(5, num_classes)))
            top1_sum += top1 * batch_size
            top5_sum += top5 * batch_size
            n_samples += batch_size

    top1_avg = top1_sum / n_samples
    top5_avg = top5_sum / n_samples
    print(f"\nEvaluation Results:")
    print(f"  Top-1 Accuracy: {top1_avg:.2f}%")
    print(f"  Top-5 Accuracy: {top5_avg:.2f}%")
    print(f"  Total Samples: {n_samples}")
    return top1_avg, top5_avg, n_samples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate quantized PyTorch model on ImageNet dataset (without timm/HF)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to quantized PyTorch model file (.pth)"
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        required=True,
        help="Path to ImageNet validation directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 64)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading (default: 4)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of classes (default: 1000 for ImageNet)"
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="torch",
        choices=["torch", 'fixed', 'manual', 'origin', 'fixed_op'],
        help="Quantize EVA model"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dynamic",
        help="Quantization mode (default: dynamic)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="qint8",
        help="Quantization dtype (default: qint8)"
    )
    parser.add_argument(
        "--weight-bits",
        type=int,
        default=8,
        help="Quantization weight bits (default: 8)"
    )
    parser.add_argument(
        "--activation-bits",
        type=int,
        default=8,
        help="Quantization activation bits (default: 8)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="HF model id or local path for image processor (e.g., 'timm/eva02_base_patch14_224.mim_in22k'). "
             "If provided, will use HF image processor for preprocessing."
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (optional, will try env vars and HF cache if not provided)."
    ) 
    parser.add_argument(
        "--attention",
        action='store_true',
        help="Quantize attention layers"
    )
    parser.add_argument(
        "--mlp",
        action='store_true',
        help="Quantize MLP layers"
    )
    parser.add_argument(
        "--embedding",
        action='store_true',
        help="Quantize embedding layers"
    )
    parser.add_argument(
        "--norm",
        action='store_true',
        help="Quantize normalization layers"
    )
    parser.add_argument(
        "--head",
        action='store_true',
        help="Quantize head layers"
    )
    parser.add_argument(
        "--other",
        action='store_true',
        help="Quantize other layers"
    )
    parser.add_argument(
        "--quantize-all",
        action='store_true',
        help="Quantize all layers"
    )
    parser.add_argument(
        "--forward-format",
        type=str,
        default="fixed",
        help="Forward format (default: fixed)"
    )
    parser.add_argument(
        "--forward-wl",
        type=int,
        default=8,
        help="Forward word length (default: 8)"
    )
    parser.add_argument(
        "--forward-fl",
        type=int,
        default=4,
        help="Forward fraction length (default: 2)"
    )
    parser.add_argument(
        "--forward-exp",
        type=int,
        default=5,
        help="Forward exponent (default: 5)"
    )
    parser.add_argument(
        "--forward-man",
        type=int,
        default=1,
        help="Forward mantissa (default: 1)"
    )
    parser.add_argument(
        "--backward-exp",
        type=int,
        default=5,
        help="Backward exponent (default: 5)"
    )
    parser.add_argument(
        "--backward-man",
        type=int,
        default=1,
        help="Backward mantissa (default: 1)"
    )
    parser.add_argument(
        "--forward-rounding",
        type=str,
        default="nearest",
        help="Forward rounding (default: nearest)"
    )
    parser.add_argument(
        "--backward-rounding",
        type=str,
        default="nearest",
        help="Backward rounding (default: nearest)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.quant_type == "torch":
        model = quantize_torch(args.model, args.num_classes, args.mode, args.dtype, args.weight_bits, args.activation_bits)
    elif args.quant_type == "fixed":
        from scripts.quantize_fp import quantize_fp
        model = quantize_fp(args.model, args.num_classes, args.attention, args.mlp, args.embedding, args.norm, args.head, args.other, args.quantize_all, args.forward_format, args.forward_wl, args.forward_fl, args.forward_exp, args.forward_man, args.backward_exp, args.backward_man, args.forward_rounding, args.backward_rounding)
    elif args.quant_type == "fixed_op":
        from scripts.quantize_fixed_op import quantize_fp_op
        model = quantize_fp_op(args.model, args.num_classes, args.attention, args.mlp, args.embedding, args.norm, args.head, args.other, args.quantize_all, args.forward_format, args.forward_wl, args.forward_fl, args.forward_exp, args.forward_man, args.backward_exp, args.backward_man, args.forward_rounding, args.backward_rounding)
    elif args.quant_type == "manual":
        from scripts.quantize_manual import quantize_manual
        model = quantize_manual(args.model, args.num_classes, args.attention, args.mlp, args.embedding, args.norm, args.head, args.other, args.quantize_all)
    elif args.quant_type == "origin":
        print("Loading model from: {args.model}")
        model = load_eva_model(args.model, args.num_classes)
        model.eval()    
    else:
        raise NotImplementedError(f"Quantization type {args.quant_type} not supported")
    
    token = _get_hf_token(args.token)
    print(f"Loading image processor from: {args.model}")
    image_processor = AutoImageProcessor.from_pretrained(args.model, token=token)
    print("Image processor loaded successfully!")
    
    _, val_loader = get_imagenet_dataloaders(
        val_dir=args.val_dir,
        train_dir=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_randaug=False,
        normalize=False,
    )
    
    if hasattr(val_loader.dataset, 'classes'):
        num_classes = len(val_loader.dataset.classes)
        print(f"Detected {num_classes} classes from dataset")
    else:
        num_classes = args.num_classes
        print(f"Using {num_classes} classes from argument")
    
    top1_acc, top5_acc, n_samples = evaluate_model(
        model=model,
        val_loader=val_loader,
        device=args.device,
        num_classes=num_classes,
        image_processor=image_processor
    )
    
    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"  Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"  Total Samples: {n_samples}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()