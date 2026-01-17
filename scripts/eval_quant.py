import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def load_quantized_model(model_path: str, device: str) -> nn.Module:
    model_path = Path(model_path)
    
    print(f"Loading quantized model from: {model_path}")
    model = torch.load(model_path, map_location=device)
    
    model.eval()
    model.to(device)
    print("Model loaded successfully!")
    return model


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    num_classes: int = 1000
) -> Tuple[float, float, int]:
    top1_sum = 0.0
    top5_sum = 0.0
    n_samples = 0

    model.eval()
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader, desc="Evaluating", unit="batch")):
            batch_size = targets.size(0)
            
            images = images.to(device)
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
        "--img-size",
        type=int,
        default=224,
        help="Input image size (default: 224)"
    )
    parser.add_argument(
        "--eval-resize",
        type=int,
        default=256,
        help="Resize size before center crop (default: 256)"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of classes (default: 1000 for ImageNet)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
            
    model = load_quantized_model(args.model, args.device)
    
    _, val_loader = get_imagenet_dataloaders(
        val_dir=args.val_dir,
        train_dir=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        eval_resize=args.eval_resize,
        use_randaug=False,
        normalize=True,  # Normalize in dataloader since we're not using pipeline
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
        num_classes=num_classes
    )
    
    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"  Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"  Total Samples: {n_samples}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()