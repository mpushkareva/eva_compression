# image_dataloaders.py
from typing import Tuple, Optional, Dict
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch
import os


# Standard ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Mapping from CIFAR-10 class names to ImageNet class indices
# These are commonly used mappings based on WordNet synsets
CIFAR10_TO_IMAGENET_MAPPING: Dict[str, int] = {
    "airplane": 404,   # n02690373: airliner
    "automobile": 511, # n02930766: convertible
    "bird": 80,        # n01503061: bird (common bird)
    "cat": 281,        # n02123045: tabby cat
    "deer": 345,       # n02408429: elk
    "dog": 151,        # n02084071: dog (common dog)
    "frog": 30,        # n01641577: tree frog
    "horse": 340,      # n02389026: horse
    "ship": 510,       # n03095699: container ship
    "truck": 609,      # n03417042: pickup truck
}


def get_image_train_transform(
    img_size: int = 224,
    use_randaug: bool = True,
) -> transforms.Compose:
    """Return training transforms for image classification (CIFAR, ImageNet, etc.)."""
    t = [
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
    ]
    if use_randaug:
        # RandAugment from torchvision (requires torchvision >= 0.9)
        t.append(transforms.RandAugment())
    t.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(t)


def get_image_val_transform(
    img_size: int = 224,
    eval_resize: int = 256,
    normalize: bool = True,
) -> transforms.Compose:
    """Return validation / test transforms: resize to eval_resize, center-crop to img_size.
    
    Args:
        img_size: Target image size after center crop
        eval_resize: Resize size before center crop
        normalize: If False, skip normalization (useful when pipeline handles normalization)
    """
    t = [
        transforms.Resize(eval_resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ]
    if normalize:
        t.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(t)


class CIFAR10WithImageNetLabels(Dataset):
    """
    Wrapper for CIFAR-10 dataset that maps CIFAR-10 labels to ImageNet class indices.
    This allows using ImageNet-trained models directly on CIFAR-10 data.
    """
    def __init__(
        self,
        cifar_dataset: datasets.CIFAR10,
        label_mapping: Dict[str, int],
    ):
        """
        Args:
            cifar_dataset: Original CIFAR-10 dataset
            label_mapping: Dictionary mapping CIFAR-10 class names to ImageNet class indices
        """
        self.cifar_dataset = cifar_dataset
        self.label_mapping = label_mapping
        # Create mapping from CIFAR-10 class index to ImageNet class index
        self.cifar_idx_to_imagenet_idx = {}
        for cifar_idx, class_name in enumerate(cifar_dataset.classes):
            if class_name in label_mapping:
                self.cifar_idx_to_imagenet_idx[cifar_idx] = label_mapping[class_name]
            else:
                raise ValueError(f"CIFAR-10 class '{class_name}' not found in label_mapping")
    
    def __len__(self):
        return len(self.cifar_dataset)
    
    def __getitem__(self, idx):
        image, cifar_label = self.cifar_dataset[idx]
        # Map CIFAR-10 label to ImageNet label
        imagenet_label = self.cifar_idx_to_imagenet_idx[cifar_label]
        return image, imagenet_label
    
    @property
    def classes(self):
        """Return ImageNet class indices as 'classes' for compatibility."""
        return list(self.label_mapping.values())


def get_cifar_dataloaders(
    root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    img_size: int = 224,
    use_randaug: bool = True,
    cifar_version: str = "cifar10",
    use_imagenet_labels: bool = False,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders for CIFAR10 or CIFAR100.

    CIFAR images are 32x32; here we upsample to 224x224 by default to match ImageNet-sized backbones.
    
    Args:
        root: Root directory for CIFAR dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        img_size: Target image size (will be upsampled from 32x32)
        use_randaug: Whether to use RandAugment for training
        cifar_version: "cifar10" or "cifar100"
        use_imagenet_labels: If True, map CIFAR-10 labels to ImageNet class indices
                           (only works for cifar10, not cifar100)
    """
    assert cifar_version in ("cifar10", "cifar100")
    
    if use_imagenet_labels and cifar_version != "cifar10":
        raise ValueError("ImageNet label mapping is only supported for CIFAR-10, not CIFAR-100")
    
    train_transform = get_image_train_transform(img_size=img_size, use_randaug=use_randaug)
    val_transform = get_image_val_transform(img_size=img_size, normalize=normalize)

    DatasetClass = datasets.CIFAR10 if cifar_version == "cifar10" else datasets.CIFAR100

    train_ds_base = DatasetClass(root=root, train=True, transform=train_transform, download=True)
    val_ds_base = DatasetClass(root=root, train=False, transform=val_transform, download=True)
    
    # Wrap with ImageNet label mapping if requested
    if use_imagenet_labels:
        train_ds = CIFAR10WithImageNetLabels(train_ds_base, CIFAR10_TO_IMAGENET_MAPPING)
        val_ds = CIFAR10WithImageNetLabels(val_ds_base, CIFAR10_TO_IMAGENET_MAPPING)
    else:
        train_ds = train_ds_base
        val_ds = val_ds_base

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def get_imagenet_dataloaders(
    val_dir: str,
    train_dir: Optional[str] = None,
    batch_size: int = 256,
    num_workers: int = 8,
    img_size: int = 224,
    eval_resize: int = 256,
    use_randaug: bool = True,
    normalize: bool = True,
) -> Tuple[Optional[DataLoader], DataLoader]:
    """
    ImageNet dataloaders using torchvision.datasets.ImageNet.
    
    The ImageNet dataset class expects a root directory containing the ImageNet dataset
    with 'train' and 'val' subdirectories, and the devkit should be at root/ILSVRC2012_devkit_t12/.
    This function infers the root directory from the provided val_dir and handles devkit location.
    
    Args:
        val_dir: Path to validation directory (required). Root will be inferred as parent directory.
        train_dir: Path to training directory (optional, if None, train_loader will be None)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        img_size: Target image size
        eval_resize: Resize size before center crop for validation
        use_randaug: Whether to use RandAugment for training
        normalize: If False, skip normalization
    
    Returns:
        Tuple of (train_loader, val_loader). train_loader is None if train_dir is None.
    """
    # Infer root directory from val_dir (assuming structure: root/val/)
    # If val_dir ends with 'val', use its parent; otherwise use val_dir itself as root
    val_dir_normalized = val_dir.rstrip('/')
    if os.path.basename(val_dir_normalized) == 'val':
        root = os.path.dirname(val_dir_normalized)
    else:
        # If val_dir doesn't end with 'val', assume it's the root directory
        root = val_dir_normalized
    
    # Handle devkit location: torchvision.datasets.ImageNet expects devkit at root/ILSVRC2012_devkit_t12/
    # But it might be at root/devkit/ILSVRC2012_devkit_t12/
    devkit_expected = os.path.join(root, 'ILSVRC2012_devkit_t12')
    devkit_alternative = os.path.join(root, 'devkit', 'ILSVRC2012_devkit_t12')
    
    # Create symlink if devkit is in alternative location and expected location doesn't exist
    if not os.path.exists(devkit_expected) and os.path.exists(devkit_alternative):
        try:
            # Use relative path for symlink to make it more portable
            rel_path = os.path.relpath(devkit_alternative, root)
            os.symlink(rel_path, devkit_expected)
        except (OSError, FileExistsError):
            # Symlink might already exist or we don't have permission, try to continue
            # If it's a broken symlink, try to remove and recreate
            if os.path.islink(devkit_expected) and not os.path.exists(devkit_expected):
                try:
                    os.unlink(devkit_expected)
                    rel_path = os.path.relpath(devkit_alternative, root)
                    os.symlink(rel_path, devkit_expected)
                except OSError:
                    pass
    
    val_transform = get_image_val_transform(img_size=img_size, eval_resize=eval_resize, normalize=normalize)
    val_ds = datasets.ImageNet(root=root, split='val', transform=val_transform)
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    train_loader = None
    if train_dir is not None:
        train_transform = get_image_train_transform(img_size=img_size, use_randaug=use_randaug)
        train_ds = datasets.ImageNet(root=root, split='train', transform=train_transform)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    return train_loader, val_loader
