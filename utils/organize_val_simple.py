#!/usr/bin/env python3
"""
Simple script to organize ImageNet validation images into class folders.
Uses ground truth file and extracts synsets from meta.mat or uses a fallback method.
"""

import shutil
import re
from pathlib import Path


def extract_synsets_from_meta_mat(meta_mat_path: Path):
    """Try to extract synsets from meta.mat using various methods."""
    if not meta_mat_path.exists():
        return None
    
    # Method 1: Try to find synset patterns in binary
    try:
        with open(meta_mat_path, 'rb') as f:
            data = f.read()
        
        # Look for synset patterns (n followed by 8 digits)
        synset_pattern = rb'n\d{8}'
        matches = re.findall(synset_pattern, data)
        
        if matches:
            synsets = list(set([m.decode('ascii', errors='ignore') for m in matches]))
            synsets = [s for s in synsets if len(s) == 9 and s.startswith('n')]
            if len(synsets) >= 1000:
                synsets = sorted(synsets)[:1000]
                print(f"Extracted {len(synsets)} synsets from meta.mat")
                return synsets
    except Exception as e:
        print(f"Could not extract synsets from meta.mat: {e}")
    
    return None


def get_imagenet_synsets_standard():
    """
    Return standard ImageNet synset list in order.
    This is a fallback if meta.mat parsing fails.
    """
    # This is the standard ImageNet-1K synset list in order
    # We'll try to get it from meta.mat first, but this is a backup
    return None  # Will be populated if needed


def organize_validation_images(val_dir: str, devkit_dir: str):
    """Organize validation images into class folders."""
    val_path = Path(val_dir)
    devkit_path = Path(devkit_dir)
    
    if not val_path.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    
    # Get synsets
    synsets = None
    
    # Try to read from synsets.txt if it exists
    synsets_file = devkit_path / "data" / "synsets.txt"
    if synsets_file.exists():
        with open(synsets_file, "r") as f:
            synsets = [line.strip().split()[0] for line in f.readlines() if line.strip()]
        print(f"Loaded {len(synsets)} synsets from synsets.txt")
    
    # Try to extract from meta.mat
    if synsets is None or len(synsets) < 1000:
        meta_path = devkit_path / "data" / "meta.mat"
        if meta_path.exists():
            synsets = extract_synsets_from_meta_mat(meta_path)
    
    if synsets is None or len(synsets) < 1000:
        raise ValueError(
            f"Could not get synset list. Found {len(synsets) if synsets else 0} synsets. "
            "Please ensure meta.mat exists in devkit or create synsets.txt manually."
        )
    
    # Load ground truth labels
    val_groundtruth_file = devkit_path / "data" / "ILSVRC2012_validation_ground_truth.txt"
    if not val_groundtruth_file.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {val_groundtruth_file}"
        )
    
    with open(val_groundtruth_file, "r") as f:
        val_labels = [int(line.strip()) - 1 for line in f.readlines()]  # Convert to 0-indexed
    
    # Get all validation images
    val_images = sorted(
        [f for f in val_path.iterdir() if f.is_file() and f.suffix.upper() in (".JPEG", ".JPG")],
        key=lambda x: int(re.search(r'\d+', x.name).group()) if re.search(r'\d+', x.name) else 0
    )
    
    if len(val_images) != len(val_labels):
        raise ValueError(
            f"Mismatch: found {len(val_images)} validation images but {len(val_labels)} labels."
        )
    
    if len(synsets) != 1000:
        raise ValueError(
            f"Expected 1000 synsets, but found {len(synsets)}."
        )
    
    print(f"Organizing {len(val_images)} validation images into {len(synsets)} class folders...")
    
    # Create class directories and move images
    moved = 0
    for img_path, label_idx in zip(val_images, val_labels):
        if label_idx >= len(synsets):
            raise ValueError(f"Label index {label_idx} is out of range for {len(synsets)} synsets")
        
        class_name = synsets[label_idx]
        class_dir = val_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Move image to class folder
        dest_path = class_dir / img_path.name
        if not dest_path.exists():  # Don't move if already there
            shutil.move(str(img_path), str(dest_path))
            moved += 1
        
        if (moved + 1) % 1000 == 0:
            print(f"Processed {moved + 1} images...")
    
    print(f"Successfully organized {moved} validation images into class folders!")
    print(f"Validation images are now in: {val_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Organize ImageNet validation images into class folders"
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        required=True,
        help="Directory containing flat validation images",
    )
    parser.add_argument(
        "--devkit-dir",
        type=str,
        required=True,
        help="Path to devkit directory",
    )
    
    args = parser.parse_args()
    
    organize_validation_images(
        val_dir=args.val_dir,
        devkit_dir=args.devkit_dir,
    )

