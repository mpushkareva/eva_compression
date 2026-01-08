#!/usr/bin/env python3
"""
Fix incorrectly organized ImageNet validation images.
Reorganizes images from incorrectly named folders (like n00000995) into correct synset folders.
"""

import shutil
import re
from pathlib import Path
import argparse


def fix_validation_organization(val_dir: str, devkit_dir: str):
    """
    Reorganize validation images from incorrect folders to correct synset folders.
    
    Args:
        val_dir: Directory containing validation images (may be in incorrect folders)
        devkit_dir: Path to devkit directory (for synsets and ground truth)
    """
    val_path = Path(val_dir)
    devkit_path = Path(devkit_dir)
    
    if not val_path.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    
    # Load correct synsets - try standard list first, then fallback to synsets.txt
    original_synsets = None
    try:
        from create_synsets_from_standard import STANDARD_IMAGENET_SYNSETS
        original_synsets = STANDARD_IMAGENET_SYNSETS
        print(f"Using standard ImageNet synset list ({len(original_synsets)} synsets)")
    except ImportError:
        # Fallback to synsets.txt
        synsets_file = devkit_path / "data" / "synsets.txt"
        if synsets_file.exists():
            with open(synsets_file, "r") as f:
                original_synsets = [line.strip().split()[0] for line in f.readlines() if line.strip()]
            print(f"Loaded {len(original_synsets)} synsets from synsets.txt")
        else:
            raise FileNotFoundError(f"synsets.txt not found: {synsets_file}")
    
    # Create set of valid synsets for checking
    valid_synsets_set = set(original_synsets)
    
    # Pad synsets to 1000 if needed (some datasets have 995 synsets but labels 1-1000)
    synsets = original_synsets.copy()
    if len(synsets) < 1000:
        print(f"Warning: Found {len(synsets)} synsets, but ground truth uses 1-1000. Padding with placeholders...")
        for i in range(len(synsets), 1000):
            synsets.append(f"n{i:08d}")  # Placeholder for missing synsets
    elif len(synsets) > 1000:
        synsets = synsets[:1000]
        print(f"Warning: Found {len(synsets)} synsets, using first 1000")
    
    print(f"Using {len(synsets)} synsets")
    print(f"First 5 synsets: {synsets[:5]}")
    
    # Load ground truth labels
    val_groundtruth_file = devkit_path / "data" / "ILSVRC2012_validation_ground_truth.txt"
    if not val_groundtruth_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {val_groundtruth_file}")
    
    with open(val_groundtruth_file, "r") as f:
        val_labels = [int(line.strip()) - 1 for line in f.readlines()]  # Convert to 0-indexed
    
    print(f"Loaded {len(val_labels)} ground truth labels")
    
    # Collect all images from all subdirectories (including incorrectly named ones)
    all_images = []
    for subdir in val_path.iterdir():
        if subdir.is_dir():
            for img_file in subdir.glob("*.JPEG"):
                all_images.append(img_file)
            for img_file in subdir.glob("*.JPG"):
                all_images.append(img_file)
    
    # Also check root directory for any remaining images
    for img_file in val_path.glob("*.JPEG"):
        all_images.append(img_file)
    for img_file in val_path.glob("*.JPG"):
        all_images.append(img_file)
    
    # Sort images by their number in filename (ILSVRC2012_val_00000001.JPEG -> 1)
    def extract_number(path):
        match = re.search(r'(\d+)', path.name)
        return int(match.group(1)) if match else 0
    
    all_images.sort(key=extract_number)
    
    print(f"Found {len(all_images)} validation images")
    
    if len(all_images) != len(val_labels):
        print(f"Warning: Found {len(all_images)} images but {len(val_labels)} labels")
        print("Using only the first {len(val_labels)} images")
        all_images = all_images[:len(val_labels)]
    
    # Create mapping: image index -> correct synset
    image_to_synset = {}
    skipped_labels = 0
    for idx, (img_path, label_idx) in enumerate(zip(all_images, val_labels)):
        if label_idx >= len(synsets):
            print(f"Warning: Label index {label_idx} is out of range for {len(synsets)} synsets, skipping image {img_path.name}")
            skipped_labels += 1
            continue
        correct_synset = synsets[label_idx]
        # Skip placeholder synsets (those not in the original valid synsets)
        if correct_synset not in valid_synsets_set:
            if skipped_labels < 5:  # Only print first few warnings
                print(f"Warning: Image {img_path.name} has label {label_idx} which maps to placeholder synset {correct_synset}, skipping")
            skipped_labels += 1
            continue
        image_to_synset[img_path] = correct_synset
    
    if skipped_labels > 0:
        print(f"Warning: Skipped {skipped_labels} images due to missing synsets")
    
    # Reorganize images
    moved_count = 0
    skipped_count = 0
    
    for img_path, correct_synset in image_to_synset.items():
        correct_dir = val_path / correct_synset
        correct_dir.mkdir(exist_ok=True)
        dest_path = correct_dir / img_path.name
        
        # Only move if:
        # 1. Image is not already in the correct location
        # 2. Or it's in a different (incorrect) folder
        if img_path == dest_path:
            # Already in correct location
            skipped_count += 1
            continue
        
        # Move the image
        if dest_path.exists():
            print(f"Warning: Destination already exists: {dest_path}")
            # Remove the duplicate
            dest_path.unlink()
        
        shutil.move(str(img_path), str(dest_path))
        moved_count += 1
        
        # Remove empty source directory if it's not a valid synset
        source_dir = img_path.parent
        if source_dir != val_path and source_dir.name not in synsets:
            try:
                if not any(source_dir.iterdir()):  # Directory is empty
                    source_dir.rmdir()
            except OSError:
                pass  # Directory not empty or other error
        
        if (moved_count + skipped_count) % 1000 == 0:
            print(f"Processed {moved_count + skipped_count} images... (moved: {moved_count}, skipped: {skipped_count})")
    
    print(f"\nReorganization complete!")
    print(f"  Moved: {moved_count} images")
    print(f"  Skipped (already correct): {skipped_count} images")
    print(f"  Total processed: {moved_count + skipped_count} images")
    
    # Clean up empty incorrect directories
    print("\nCleaning up empty incorrect directories...")
    cleaned = 0
    for subdir in val_path.iterdir():
        if subdir.is_dir() and subdir.name not in synsets:
            try:
                if not any(subdir.iterdir()):  # Directory is empty
                    subdir.rmdir()
                    cleaned += 1
            except OSError:
                pass
    
    if cleaned > 0:
        print(f"Removed {cleaned} empty incorrect directories")
    
    print(f"\nValidation images are now correctly organized in: {val_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix incorrectly organized ImageNet validation images"
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        required=True,
        help="Directory containing validation images (may be in incorrect folders)",
    )
    parser.add_argument(
        "--devkit-dir",
        type=str,
        required=True,
        help="Path to devkit directory (for synsets and ground truth)",
    )
    
    args = parser.parse_args()
    
    fix_validation_organization(
        val_dir=args.val_dir,
        devkit_dir=args.devkit_dir,
    )

