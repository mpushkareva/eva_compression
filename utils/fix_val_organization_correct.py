#!/usr/bin/env python3
"""
Correctly reorganize ImageNet validation images.
The ground truth file expects images sorted alphabetically by filename.
Line 1 of ground truth = label for first image when sorted alphabetically.
"""

import shutil
from pathlib import Path
import argparse


def fix_validation_organization(val_dir: str, devkit_dir: str):
    """
    Reorganize validation images correctly based on ground truth.
    
    The key insight: Ground truth file line N corresponds to the Nth image
    when sorted alphabetically by filename.
    """
    val_path = Path(val_dir)
    devkit_path = Path(devkit_dir)
    
    if not val_path.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    
    # Load correct synsets
    try:
        from create_synsets_from_standard import STANDARD_IMAGENET_SYNSETS
        original_synsets = STANDARD_IMAGENET_SYNSETS
        print(f"Using standard ImageNet synset list ({len(original_synsets)} synsets)")
    except ImportError:
        synsets_file = devkit_path / "data" / "synsets.txt"
        if synsets_file.exists():
            with open(synsets_file, "r") as f:
                original_synsets = [line.strip().split()[0] for line in f.readlines() if line.strip()]
            print(f"Loaded {len(original_synsets)} synsets from synsets.txt")
        else:
            raise FileNotFoundError(f"synsets.txt not found: {synsets_file}")
    
    valid_synsets_set = set(original_synsets)
    
    # Pad synsets to 1000
    synsets = original_synsets.copy()
    if len(synsets) < 1000:
        for i in range(len(synsets), 1000):
            synsets.append(f"n{i:08d}")
    
    print(f"Using {len(synsets)} synsets")
    
    # Load ground truth labels
    val_groundtruth_file = devkit_path / "data" / "ILSVRC2012_validation_ground_truth.txt"
    if not val_groundtruth_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {val_groundtruth_file}")
    
    with open(val_groundtruth_file, "r") as f:
        val_labels = [int(line.strip()) - 1 for line in f.readlines()]  # Convert to 0-indexed
    
    print(f"Loaded {len(val_labels)} ground truth labels")
    
    # Collect ALL images from all subdirectories
    all_images = []
    for subdir in val_path.iterdir():
        if subdir.is_dir():
            for img_file in subdir.glob("*.JPEG"):
                all_images.append(img_file)
            for img_file in subdir.glob("*.JPG"):
                all_images.append(img_file)
    
    # Also check root directory
    for img_file in val_path.glob("*.JPEG"):
        all_images.append(img_file)
    for img_file in val_path.glob("*.JPG"):
        all_images.append(img_file)
    
    # CRITICAL: Sort alphabetically by filename (this is what ground truth expects)
    all_images.sort(key=lambda x: x.name)
    
    print(f"Found {len(all_images)} validation images")
    
    if len(all_images) != len(val_labels):
        print(f"Warning: Found {len(all_images)} images but {len(val_labels)} labels")
        min_len = min(len(all_images), len(val_labels))
        all_images = all_images[:min_len]
        val_labels = val_labels[:min_len]
        print(f"Using first {min_len} images and labels")
    
    # Create mapping: image -> correct synset based on its position in sorted list
    image_to_synset = {}
    skipped_count = 0
    
    for idx, (img_path, label_idx) in enumerate(zip(all_images, val_labels)):
        if label_idx >= len(synsets):
            print(f"Warning: Label index {label_idx} out of range, skipping {img_path.name}")
            skipped_count += 1
            continue
        
        correct_synset = synsets[label_idx]
        
        # Skip placeholder synsets
        if correct_synset not in valid_synsets_set:
            if skipped_count < 5:
                print(f"Warning: Skipping {img_path.name} (label {label_idx} -> placeholder {correct_synset})")
            skipped_count += 1
            continue
        
        image_to_synset[img_path] = correct_synset
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} images due to missing synsets or out-of-range labels")
    
    print(f"\nReorganizing {len(image_to_synset)} images...")
    
    # Reorganize images
    moved_count = 0
    already_correct = 0
    
    for img_path, correct_synset in image_to_synset.items():
        correct_dir = val_path / correct_synset
        correct_dir.mkdir(exist_ok=True)
        dest_path = correct_dir / img_path.name
        
        # Check if already in correct location
        if img_path == dest_path:
            already_correct += 1
            continue
        
        # Move the image
        if dest_path.exists():
            # Remove duplicate
            dest_path.unlink()
        
        shutil.move(str(img_path), str(dest_path))
        moved_count += 1
        
        # Remove empty source directory if it's not a valid synset
        source_dir = img_path.parent
        if source_dir != val_path and source_dir.name not in valid_synsets_set:
            try:
                if not any(source_dir.iterdir()):
                    source_dir.rmdir()
            except OSError:
                pass
        
        if (moved_count + already_correct) % 5000 == 0:
            print(f"  Processed {moved_count + already_correct} images...")
    
    print(f"\nReorganization complete!")
    print(f"  Moved: {moved_count} images")
    print(f"  Already correct: {already_correct} images")
    print(f"  Total processed: {moved_count + already_correct} images")
    
    # Clean up empty incorrect directories
    print("\nCleaning up empty directories...")
    cleaned = 0
    for subdir in val_path.iterdir():
        if subdir.is_dir() and subdir.name not in valid_synsets_set:
            try:
                if not any(subdir.iterdir()):
                    subdir.rmdir()
                    cleaned += 1
            except OSError:
                pass
    
    if cleaned > 0:
        print(f"Removed {cleaned} empty incorrect directories")
    
    # Verify a few samples
    print("\nVerifying first 10 images...")
    sorted_images = sorted([img for img in val_path.rglob("*.JPEG")], key=lambda x: x.name)[:10]
    for i, img_path in enumerate(sorted_images):
        label_idx = val_labels[i]
        expected_synset = synsets[label_idx] if label_idx < len(synsets) and synsets[label_idx] in valid_synsets_set else "PLACEHOLDER"
        actual_synset = img_path.parent.name
        match = "✓" if actual_synset == expected_synset else "✗"
        print(f"  {i+1:2d}. {img_path.name:40s} Label: {label_idx:3d} Expected: {expected_synset:12s} Actual: {actual_synset:12s} {match}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Correctly reorganize ImageNet validation images"
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        required=True,
        help="Directory containing validation images",
    )
    parser.add_argument(
        "--devkit-dir",
        type=str,
        required=True,
        help="Path to devkit directory",
    )
    
    args = parser.parse_args()
    
    fix_validation_organization(
        val_dir=args.val_dir,
        devkit_dir=args.devkit_dir,
    )

