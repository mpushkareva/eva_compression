#!/usr/bin/env python3
"""
Correctly reorganize ImageNet validation images.
The ground truth file line N corresponds to image ILSVRC2012_val_00000XXX.JPEG
where XXX is the line number (1-indexed, zero-padded to 8 digits).
"""

import shutil
import re
from pathlib import Path
import argparse


def fix_validation_organization(val_dir: str, devkit_dir: str):
    """
    Reorganize validation images correctly based on ground truth.
    
    Key: Ground truth line N (1-indexed) = image ILSVRC2012_val_00000XXX.JPEG
    where XXX = N (zero-padded to 8 digits).
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
    
    # Create mapping: image_filename -> correct_synset
    # Line 1 (index 0) = ILSVRC2012_val_00000001.JPEG -> synsets[val_labels[0]]
    # Line 2 (index 1) = ILSVRC2012_val_00000002.JPEG -> synsets[val_labels[1]]
    # etc.
    image_to_synset = {}
    skipped_count = 0
    
    for line_num, label_idx in enumerate(val_labels):
        img_num = line_num + 1  # 1-indexed
        img_name = f"ILSVRC2012_val_{img_num:08d}.JPEG"
        
        if label_idx >= len(synsets):
            skipped_count += 1
            continue
        
        correct_synset = synsets[label_idx]
        
        # Skip placeholder synsets
        if correct_synset not in valid_synsets_set:
            skipped_count += 1
            continue
        
        image_to_synset[img_name] = correct_synset
    
    print(f"Created mapping for {len(image_to_synset)} images")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} images (out of range or placeholder synsets)")
    
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
    
    print(f"Found {len(all_images)} validation images")
    
    # Reorganize images based on filename mapping
    moved_count = 0
    already_correct = 0
    not_in_mapping = 0
    
    for img_path in all_images:
        img_name = img_path.name
        correct_synset = image_to_synset.get(img_name)
        
        if correct_synset is None:
            # Image not in mapping (might be one of the skipped ones)
            not_in_mapping += 1
            continue
        
        correct_dir = val_path / correct_synset
        correct_dir.mkdir(exist_ok=True)
        dest_path = correct_dir / img_name
        
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
    print(f"  Not in mapping: {not_in_mapping} images")
    print(f"  Total processed: {moved_count + already_correct + not_in_mapping} images")
    
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
    print("\nVerifying sample images...")
    test_images = [
        "ILSVRC2012_val_00000001.JPEG",
        "ILSVRC2012_val_00000002.JPEG",
        "ILSVRC2012_val_00000003.JPEG",
        "ILSVRC2012_val_00000010.JPEG",
        "ILSVRC2012_val_00000100.JPEG",
    ]
    
    for img_name in test_images:
        expected_synset = image_to_synset.get(img_name, "NOT_IN_MAPPING")
        if expected_synset != "NOT_IN_MAPPING":
            # Find the image
            img_path = val_path / expected_synset / img_name
            if img_path.exists():
                actual_synset = img_path.parent.name
                match = "✓" if actual_synset == expected_synset else "✗"
                line_num = int(re.search(r'(\d+)', img_name).group(1))
                label_idx = val_labels[line_num - 1]
                print(f"  {img_name:40s} Line: {line_num:5d} Label: {label_idx:3d} Expected: {expected_synset:12s} Actual: {actual_synset:12s} {match}")
            else:
                print(f"  {img_name:40s} Expected: {expected_synset:12s} NOT FOUND ✗")
        else:
            print(f"  {img_name:40s} NOT_IN_MAPPING")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Correctly reorganize ImageNet validation images using filename-based mapping"
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

