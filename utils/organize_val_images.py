#!/usr/bin/env python3
"""
Organize existing ImageNet validation images into class folders.

This script organizes validation images that are already extracted but not yet
organized into class-specific folders.
"""

import shutil
import re
from pathlib import Path
import argparse


def get_synsets_from_meta_mat(devkit_dir: Path):
    """Extract synset names from meta.mat file."""
    meta_path = devkit_dir / "data" / "meta.mat"
    if not meta_path.exists():
        return None
    
    # Try scipy first
    try:
        import scipy.io
        mat = scipy.io.loadmat(str(meta_path), squeeze_me=True)
        # The synsets are typically in 'synsets' or 'ILSVRC2012_ID' field
        if 'synsets' in mat:
            synsets_data = mat['synsets']
            synsets = []
            
            # Handle different data structures
            if hasattr(synsets_data, '__iter__') and not isinstance(synsets_data, (str, bytes)):
                for item in synsets_data:
                    if isinstance(item, dict):
                        if 'ILSVRC2012_ID' in item:
                            synsets.append(item['ILSVRC2012_ID'])
                        elif 'WNID' in item:
                            synsets.append(item['WNID'])
                    elif isinstance(item, str):
                        synsets.append(item)
                    elif hasattr(item, '__dict__'):
                        # Try to get ID from object attributes
                        if hasattr(item, 'ILSVRC2012_ID'):
                            synsets.append(item.ILSVRC2012_ID)
                        elif hasattr(item, 'WNID'):
                            synsets.append(item.WNID)
            
            if synsets and len(synsets) >= 1000:
                print(f"Extracted {len(synsets)} synsets from meta.mat using scipy")
                return synsets[:1000]
    except ImportError:
        print("Warning: scipy not available. Trying alternative method...")
        # Try to extract synsets using a simple binary parser
        return get_synsets_from_meta_mat_simple(devkit_dir)
    except Exception as e:
        print(f"Warning: Could not load synsets from meta.mat with scipy: {e}")
        # Try simple parser as fallback
        return get_synsets_from_meta_mat_simple(devkit_dir)
    
    # If scipy method didn't return synsets, try simple parser
    return get_synsets_from_meta_mat_simple(devkit_dir)


def get_synsets_from_meta_mat_simple(devkit_dir: Path):
    """Extract synset names from meta.mat using a simple approach."""
    meta_path = devkit_dir / "data" / "meta.mat"
    if not meta_path.exists():
        return None
    
    # Try to find synset patterns in the binary file
    # ImageNet synsets are 9-character strings like "n01440764"
    try:
        import re
        with open(meta_path, 'rb') as f:
            data = f.read()
        
        # Method 1: Look for synset patterns in binary (n followed by 8 digits)
        synset_pattern = rb'n\d{8}'
        matches = re.findall(synset_pattern, data)
        if matches:
            synsets = list(set([m.decode('ascii', errors='ignore') for m in matches]))
            synsets = [s for s in synsets if len(s) == 9 and s.startswith('n')]
            if len(synsets) >= 1000:
                synsets = sorted(synsets)[:1000]
                print(f"Extracted {len(synsets)} synsets from meta.mat using simple parser")
                return synsets
        
        # Method 2: Try to decode as text and look for patterns
        try:
            text_data = data.decode('latin-1', errors='ignore')  # Latin-1 preserves all bytes
            text_matches = re.findall(r'n\d{8}', text_data)
            if text_matches:
                synsets = sorted(set(text_matches))
                if len(synsets) >= 1000:
                    synsets = synsets[:1000]
                    print(f"Extracted {len(synsets)} synsets from meta.mat using text decoding")
                    return synsets
        except:
            pass
            
    except Exception as e:
        print(f"Could not extract synsets using simple parser: {e}")
    
    return None


def get_synsets_from_train_dir(train_dir: Path):
    """Extract synset names from training directory structure."""
    if not train_dir.exists():
        return None
    
    synsets = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if len(synsets) == 1000:
        print(f"Found {len(synsets)} class folders in training directory")
        return synsets
    return None


def organize_validation_images(
    val_dir: str,
    devkit_dir: str = None,
    train_dir: str = None,
):
    """
    Organize validation images into class folders.
    
    Args:
        val_dir: Directory containing flat validation images
        devkit_dir: Path to devkit directory (for synsets and ground truth)
        train_dir: Path to training directory (alternative source for synsets)
    """
    val_path = Path(val_dir)
    if not val_path.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    
    # Find synsets
    synsets = None
    
    if devkit_dir:
        devkit_path = Path(devkit_dir)
        # Try to get synsets from meta.mat
        synsets = get_synsets_from_meta_mat(devkit_path)
        
        # If that fails, try to read from synsets.txt if it exists
        if synsets is None:
            synsets_file = devkit_path / "data" / "synsets.txt"
            if synsets_file.exists():
                with open(synsets_file, "r") as f:
                    synsets = [line.strip().split()[0] for line in f.readlines()]
    
    # Fallback: get synsets from training directory
    if synsets is None and train_dir:
        train_path = Path(train_dir)
        synsets = get_synsets_from_train_dir(train_path)
    
    if synsets is None:
        raise ValueError(
            "Could not determine synset names. Please provide either:\n"
            "  - devkit_dir with meta.mat or synsets.txt\n"
            "  - train_dir with organized class folders"
        )
    
    # Load ground truth labels
    if devkit_dir:
        devkit_path = Path(devkit_dir)
        val_groundtruth_file = devkit_path / "data" / "ILSVRC2012_validation_ground_truth.txt"
        if val_groundtruth_file.exists():
            with open(val_groundtruth_file, "r") as f:
                val_labels = [int(line.strip()) - 1 for line in f.readlines()]  # Convert to 0-indexed
        else:
            raise FileNotFoundError(
                f"Ground truth file not found: {val_groundtruth_file}\n"
                "This file is required to organize validation images."
            )
    else:
        raise ValueError("devkit_dir is required to get ground truth labels")
    
    # Get all validation images (only from root directory, not subdirectories)
    val_images = sorted(
        [f for f in val_path.iterdir() if f.is_file() and f.suffix.upper() in (".JPEG", ".JPG")],
        key=lambda x: int(re.search(r'\d+', x.name).group()) if re.search(r'\d+', x.name) else 0
    )
    
    if len(val_images) != len(val_labels):
        print(f"Warning: Found {len(val_images)} validation images in root directory but {len(val_labels)} labels.")
        print("Some images may have already been organized. Continuing with available images...")
        # Use only the images we have
        val_labels = val_labels[:len(val_images)]
    
    # Ensure we have at least 1000 synsets (pad with placeholders if needed)
    if len(synsets) < 1000:
        print(f"Warning: Found {len(synsets)} synsets, expected 1000. Adding placeholder synsets...")
        # Add placeholder synsets for missing indices
        for i in range(len(synsets), 1000):
            synsets.append(f"n{i:08d}")  # Placeholder format
    elif len(synsets) > 1000:
        print(f"Warning: Found {len(synsets)} synsets, expected 1000. Using first 1000...")
        synsets = synsets[:1000]
    
    print(f"Organizing {len(val_images)} validation images into {len(synsets)} class folders...")
    
    # Create class directories and move images
    for img_path, label_idx in zip(val_images, val_labels):
        if label_idx >= len(synsets):
            raise ValueError(f"Label index {label_idx} is out of range for {len(synsets)} synsets")
        
        class_name = synsets[label_idx]
        class_dir = val_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Move image to class folder
        shutil.move(str(img_path), str(class_dir / img_path.name))
    
    print(f"Successfully organized validation images into class folders!")
    print(f"Validation images are now in: {val_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Organize existing ImageNet validation images into class folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # With devkit directory
  python organize_val_images.py \\
    --val-dir ./data/imagenet/val \\
    --devkit-dir ./data/imagenet/devkit/ILSVRC2012_devkit_t12

  # With devkit and training directory (as fallback)
  python organize_val_images.py \\
    --val-dir ./data/imagenet/val \\
    --devkit-dir ./data/imagenet/devkit/ILSVRC2012_devkit_t12 \\
    --train-dir ./data/imagenet/train
        """
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
        default=None,
        help="Path to devkit directory (required for ground truth labels)",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=None,
        help="Path to training directory (optional, used as fallback for synsets)",
    )
    
    args = parser.parse_args()
    
    organize_validation_images(
        val_dir=args.val_dir,
        devkit_dir=args.devkit_dir,
        train_dir=args.train_dir,
    )


if __name__ == "__main__":
    main()

