#!/usr/bin/env python3
"""
Download and organize ImageNet-1k dataset for use with the ImageNet dataloader.

ImageNet-1k requires registration at: https://www.image-net.org/download.php
After registration, you can download:
- ILSVRC2012_img_train.tar (training images, ~138GB)
- ILSVRC2012_img_val.tar (validation images, ~6.3GB)
- ILSVRC2012_devkit_t12.tar.gz (development kit with labels)

This script will:
1. Extract training images into class-specific folders
2. Extract and organize validation images into class-specific folders
3. Create the expected directory structure for the dataloader
"""

import os
import sys
import tarfile
import shutil
from pathlib import Path
from typing import Optional
import argparse


def extract_train_tar(train_tar_path: str, output_dir: str) -> None:
    """
    Extract ImageNet training tar file.
    
    The training tar contains 1000 tar files (one per class).
    Each class tar needs to be extracted into its own folder.
    """
    print(f"Extracting training images from {train_tar_path}...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # First, extract the main tar to get individual class tars
    temp_dir = output_path / "_temp_train"
    temp_dir.mkdir(exist_ok=True)
    
    print("Step 1: Extracting main training tar...")
    with tarfile.open(train_tar_path, "r") as main_tar:
        main_tar.extractall(temp_dir)
    
    # Now extract each class tar into its own folder
    print("Step 2: Extracting class-specific tars...")
    class_tars = sorted(temp_dir.glob("*.tar"))
    total_classes = len(class_tars)
    
    for idx, class_tar in enumerate(class_tars, 1):
        class_name = class_tar.stem  # e.g., "n01440764"
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        with tarfile.open(class_tar, "r") as tar:
            tar.extractall(class_dir)
        
        if idx % 100 == 0 or idx == total_classes:
            print(f"  Extracted {idx}/{total_classes} classes...")
    
    # Clean up temp directory
    shutil.rmtree(temp_dir)
    print(f"Training images extracted to {output_dir}")


def extract_val_tar(val_tar_path: str, output_dir: str, devkit_dir: Optional[str] = None) -> None:
    """
    Extract ImageNet validation tar file and organize by class.
    
    Validation images are all in one folder. We need to move them
    into class-specific folders using the devkit metadata.
    """
    print(f"Extracting validation images from {val_tar_path}...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # First, extract all validation images to a temp folder
    temp_val_dir = output_path / "_temp_val"
    temp_val_dir.mkdir(exist_ok=True)
    
    print("Step 1: Extracting validation tar...")
    with tarfile.open(val_tar_path, "r") as tar:
        tar.extractall(temp_val_dir)
    
    # Find the actual images directory (might be nested)
    # Common structures: images directly in temp_dir, or in a subdirectory like "val" or "ILSVRC2012_img_val"
    val_images_dir = None
    
    # First check if images are directly in temp_dir
    if any(f.suffix.upper() == ".JPEG" for f in temp_val_dir.iterdir() if f.is_file()):
        val_images_dir = temp_val_dir
    else:
        # Look for subdirectories that might contain images
        for item in temp_val_dir.iterdir():
            if item.is_dir():
                # Check if this directory contains JPEG images
                if any(f.suffix.upper() == ".JPEG" for f in item.iterdir() if f.is_file()):
                    val_images_dir = item
                    break
    
    if val_images_dir is None:
        raise ValueError("Could not find validation images in extracted tar. Please check the tar file structure.")
    
    # Load class mapping from devkit if available
    class_to_idx = {}
    if devkit_dir:
        devkit_path = Path(devkit_dir)
        val_groundtruth_file = devkit_path / "data" / "ILSVRC2012_validation_ground_truth.txt"
        synset_file = devkit_path / "data" / "synsets.txt"
        
        if val_groundtruth_file.exists() and synset_file.exists():
            # Load synset names
            with open(synset_file, "r") as f:
                synsets = [line.strip().split()[0] for line in f.readlines()]
            
            # Load ground truth labels
            with open(val_groundtruth_file, "r") as f:
                val_labels = [int(line.strip()) - 1 for line in f.readlines()]  # Convert to 0-indexed
            
            # Create mapping: image_name -> class_name
            # Validation images are named sequentially: ILSVRC2012_val_00000001.JPEG, etc.
            # Sort by filename to match the order in ground truth file
            val_images = sorted(
                [f for f in val_images_dir.iterdir() if f.is_file() and f.suffix.upper() == ".JPEG"],
                key=lambda x: x.name
            )
            
            if len(val_images) != len(val_labels):
                raise ValueError(
                    f"Mismatch: found {len(val_images)} validation images but {len(val_labels)} labels. "
                    "Please check that the devkit matches the validation set."
                )
            
            for img_path, label_idx in zip(val_images, val_labels):
                class_name = synsets[label_idx]
                class_dir = output_path / class_name
                class_dir.mkdir(exist_ok=True)
                shutil.move(str(img_path), str(class_dir / img_path.name))
            
            print(f"Validation images organized into {len(synsets)} class folders")
        else:
            print("Warning: Devkit files not found. Organizing validation images without class structure.")
            print("You may need to manually organize validation images or provide devkit path.")
            # Fallback: just move all images to output (not ideal, but better than nothing)
            for img_path in val_images_dir.glob("*.JPEG"):
                if img_path.is_file():
                    shutil.move(str(img_path), str(output_path / img_path.name))
    else:
        print("Warning: No devkit provided. Validation images will not be organized by class.")
        print("Please provide --devkit-dir or manually organize validation images.")
        # Fallback: move all images to output
        for img_path in val_images_dir.glob("*.JPEG"):
            if img_path.is_file():
                shutil.move(str(img_path), str(output_path / img_path.name))
    
    # Clean up temp directory
    shutil.rmtree(temp_val_dir)
    print(f"Validation images extracted to {output_dir}")


def organize_existing_val_images(val_dir: str, devkit_dir: Optional[str] = None) -> None:
    """
    Organize already-extracted validation images into class folders.
    
    This function works with validation images that are already extracted
    but not yet organized into class-specific folders.
    """
    val_path = Path(val_dir)
    if not val_path.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    
    # Check if images are already organized
    subdirs = [d for d in val_path.iterdir() if d.is_dir()]
    if subdirs and all(d.name.startswith('n') and len(d.name) == 9 for d in subdirs):
        print(f"Validation images appear to already be organized into {len(subdirs)} class folders.")
        return
    
    # Get all validation images (should be in the val_dir itself)
    val_images = sorted(
        [f for f in val_path.iterdir() if f.is_file() and f.suffix.upper() in (".JPEG", ".JPG")],
        key=lambda x: x.name
    )
    
    if not val_images:
        raise ValueError(f"No validation images found in {val_dir}")
    
    print(f"Found {len(val_images)} validation images to organize...")
    
    # Load synsets from devkit
    synsets = None
    if devkit_dir:
        devkit_path = Path(devkit_dir)
        synset_file = devkit_path / "data" / "synsets.txt"
        
        if synset_file.exists():
            with open(synset_file, "r") as f:
                synsets = [line.strip().split()[0] for line in f.readlines()]
        else:
            # Try to extract from meta.mat
            meta_path = devkit_path / "data" / "meta.mat"
            if meta_path.exists():
                try:
                    import scipy.io
                    mat = scipy.io.loadmat(str(meta_path), squeeze_me=True)
                    if 'synsets' in mat:
                        synsets_data = mat['synsets']
                        # Extract synset IDs
                        if hasattr(synsets_data, '__iter__'):
                            synsets = []
                            for item in synsets_data:
                                if isinstance(item, dict):
                                    if 'ILSVRC2012_ID' in item:
                                        synsets.append(item['ILSVRC2012_ID'])
                                    elif 'WNID' in item:
                                        synsets.append(item['WNID'])
                                elif isinstance(item, str):
                                    synsets.append(item)
                            if synsets and len(synsets) >= 1000:
                                synsets = synsets[:1000]
                                # Save to synsets.txt for future use
                                with open(synset_file, "w") as f:
                                    for s in synsets:
                                        f.write(f"{s}\n")
                                print(f"Extracted and saved synsets to {synset_file}")
                except ImportError:
                    print("ERROR: scipy is required to extract synsets from meta.mat.")
                    print("Please install scipy by running: pip install scipy")
                    print("Or add 'scipy' to your requirements.txt and run: pip install -r requirements.txt")
                    raise
                except Exception as e:
                    print(f"Warning: Could not extract synsets from meta.mat: {e}")
    
    if synsets is None or len(synsets) < 1000:
        raise ValueError(
            "Could not determine synset names. Please ensure:\n"
            "  1. devkit_dir contains data/synsets.txt, OR\n"
            "  2. devkit_dir contains data/meta.mat and scipy is installed\n"
            "  3. Or provide --train-dir with organized class folders"
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
    
    if len(val_images) != len(val_labels):
        raise ValueError(
            f"Mismatch: found {len(val_images)} validation images but {len(val_labels)} labels. "
            "Please check that the devkit matches the validation set."
        )
    
    if len(synsets) != 1000:
        raise ValueError(
            f"Expected 1000 synsets, but found {len(synsets)}. "
            "Please check the devkit."
        )
    
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


def extract_devkit(devkit_tar_path: str, output_dir: str) -> str:
    """Extract ImageNet devkit and return path to extracted directory."""
    if not os.path.exists(devkit_tar_path):
        raise FileNotFoundError(
            f"Devkit tar not found: {devkit_tar_path}\n"
            "Please download ILSVRC2012_devkit_t12.tar.gz from ImageNet website "
            "and place it in the ./data/ folder."
        )
    
    print(f"Extracting devkit from {devkit_tar_path}...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(devkit_tar_path, "r:gz") as tar:
        tar.extractall(output_path)
    
    # Find the devkit directory (usually ILSVRC2012_devkit_t12)
    devkit_dir = None
    for item in output_path.iterdir():
        if item.is_dir() and "devkit" in item.name.lower():
            devkit_dir = item
            break
    
    if devkit_dir is None:
        devkit_dir = output_path
    
    print(f"Devkit extracted to {devkit_dir}")
    return str(devkit_dir)


def organize_imagenet(
    train_tar: Optional[str] = None,
    val_tar: Optional[str] = None,
    devkit_tar: Optional[str] = None,
    output_dir: str = "./data/imagenet",
    devkit_dir: Optional[str] = None,
) -> None:
    """
    Main function to organize ImageNet-1k dataset.
    
    Args:
        train_tar: Path to ILSVRC2012_img_train.tar
        val_tar: Path to ILSVRC2012_img_val.tar
        devkit_tar: Path to ILSVRC2012_devkit_t12.tar.gz
        output_dir: Base directory for organized dataset
        devkit_dir: Path to already-extracted devkit (if devkit_tar not provided)
    """
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    
    # Extract devkit first if needed
    if devkit_tar:
        if os.path.exists(devkit_tar):
            devkit_extracted = extract_devkit(devkit_tar, output_path / "devkit")
            devkit_dir = devkit_extracted
        else:
            print(f"Warning: Devkit tar not found: {devkit_tar}")
            print("Skipping devkit extraction. Validation images may not be organized by class.")
            devkit_dir = None
    elif devkit_dir is None:
        # Try to find devkit in output directory
        devkit_candidates = list(output_path.glob("**/ILSVRC2012_devkit*"))
        if devkit_candidates:
            devkit_dir = str(devkit_candidates[0])
            print(f"Found existing devkit at {devkit_dir}")
        else:
            # Check if devkit is already extracted in the devkit subdirectory
            existing_devkit = output_path / "devkit"
            if existing_devkit.exists():
                # Look for the actual devkit folder inside
                for item in existing_devkit.iterdir():
                    if item.is_dir() and "devkit" in item.name.lower():
                        devkit_dir = str(item)
                        print(f"Found existing devkit at {devkit_dir}")
                        break
    
    # Extract training images
    if train_tar:
        if not os.path.exists(train_tar):
            raise FileNotFoundError(
                f"Training tar not found: {train_tar}\n"
                "Please download ILSVRC2012_img_train.tar from ImageNet website "
                "and place it in the ./data/ folder."
            )
        extract_train_tar(train_tar, train_dir)
    else:
        print("No training tar provided. Skipping training set extraction.")
    
    # Extract validation images
    if val_tar:
        if not os.path.exists(val_tar):
            raise FileNotFoundError(
                f"Validation tar not found: {val_tar}\n"
                "Please download ILSVRC2012_img_val.tar from ImageNet website "
                "and place it in the ./data/ folder."
            )
        extract_val_tar(val_tar, val_dir, devkit_dir)
    else:
        print("No validation tar provided. Skipping validation set extraction.")
    
    print("\n" + "="*60)
    print("ImageNet-1k organization complete!")
    print(f"Training images: {train_dir}")
    print(f"Validation images: {val_dir}")
    print("\nYou can now use these paths with the ImageNet dataloader:")
    print(f"  --train-dir {train_dir}")
    print(f"  --val-dir {val_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Download and organize ImageNet-1k dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Extract all components
  python scripts/download_imagenet.py \\
    --train-tar ./data/ILSVRC2012_img_train.tar \\
    --val-tar ./data/ILSVRC2012_img_val.tar \\
    --devkit-tar ./data/ILSVRC2012_devkit_t12.tar.gz \\
    --output-dir ./data/imagenet

  # Extract only validation set (if devkit already extracted)
  python scripts/download_imagenet.py \\
    --val-tar ./data/ILSVRC2012_img_val.tar \\
    --devkit-dir ./data/imagenet/devkit/ILSVRC2012_devkit_t12 \\
    --output-dir ./data/imagenet

Note: ImageNet-1k requires registration at https://www.image-net.org/download.php
Place the downloaded ImageNet tar files in the ./data/ folder before running the script.
        """
    )
    parser.add_argument(
        "--train-tar",
        type=str,
        default=None,
        help="Path to ILSVRC2012_img_train.tar (training images, ~138GB)",
    )
    parser.add_argument(
        "--val-tar",
        type=str,
        default=None,
        help="Path to ILSVRC2012_img_val.tar (validation images, ~6.3GB)",
    )
    parser.add_argument(
        "--devkit-tar",
        type=str,
        default=None,
        help="Path to ILSVRC2012_devkit_t12.tar.gz (devkit with labels)",
    )
    parser.add_argument(
        "--devkit-dir",
        type=str,
        default=None,
        help="Path to already-extracted devkit directory (alternative to --devkit-tar)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/imagenet",
        help="Output directory for organized dataset (default: ./data/imagenet)",
    )
    parser.add_argument(
        "--organize-existing-val",
        action="store_true",
        help="Organize already-extracted validation images (use with --val-dir and --devkit-dir)",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=None,
        help="Path to validation directory (for --organize-existing-val)",
    )
    
    args = parser.parse_args()
    
    # Handle organizing existing validation images
    if args.organize_existing_val:
        if not args.val_dir:
            parser.error("--val-dir is required when using --organize-existing-val")
        if not args.devkit_dir:
            # Try to find devkit in output directory
            output_path = Path(args.output_dir)
            devkit_candidates = list(output_path.glob("**/ILSVRC2012_devkit*"))
            if devkit_candidates:
                args.devkit_dir = str(devkit_candidates[0])
                print(f"Found existing devkit at {args.devkit_dir}")
            else:
                parser.error("--devkit-dir is required when using --organize-existing-val")
        
        organize_existing_val_images(args.val_dir, args.devkit_dir)
        return
    
    if not any([args.train_tar, args.val_tar]):
        parser.error("At least one of --train-tar or --val-tar must be provided")
    
    organize_imagenet(
        train_tar=args.train_tar,
        val_tar=args.val_tar,
        devkit_tar=args.devkit_tar,
        output_dir=args.output_dir,
        devkit_dir=args.devkit_dir,
    )


if __name__ == "__main__":
    main()

