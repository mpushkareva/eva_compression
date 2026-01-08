#!/usr/bin/env python3
"""
Create synsets.txt from meta.mat file.

This script requires scipy to be installed. Run it in an environment where scipy is available.
"""

import sys
from pathlib import Path


def create_synsets_txt(meta_mat_path: str, output_path: str):
    """Extract synsets from meta.mat and save to synsets.txt"""
    try:
        import scipy.io
    except ImportError:
        print("ERROR: scipy is required. Please install it with: pip install scipy")
        sys.exit(1)
    
    meta_path = Path(meta_mat_path)
    if not meta_path.exists():
        print(f"ERROR: meta.mat not found at {meta_mat_path}")
        sys.exit(1)
    
    print(f"Reading {meta_mat_path}...")
    mat = scipy.io.loadmat(str(meta_path), squeeze_me=True)
    
    synsets = None
    if 'synsets' in mat:
        synsets_data = mat['synsets']
        synsets = []
        
        # Extract synset IDs from the structure
        if hasattr(synsets_data, '__iter__'):
            for item in synsets_data:
                if isinstance(item, dict):
                    if 'ILSVRC2012_ID' in item:
                        synsets.append(item['ILSVRC2012_ID'])
                    elif 'WNID' in item:
                        synsets.append(item['WNID'])
                elif isinstance(item, str):
                    synsets.append(item)
                elif hasattr(item, '__dict__'):
                    # Try to get ID from object
                    if hasattr(item, 'ILSVRC2012_ID'):
                        synsets.append(item.ILSVRC2012_ID)
                    elif hasattr(item, 'WNID'):
                        synsets.append(item.WNID)
        
        if not synsets:
            # Try alternative structure
            if isinstance(synsets_data, dict):
                if 'ILSVRC2012_ID' in synsets_data:
                    synsets = list(synsets_data['ILSVRC2012_ID'])
                elif 'WNID' in synsets_data:
                    synsets = list(synsets_data['WNID'])
    
    if synsets is None or len(synsets) < 1000:
        print(f"ERROR: Could not extract synsets. Found {len(synsets) if synsets else 0} synsets.")
        print("Available keys in mat file:", list(mat.keys()))
        sys.exit(1)
    
    # Ensure we have exactly 1000
    synsets = synsets[:1000]
    
    # Write to file
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, 'w') as f:
        for synset in synsets:
            f.write(f"{synset}\n")
    
    print(f"Successfully created {output_path} with {len(synsets)} synsets")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_synsets_txt.py <meta.mat path> <output synsets.txt path>")
        print("\nExample:")
        print("  python create_synsets_txt.py \\")
        print("    ./data/imagenet/devkit/ILSVRC2012_devkit_t12/data/meta.mat \\")
        print("    ./data/imagenet/devkit/ILSVRC2012_devkit_t12/data/synsets.txt")
        sys.exit(1)
    
    sys.exit(create_synsets_txt(sys.argv[1], sys.argv[2]))

