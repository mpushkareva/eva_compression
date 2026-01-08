#!/usr/bin/env python3
"""
Extract ImageNet synset names from meta.mat file without scipy.

This script attempts to extract synset names from the MATLAB .mat file
using a simple binary parser.
"""

import re
from pathlib import Path


def extract_synsets_from_mat(meta_mat_path: Path):
    """Extract synset names from meta.mat using regex pattern matching."""
    if not meta_mat_path.exists():
        return None
    
    print(f"Reading {meta_mat_path}...")
    with open(meta_mat_path, 'rb') as f:
        data = f.read()
    
    # ImageNet synsets are 9-character strings: 'n' followed by 8 digits
    # Pattern: n followed by exactly 8 digits
    synset_pattern = rb'n\d{8}'
    matches = re.findall(synset_pattern, data)
    
    if matches:
        # Convert to strings and get unique values
        synsets = list(set([m.decode('ascii', errors='ignore') for m in matches]))
        # Filter out invalid synsets (should be exactly 9 characters)
        synsets = [s for s in synsets if len(s) == 9 and s.startswith('n')]
        
        if len(synsets) >= 1000:
            # Sort to get consistent order
            synsets = sorted(synsets)
            print(f"Found {len(synsets)} unique synsets")
            return synsets[:1000]  # Take first 1000
    
    return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract synsets from meta.mat")
    parser.add_argument(
        "--meta-mat",
        type=str,
        required=True,
        help="Path to meta.mat file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for synsets.txt",
    )
    
    args = parser.parse_args()
    
    meta_path = Path(args.meta_mat)
    synsets = extract_synsets_from_mat(meta_path)
    
    if synsets is None or len(synsets) < 1000:
        print("Error: Could not extract 1000 synsets from meta.mat")
        print("Please install scipy and use: python -c \"import scipy.io; mat=scipy.io.loadmat('meta.mat'); print(mat['synsets'])\"")
        return 1
    
    # Write synsets to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for synset in synsets:
            f.write(f"{synset}\n")
    
    print(f"Successfully extracted {len(synsets)} synsets to {output_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

