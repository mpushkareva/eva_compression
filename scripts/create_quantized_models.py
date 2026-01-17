#!/usr/bin/env python3
"""
Create quantized models using different methods and parameters.

This script generates multiple quantized model variants by running different
quantization methods (PyTorch quantization, QPyTorch fixed-point, manual quantization)
with various parameter combinations.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def run_quantization(
    script_path: Path,
    model: str,
    output_path: Path,
    extra_args: List[str],
    token: Optional[str] = None,
) -> bool:
    """Run a quantization script with given arguments."""
    cmd = [sys.executable, str(script_path), "--model", model, "--output", str(output_path)]
    
    if token:
        cmd.extend(["--token", token])
    
    cmd.extend(extra_args)
    
    print("\n" + "=" * 80)
    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Quantization failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create quantized models using different methods and parameters"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path to EVA model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory for quantized models (default: checkpoints)"
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=1000,
        help="Number of classification labels (default: 1000 for ImageNet)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip quantization if output file already exists"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=["eva", "fp", "manual", "all"],
        default=["all"],
        help="Quantization methods to use: eva (PyTorch), fp (QPyTorch), manual, or all"
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which methods to run
    use_eva = "all" in args.methods or "eva" in args.methods
    use_fp = "all" in args.methods or "fp" in args.methods
    use_manual = "all" in args.methods or "manual" in args.methods
    
    # Define quantization configurations
    quant_configs = []
    
    # ===== PyTorch Quantization (quantize_eva.py) =====
    if use_eva:
        # Dynamic quantization variants
        quant_configs.extend([
            {
                "script": "quantize_eva.py",
                "output": "quantized_eva_dynamic_8bit_qint8.pth",
                "extra_args": [
                    "--mode", "dynamic",
                    "--bits", "8",
                    "--dtype", "qint8",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "PyTorch Dynamic Quantization (8-bit, qint8)"
            },
            {
                "script": "quantize_eva.py",
                "output": "quantized_eva_dynamic_8bit_quint8.pth",
                "extra_args": [
                    "--mode", "dynamic",
                    "--bits", "8",
                    "--dtype", "quint8",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "PyTorch Dynamic Quantization (8-bit, quint8)"
            },
            # Dynamic with selective layer quantization
            {
                "script": "quantize_eva.py",
                "output": "quantized_eva_dynamic_attn_mlp_only.pth",
                "extra_args": [
                    "--mode", "dynamic",
                    "--bits", "8",
                    "--dtype", "qint8",
                    "--no-quantize-embedding",
                    "--no-quantize-head",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "PyTorch Dynamic Quantization (attention + MLP only)"
            },
        ])
        
        # Static quantization variants
        quant_configs.extend([
            {
                "script": "quantize_eva.py",
                "output": "quantized_eva_static_8bit_per_tensor.pth",
                "extra_args": [
                    "--mode", "static",
                    "--bits", "8",
                    "--dtype", "qint8",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "PyTorch Static Quantization (8-bit, per-tensor)"
            },
            {
                "script": "quantize_eva.py",
                "output": "quantized_eva_static_8bit_per_channel.pth",
                "extra_args": [
                    "--mode", "static",
                    "--bits", "8",
                    "--dtype", "qint8",
                    "--per-channel",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "PyTorch Static Quantization (8-bit, per-channel)"
            },
            {
                "script": "quantize_eva.py",
                "output": "quantized_eva_static_4bit_per_tensor.pth",
                "extra_args": [
                    "--mode", "static",
                    "--bits", "4",
                    "--dtype", "qint8",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "PyTorch Static Quantization (4-bit, per-tensor, experimental)"
            },
        ])
        
        # QAT variants
        quant_configs.extend([
            {
                "script": "quantize_eva.py",
                "output": "quantized_eva_qat_8bit_per_tensor.pth",
                "extra_args": [
                    "--mode", "qat",
                    "--bits", "8",
                    "--dtype", "qint8",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "PyTorch QAT Quantization (8-bit, per-tensor)"
            },
            {
                "script": "quantize_eva.py",
                "output": "quantized_eva_qat_8bit_per_channel.pth",
                "extra_args": [
                    "--mode", "qat",
                    "--bits", "8",
                    "--dtype", "qint8",
                    "--per-channel",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "PyTorch QAT Quantization (8-bit, per-channel)"
            },
        ])
    
    # ===== QPyTorch Fixed-Point Quantization (quantize_fp.py) =====
    if use_fp:
        # Fixed-point variants
        quant_configs.extend([
            {
                "script": "quantize_fp.py",
                "output": "quantized_fp_fixed_8bit_2fl.pth",
                "extra_args": [
                    "--forward-format", "fixed",
                    "--forward-wl", "8",
                    "--forward-fl", "2",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "QPyTorch Fixed-Point (8-bit word, 2-bit fractional)"
            },
            {
                "script": "quantize_fp.py",
                "output": "quantized_fp_fixed_8bit_3fl.pth",
                "extra_args": [
                    "--forward-format", "fixed",
                    "--forward-wl", "8",
                    "--forward-fl", "3",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "QPyTorch Fixed-Point (8-bit word, 3-bit fractional)"
            },
            {
                "script": "quantize_fp.py",
                "output": "quantized_fp_fixed_8bit_4fl.pth",
                "extra_args": [
                    "--forward-format", "fixed",
                    "--forward-wl", "8",
                    "--forward-fl", "4",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "QPyTorch Fixed-Point (8-bit word, 4-bit fractional)"
            },
            {
                "script": "quantize_fp.py",
                "output": "quantized_fp_fixed_6bit_2fl.pth",
                "extra_args": [
                    "--forward-format", "fixed",
                    "--forward-wl", "6",
                    "--forward-fl", "2",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "QPyTorch Fixed-Point (6-bit word, 2-bit fractional)"
            },
            {
                "script": "quantize_fp.py",
                "output": "quantized_fp_fixed_4bit_1fl.pth",
                "extra_args": [
                    "--forward-format", "fixed",
                    "--forward-wl", "4",
                    "--forward-fl", "1",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "QPyTorch Fixed-Point (4-bit word, 1-bit fractional)"
            },
        ])
        
        # Floating-point variants
        quant_configs.extend([
            {
                "script": "quantize_fp.py",
                "output": "quantized_fp_float_5exp_2man.pth",
                "extra_args": [
                    "--forward-format", "float",
                    "--forward-exp", "5",
                    "--forward-man", "2",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "QPyTorch Floating-Point (5 exp bits, 2 mantissa bits)"
            },
            {
                "script": "quantize_fp.py",
                "output": "quantized_fp_float_4exp_3man.pth",
                "extra_args": [
                    "--forward-format", "float",
                    "--forward-exp", "4",
                    "--forward-man", "3",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "QPyTorch Floating-Point (4 exp bits, 3 mantissa bits)"
            },
            {
                "script": "quantize_fp.py",
                "output": "quantized_fp_float_6exp_1man.pth",
                "extra_args": [
                    "--forward-format", "float",
                    "--forward-exp", "6",
                    "--forward-man", "1",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "QPyTorch Floating-Point (6 exp bits, 1 mantissa bit)"
            },
        ])
        
        # QPyTorch with different rounding modes
        quant_configs.extend([
            {
                "script": "quantize_fp.py",
                "output": "quantized_fp_fixed_8bit_2fl_stochastic.pth",
                "extra_args": [
                    "--forward-format", "fixed",
                    "--forward-wl", "8",
                    "--forward-fl", "2",
                    "--forward-rounding", "stochastic",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "QPyTorch Fixed-Point (8-bit, 2-fl, stochastic rounding)"
            },
        ])
    
    # ===== Manual Quantization (quantize_manual.py) =====
    if use_manual:
        quant_configs.extend([
            {
                "script": "quantize_manual.py",
                "output": "quantized_manual_8bit_qint8_per_tensor",
                "extra_args": [
                    "--dtype", "qint8",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "Manual Quantization (8-bit qint8, per-tensor)"
            },
            {
                "script": "quantize_manual.py",
                "output": "quantized_manual_8bit_qint8_per_channel",
                "extra_args": [
                    "--dtype", "qint8",
                    "--per-channel",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "Manual Quantization (8-bit qint8, per-channel)"
            },
            {
                "script": "quantize_manual.py",
                "output": "quantized_manual_8bit_quint8_per_tensor",
                "extra_args": [
                    "--dtype", "quint8",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "Manual Quantization (8-bit quint8, per-tensor)"
            },
            {
                "script": "quantize_manual.py",
                "output": "quantized_manual_8bit_quint8_per_channel",
                "extra_args": [
                    "--dtype", "quint8",
                    "--per-channel",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "Manual Quantization (8-bit quint8, per-channel)"
            },
            # Manual with selective layer quantization
            {
                "script": "quantize_manual.py",
                "output": "quantized_manual_attn_mlp_only",
                "extra_args": [
                    "--dtype", "qint8",
                    "--no-quantize-embedding",
                    "--no-quantize-head",
                    "--num-labels", str(args.num_labels),
                ],
                "description": "Manual Quantization (attention + MLP only)"
            },
        ])
    
    # Run all quantization configurations
    print(f"\n{'='*80}")
    print(f"Creating {len(quant_configs)} quantized model variants")
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    successful = 0
    failed = 0
    skipped = 0
    
    for i, config in enumerate(quant_configs, 1):
        script_path = script_dir / config["script"]
        output_path = output_dir / config["output"]
        
        # Check if output already exists
        if args.skip_existing:
            # For manual quantization, check for .weights.pth file
            if config["script"] == "quantize_manual.py":
                weights_path = output_path.with_suffix(".weights.pth")
                if weights_path.exists():
                    print(f"\n[{i}/{len(quant_configs)}] SKIPPING: {config['description']}")
                    print(f"  Output already exists: {weights_path}")
                    skipped += 1
                    continue
            else:
                if output_path.exists():
                    print(f"\n[{i}/{len(quant_configs)}] SKIPPING: {config['description']}")
                    print(f"  Output already exists: {output_path}")
                    skipped += 1
                    continue
        
        print(f"\n[{i}/{len(quant_configs)}] {config['description']}")
        print(f"  Script: {config['script']}")
        print(f"  Output: {output_path}")
        
        success = run_quantization(
            script_path=script_path,
            model=args.model,
            output_path=output_path,
            extra_args=config["extra_args"],
            token=args.token,
        )
        
        if success:
            successful += 1
            print(f"  ✓ SUCCESS")
        else:
            failed += 1
            print(f"  ✗ FAILED")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total configurations: {len(quant_configs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"\nOutput directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

