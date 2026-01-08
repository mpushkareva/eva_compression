#!/usr/bin/env python3
"""
Script to parse evaluation outputs and build a comparison table
for different quantization methods and the initial model.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


def parse_eval_line(line: str) -> Optional[Tuple[float, float, int]]:
    """
    Parse a line like: "Eval: Top-1 = 85.23%, Top-5 = 92.45% (N=640)"
    Returns: (top1, top5, n_samples) or None if not matched
    """
    pattern = r'Eval:\s*Top-1\s*=\s*([\d.]+)%\s*,\s*Top-5\s*=\s*([\d.]+)%\s*\(N=(\d+)\)'
    match = re.search(pattern, line)
    if match:
        top1 = float(match.group(1))
        top5 = float(match.group(2))
        n_samples = int(match.group(3))
        return (top1, top5, n_samples)
    return None


def extract_model_name_from_output(lines: List[str], start_idx: int) -> Optional[str]:
    """
    Try to extract model name from context around the evaluation result.
    Looks for patterns like:
    - "Loading model from: ..."
    - "--model ..."
    - Model path in command line
    """
    # Look backwards for model information
    for i in range(max(0, start_idx - 20), start_idx):
        line = lines[i]
        # Check for loading messages
        if "Loading model from:" in line or "Loading quantized model from:" in line:
            match = re.search(r'(?:Loading (?:quantized )?model from:)\s*(.+)', line)
            if match:
                model_path = match.group(1).strip()
                # Extract just the model name or last part of path
                if '/' in model_path:
                    return model_path.split('/')[-1]
                return model_path
        # Check for quantized model path
        if "quantized-model" in line or "--quantized-model" in line:
            match = re.search(r'--quantized-model\s+([^\s]+)', line)
            if match:
                quantized_path = match.group(1).strip()
                # Extract quantization type from path
                if 'manual' in quantized_path.lower():
                    return "Manual Quantization"
                elif 'fp' in quantized_path.lower():
                    return "FP Quantization"
                elif 'quantized' in quantized_path.lower():
                    return "Standard Quantization"
        # Check for quantization config
        if "Quantization Configuration" in line or "=== Manual Quantization Configuration ===" in line:
            # Look for quantization type in following lines
            for j in range(i, min(len(lines), i + 10)):
                if "attention:" in lines[j].lower() or "mlp:" in lines[j].lower():
                    # Try to determine quantization type
                    if "Manual" in lines[i] or "manual" in lines[i]:
                        return "Manual Quantization"
    return None


def parse_output_file(file_path: Path) -> List[Dict]:
    """
    Parse evaluation output file and extract results.
    Returns list of dictionaries with model info and metrics.
    """
    results = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        parsed = parse_eval_line(line)
        if parsed:
            top1, top5, n_samples = parsed
            model_name = extract_model_name_from_output(lines, i)
            
            # Try to determine quantization type from context
            quantization_type = "Initial Model"
            if model_name:
                if "manual" in model_name.lower() or "Manual" in model_name:
                    quantization_type = "Manual Quantization"
                elif "fp" in model_name.lower() or "FP" in model_name:
                    quantization_type = "FP Quantization"
                elif "quantized" in model_name.lower() and "initial" not in model_name.lower():
                    quantization_type = "Standard Quantization"
            else:
                # Look for quantization indicators in nearby lines
                context = ' '.join(lines[max(0, i-10):i+1]).lower()
                if "quantized" in context and "manual" in context:
                    quantization_type = "Manual Quantization"
                elif "quantized" in context and "fp" in context:
                    quantization_type = "FP Quantization"
                elif "quantized" in context:
                    quantization_type = "Standard Quantization"
            
            results.append({
                'model_name': model_name or f"Model {len(results)+1}",
                'quantization_type': quantization_type,
                'top1': top1,
                'top5': top5,
                'n_samples': n_samples,
                'line_number': i + 1
            })
    
    return results


def parse_stdin() -> List[Dict]:
    """Parse evaluation results from stdin."""
    lines = sys.stdin.readlines()
    results = []
    
    for i, line in enumerate(lines):
        parsed = parse_eval_line(line)
        if parsed:
            top1, top5, n_samples = parsed
            model_name = extract_model_name_from_output(lines, i)
            
            quantization_type = "Initial Model"
            if model_name:
                if "manual" in model_name.lower():
                    quantization_type = "Manual Quantization"
                elif "fp" in model_name.lower():
                    quantization_type = "FP Quantization"
                elif "quantized" in model_name.lower():
                    quantization_type = "Standard Quantization"
            
            results.append({
                'model_name': model_name or f"Model {len(results)+1}",
                'quantization_type': quantization_type,
                'top1': top1,
                'top5': top5,
                'n_samples': n_samples,
                'line_number': i + 1
            })
    
    return results


def build_table(results: List[Dict], format: str = "markdown") -> str:
    """
    Build a formatted table from results.
    
    Args:
        results: List of result dictionaries
        format: Output format ("markdown", "latex", "csv", "plain")
    """
    if not results:
        return "No results found!"
    
    # Group by quantization type
    grouped = {}
    for result in results:
        qtype = result['quantization_type']
        if qtype not in grouped:
            grouped[qtype] = []
        grouped[qtype].append(result)
    
    # Sort quantization types: Initial Model first, then others
    qtype_order = ["Initial Model"]
    for qtype in sorted(grouped.keys()):
        if qtype not in qtype_order:
            qtype_order.append(qtype)
    
    if format == "markdown":
        return build_markdown_table(grouped, qtype_order)
    elif format == "latex":
        return build_latex_table(grouped, qtype_order)
    elif format == "csv":
        return build_csv_table(results)
    else:  # plain
        return build_plain_table(grouped, qtype_order)


def build_markdown_table(grouped: Dict, qtype_order: List[str]) -> str:
    """Build a markdown table."""
    lines = []
    lines.append("## Model Quality Comparison")
    lines.append("")
    lines.append("| Quantization Type | Model Name | Top-1 Accuracy (%) | Top-5 Accuracy (%) | N Samples |")
    lines.append("|-------------------|------------|---------------------|---------------------|-----------|")
    
    for qtype in qtype_order:
        if qtype in grouped:
            for result in grouped[qtype]:
                lines.append(
                    f"| {qtype} | {result['model_name']} | "
                    f"{result['top1']:.2f} | {result['top5']:.2f} | {result['n_samples']} |"
                )
    
    return "\n".join(lines)


def build_latex_table(grouped: Dict, qtype_order: List[str]) -> str:
    """Build a LaTeX table."""
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{|l|l|r|r|r|}")
    lines.append("\\hline")
    lines.append("Quantization Type & Model Name & Top-1 (\\%) & Top-5 (\\%) & N Samples \\\\")
    lines.append("\\hline")
    
    for qtype in qtype_order:
        if qtype in grouped:
            for result in grouped[qtype]:
                lines.append(
                    f"{qtype} & {result['model_name']} & "
                    f"{result['top1']:.2f} & {result['top5']:.2f} & {result['n_samples']} \\\\"
                )
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Model Quality Comparison}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def build_csv_table(results: List[Dict]) -> str:
    """Build a CSV table."""
    lines = []
    lines.append("Quantization Type,Model Name,Top-1 Accuracy (%),Top-5 Accuracy (%),N Samples")
    
    for result in results:
        lines.append(
            f"{result['quantization_type']},{result['model_name']},"
            f"{result['top1']:.2f},{result['top5']:.2f},{result['n_samples']}"
        )
    
    return "\n".join(lines)


def build_plain_table(grouped: Dict, qtype_order: List[str]) -> str:
    """Build a plain text table."""
    lines = []
    lines.append("=" * 80)
    lines.append("Model Quality Comparison")
    lines.append("=" * 80)
    lines.append("")
    
    # Calculate column widths
    max_qtype_len = max(len(qtype) for qtype in qtype_order) if qtype_order else 20
    max_model_len = max(len(r['model_name']) for results in grouped.values() for r in results) if grouped else 20
    max_qtype_len = max(max_qtype_len, len("Quantization Type"))
    max_model_len = max(max_model_len, len("Model Name"))
    
    # Header
    header = f"{'Quantization Type':<{max_qtype_len}} | {'Model Name':<{max_model_len}} | "
    header += f"{'Top-1 (%)':>10} | {'Top-5 (%)':>10} | {'N':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    
    # Rows
    for qtype in qtype_order:
        if qtype in grouped:
            for result in grouped[qtype]:
                row = f"{qtype:<{max_qtype_len}} | {result['model_name']:<{max_model_len}} | "
                row += f"{result['top1']:>10.2f} | {result['top5']:>10.2f} | {result['n_samples']:>6}"
                lines.append(row)
    
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Parse evaluation outputs and build comparison table"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input file with evaluation output (if not provided, reads from stdin)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for table (if not provided, prints to stdout)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="markdown",
        choices=["markdown", "latex", "csv", "plain"],
        help="Output format (default: markdown)"
    )
    
    args = parser.parse_args()
    
    # Parse input
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        results = parse_output_file(input_path)
    else:
        results = parse_stdin()
    
    if not results:
        print("No evaluation results found in input!", file=sys.stderr)
        print("Expected format: 'Eval: Top-1 = XX.XX%, Top-5 = XX.XX% (N=XXX)'", file=sys.stderr)
        sys.exit(1)
    
    # Build table
    table = build_table(results, format=args.format)
    
    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(table)
        print(f"Table saved to: {output_path}")
    else:
        print(table)


if __name__ == "__main__":
    main()

