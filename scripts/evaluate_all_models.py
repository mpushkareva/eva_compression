#!/usr/bin/env python3
"""
Evaluate all quantized models and the base model, saving results incrementally to JSON.

This script:
1. Finds all quantized model files (*.pth, excluding metadata/scales files)
2. Evaluates each quantized model
3. Evaluates the base (non-quantized) model
4. Saves results to JSON after each evaluation
5. Can resume from where it stopped if interrupted
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import traceback


def find_quantized_models(work_dir: Path) -> List[Tuple[str, Path]]:
    """
    Find all quantized model files.
    
    Returns:
        List of tuples: (model_name, model_path)
        Excludes metadata, scales, and wrapper files (we'll handle wrappers separately)
    """
    quantized_models = []
    
    # Patterns to exclude (metadata, scales, etc.)
    exclude_patterns = ['.metadata.pth', '.scales.pth']
    
    # Find all .pth files
    for pth_file in work_dir.glob('*.pth'):
        # Skip excluded patterns
        if any(pattern in pth_file.name for pattern in exclude_patterns):
            continue
        
        # Skip wrapper files (we'll check for them separately when needed)
        if pth_file.name.endswith('.wrapper.pth'):
            continue
        
        # Check if this is a quantized model (not just any .pth file)
        # We'll consider files with "quantized" in the name, or check if corresponding wrapper exists
        if 'quantized' in pth_file.name.lower():
            model_name = pth_file.stem
            quantized_models.append((model_name, pth_file))
        elif pth_file.name.endswith('.weights.pth'):
            # Manual quantization uses .weights.pth
            model_name = pth_file.stem.replace('.weights', '')
            quantized_models.append((model_name, pth_file))
    
    return quantized_models


def evaluate_model(
    model_path: str,
    quantized_model_path: Optional[str],
    args: argparse.Namespace,
    work_dir: Path
) -> Dict:
    """
    Evaluate a single model using eval_classification.py.
    
    Returns:
        Dictionary with evaluation results
    """
    # Get the script directory (where eval_classification.py is located)
    script_dir = Path(__file__).parent
    cmd = [
        sys.executable,
        str(script_dir / "eval_classification.py"),
        "--model", model_path,
        "--val-dir", args.val_dir,
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--img-size", str(args.img_size),
        "--eval-resize", str(args.eval_resize),
        "--dataset", args.dataset,
    ]
    
    if args.train_dir:
        cmd.extend(["--train-dir", args.train_dir])
    
    if args.token:
        cmd.extend(["--token", args.token])
    
    if args.use_imagenet_labels:
        cmd.append("--use-imagenet-labels")
    
    if quantized_model_path:
        cmd.extend(["--quantized-model", str(quantized_model_path)])
    
    print(f"\n{'='*80}")
    print(f"Evaluating model: {model_path}")
    if quantized_model_path:
        print(f"  Quantized model: {quantized_model_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Run evaluation and capture output
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # Don't raise on non-zero exit
        )
        
        # Parse output to extract Top-1 and Top-5 accuracy
        top1 = None
        top5 = None
        n_samples = None
        
        # Look for the final evaluation line: "Eval: Top-1 = X.XX%, Top-5 = Y.YY% (N=ZZZ)"
        for line in result.stdout.split('\n'):
            if 'Eval: Top-1' in line:
                # Parse: "Eval: Top-1 = 85.23%, Top-5 = 95.67% (N=50000)"
                try:
                    parts = line.split('Eval:')[1].strip()
                    top1_part = [p for p in parts.split(',') if 'Top-1' in p][0]
                    top5_part = [p for p in parts.split(',') if 'Top-5' in p][0]
                    n_part = [p for p in parts.split('(') if 'N=' in p][0]
                    
                    top1 = float(top1_part.split('=')[1].strip().replace('%', ''))
                    top5 = float(top5_part.split('=')[1].strip().replace('%', ''))
                    n_samples = int(n_part.split('=')[1].strip().replace(')', ''))
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse evaluation line: {line}")
                    print(f"Error: {e}")
        
        # If parsing failed, try to extract from stderr or look for other patterns
        if top1 is None:
            # Try stderr
            for line in result.stderr.split('\n'):
                if 'Eval: Top-1' in line:
                    try:
                        parts = line.split('Eval:')[1].strip()
                        top1_part = [p for p in parts.split(',') if 'Top-1' in p][0]
                        top5_part = [p for p in parts.split(',') if 'Top-5' in p][0]
                        n_part = [p for p in parts.split('(') if 'N=' in p][0]
                        
                        top1 = float(top1_part.split('=')[1].strip().replace('%', ''))
                        top5 = float(top5_part.split('=')[1].strip().replace('%', ''))
                        n_samples = int(n_part.split('=')[1].strip().replace(')', ''))
                    except (ValueError, IndexError):
                        pass
        
        # Check if evaluation was successful
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Unknown error"
            print(f"ERROR: Evaluation failed with return code {result.returncode}")
            print(f"Error output: {error_msg[:500]}")
            return {
                "model": model_path,
                "quantized_model": quantized_model_path,
                "status": "error",
                "error": error_msg[:1000],  # Truncate long errors
                "return_code": result.returncode,
                "top1": None,
                "top5": None,
                "n_samples": None,
                "timestamp": datetime.now().isoformat()
            }
        
        if top1 is None or top5 is None:
            return {
                "model": model_path,
                "quantized_model": quantized_model_path,
                "status": "error",
                "error": "Could not parse evaluation results from output",
                "stdout": result.stdout[-1000:],  # Last 1000 chars
                "stderr": result.stderr[-1000:] if result.stderr else None,
                "top1": None,
                "top5": None,
                "n_samples": None,
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "model": model_path,
            "quantized_model": quantized_model_path,
            "status": "success",
            "top1": top1,
            "top5": top5,
            "n_samples": n_samples,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR: Exception during evaluation: {error_msg}")
        traceback.print_exc()
        return {
            "model": model_path,
            "quantized_model": quantized_model_path,
            "status": "error",
            "error": error_msg,
            "top1": None,
            "top5": None,
            "n_samples": None,
            "timestamp": datetime.now().isoformat()
        }


def load_results(json_path: Path) -> Dict:
    """Load existing results from JSON file."""
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {json_path}, starting fresh")
            return {"results": [], "evaluated_models": []}
    return {"results": [], "evaluated_models": []}


def save_results(json_path: Path, results: Dict):
    """Save results to JSON file."""
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)


def is_model_evaluated(model_name: str, quantized_path: Optional[Path], results: Dict) -> bool:
    """Check if a model has already been evaluated."""
    evaluated = results.get("evaluated_models", [])
    
    # Create identifier for this model
    if quantized_path:
        identifier = f"{model_name}:{quantized_path.name}"
    else:
        identifier = f"{model_name}:base"
    
    return identifier in evaluated


def mark_model_evaluated(model_name: str, quantized_path: Optional[Path], results: Dict):
    """Mark a model as evaluated."""
    if "evaluated_models" not in results:
        results["evaluated_models"] = []
    
    if quantized_path:
        identifier = f"{model_name}:{quantized_path.name}"
    else:
        identifier = f"{model_name}:base"
    
    if identifier not in results["evaluated_models"]:
        results["evaluated_models"].append(identifier)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all quantized models and base model, saving results incrementally."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="timm/eva02_base_patch14_224.mim_in22k",
        help="Base model path or HuggingFace model ID (default: timm/eva02_base_patch14_224.mim_in22k)",
    )
    parser.add_argument("--train-dir", type=str, default="")
    parser.add_argument("--val-dir", type=str, required=True, help="Validation data directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--eval-resize", type=int, default=256)
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        choices=["imagenet", "cifar10", "cifar100"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional)",
    )
    parser.add_argument(
        "--use-imagenet-labels",
        action="store_true",
        help="For CIFAR: map labels to ImageNet indices",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="checkpoints/evaluation_results.json",
        help="Output JSON file path (default: checkpoints/evaluation_results.json)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="checkpoints",
        help="Working directory where quantized models are stored (default: checkpoints)",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip evaluation of base (non-quantized) model",
    )
    
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir).resolve()
    output_json = Path(args.output_json)
    
    print(f"Working directory: {work_dir}")
    print(f"Output JSON: {output_json}")
    print(f"Base model: {args.base_model}")
    
    # Load existing results
    results = load_results(output_json)
    print(f"Loaded {len(results.get('results', []))} existing results")
    
    # Find all quantized models
    quantized_models = find_quantized_models(work_dir)
    print(f"\nFound {len(quantized_models)} quantized model(s):")
    for name, path in quantized_models:
        print(f"  - {name}: {path.name}")
    
    # Prepare evaluation tasks
    tasks = []
    
    # Add base model evaluation (if not skipped)
    if not args.skip_base:
        base_identifier = f"{args.base_model}:base"
        if not is_model_evaluated(args.base_model, None, results):
            tasks.append(("base", args.base_model, None))
            print(f"\nWill evaluate base model: {args.base_model}")
        else:
            print(f"\nSkipping base model (already evaluated): {args.base_model}")
    
    # Add quantized model evaluations
    for model_name, quantized_path in quantized_models:
        # Check if wrapper exists (preferred)
        wrapper_path = quantized_path.with_suffix('.wrapper.pth')
        if wrapper_path.exists():
            # Use wrapper if available
            if not is_model_evaluated(model_name, wrapper_path, results):
                tasks.append((model_name, args.base_model, wrapper_path))
                print(f"Will evaluate quantized model: {model_name} (using wrapper: {wrapper_path.name})")
            else:
                print(f"Skipping {model_name} (already evaluated)")
        else:
            # Use regular quantized model
            if not is_model_evaluated(model_name, quantized_path, results):
                tasks.append((model_name, args.base_model, quantized_path))
                print(f"Will evaluate quantized model: {model_name} (using: {quantized_path.name})")
            else:
                print(f"Skipping {model_name} (already evaluated)")
    
    if not tasks:
        print("\nAll models have already been evaluated!")
        return
    
    print(f"\n{'='*80}")
    print(f"Starting evaluation of {len(tasks)} model(s)...")
    print(f"{'='*80}\n")
    
    # Evaluate each model
    for task_idx, (model_name, base_model, quantized_path) in enumerate(tasks, 1):
        print(f"\n[{task_idx}/{len(tasks)}] Evaluating: {model_name}")
        if quantized_path:
            print(f"  Quantized model: {quantized_path}")
        
        # Evaluate
        result = evaluate_model(
            base_model,
            str(quantized_path) if quantized_path else None,
            args,
            work_dir
        )
        
        # Add model name to result
        result["model_name"] = model_name
        
        # Save result
        if "results" not in results:
            results["results"] = []
        results["results"].append(result)
        
        # Mark as evaluated
        mark_model_evaluated(model_name, quantized_path, results)
        
        # Save results immediately (incremental save)
        save_results(output_json, results)
        print(f"  Results saved to {output_json}")
        
        # Print summary
        if result["status"] == "success":
            print(f"  ✓ Success: Top-1 = {result['top1']:.2f}%, Top-5 = {result['top5']:.2f}% (N={result['n_samples']})")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete! Results saved to {output_json}")
    print(f"{'='*80}\n")
    
    # Print summary
    successful = [r for r in results["results"] if r.get("status") == "success"]
    failed = [r for r in results["results"] if r.get("status") != "success"]
    
    print(f"Summary:")
    print(f"  Total evaluations: {len(results['results'])}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    if successful:
        print(f"\nSuccessful evaluations:")
        for r in successful:
            model_name = r.get("model_name", r.get("model", "unknown"))
            print(f"  {model_name}: Top-1 = {r['top1']:.2f}%, Top-5 = {r['top5']:.2f}%")


if __name__ == "__main__":
    main()

