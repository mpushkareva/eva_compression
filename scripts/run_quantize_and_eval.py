#!/usr/bin/env python3
"""
Small orchestration script to:
1) Create quantized EVA models (using existing quantization scripts).
2) Evaluate them with existing evaluation script.
3) Store evaluation results incrementally in JSON so the run can be resumed.
4) Optionally remove quantized checkpoints after evaluation.
5) Build a simple Markdown table from the JSON results.

This script is intentionally minimal and only glues together existing pieces.
To add more quantized variants, edit the QUANT_JOBS list below.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def load_existing_results(json_path: Path) -> Dict:
    """Load existing evaluation JSON (if present)."""
    if not json_path.exists():
        return {"results": []}
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"results": []}
        if "results" not in data or not isinstance(data["results"], list):
            data["results"] = []
        return data
    except Exception:
        # If file is corrupted, start fresh but keep the file around.
        return {"results": []}


def get_evaluated_model_names(results_json: Dict) -> List[str]:
    """Return list of successfully evaluated model names from JSON."""
    evaluated = []
    for r in results_json.get("results", []):
        if r.get("status") == "success" and "model_name" in r:
            evaluated.append(r["model_name"])
    return evaluated


def run_quantization_job(
    job: Dict,
    base_model: str,
    work_dir: Path,
    token: str = None,
) -> None:
    """Run one quantization job by calling the appropriate script."""
    script_dir = Path(__file__).parent

    output_path = work_dir / job["output_name"]
    output_arg = str(output_path)

    cmd: List[str] = [
        sys.executable,
        str(script_dir / job["script"]),
        "--model",
        base_model,
        "--output",
        output_arg,
    ]

    if token:
        cmd.extend(["--token", token])

    extra_args = job.get("extra_args", [])
    cmd.extend(extra_args)

    print("\n" + "=" * 80)
    print(f"Running quantization job: {job['id']}")
    print("Command:", " ".join(cmd))
    print("=" * 80 + "\n")

    subprocess.run(cmd, check=True)


def run_evaluation(
    base_model: str,
    work_dir: Path,
    results_json_path: Path,
    args: argparse.Namespace,
) -> None:
    """Call existing evaluate_all_models.py to evaluate all unevaluated models."""
    script_dir = Path(__file__).parent
    cmd: List[str] = [
        sys.executable,
        str(script_dir / "evaluate_all_models.py"),
        "--base-model",
        base_model,
        "--val-dir",
        args.val_dir,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--img-size",
        str(args.img_size),
        "--eval-resize",
        str(args.eval_resize),
        "--dataset",
        args.dataset,
        "--work-dir",
        str(work_dir),
        "--output-json",
        str(results_json_path),
    ]

    if args.train_dir:
        cmd.extend(["--train-dir", args.train_dir])
    if args.token:
        cmd.extend(["--token", args.token])
    if args.use_imagenet_labels:
        cmd.append("--use-imagenet-labels")
    if args.skip_base:
        cmd.append("--skip-base")

    print("\n" + "=" * 80)
    print("Running evaluation for all pending models")
    print("Command:", " ".join(cmd))
    print("=" * 80 + "\n")

    subprocess.run(cmd, check=True)


def remove_checkpoints(work_dir: Path, job_id: str) -> None:
    """Remove checkpoint files related to a given job id."""
    # We remove a small set of common patterns; missing files are ignored.
    patterns = [
        f"{job_id}.pth",
        f"{job_id}.wrapper.pth",
        f"{job_id}.metadata.pth",
        f"{job_id}.weights.pth",
        f"{job_id}.scales.pth",
    ]
    for name in patterns:
        p = work_dir / name
        if p.exists():
            print(f"Removing checkpoint file: {p}")
            try:
                p.unlink()
            except Exception as e:
                print(f"Warning: could not remove {p}: {e}")


def build_markdown_table_from_json(results_json: Dict) -> str:
    """Build a minimal Markdown table from evaluation_results.json."""
    rows = [r for r in results_json.get("results", []) if r.get("status") == "success"]
    if not rows:
        return "# Evaluation Results\n\nNo successful evaluations found.\n"

    def infer_quant_type(r: Dict) -> str:
        name = (r.get("model_name") or "").lower()
        qpath = (r.get("quantized_model") or "").lower()
        text = name + " " + qpath
        if "manual" in text:
            return "Manual Quantization"
        if "fp" in text:
            return "FP Quantization"
        if "quantized" in text:
            return "Standard Quantization"
        return "Initial Model"

    lines: List[str] = []
    lines.append("## Evaluation Results")
    lines.append("")
    lines.append("| Quantization Type | Model Name | Top-1 (%) | Top-5 (%) | N Samples |")
    lines.append("|-------------------|------------|-----------|-----------|-----------|")

    for r in rows:
        qtype = infer_quant_type(r)
        model_name = r.get("model_name", "unknown")
        top1 = r.get("top1")
        top5 = r.get("top5")
        n = r.get("n_samples")
        if top1 is None or top5 is None or n is None:
            continue
        lines.append(
            f"| {qtype} | {model_name} | "
            f"{top1:.2f} | {top5:.2f} | {n} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create quantized EVA models, evaluate them, and build a table."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="timm/eva02_base_patch14_224.mim_in22k",
        help="Base model HF id or local path (same as in evaluate_all_models.py).",
    )
    parser.add_argument("--train-dir", type=str, default="")
    parser.add_argument("--val-dir", type=str, required=True)
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
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face token.",
    )
    parser.add_argument(
        "--use-imagenet-labels",
        action="store_true",
        help="For CIFAR-10: map labels to ImageNet indices.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="checkpoints",
        help="Directory to store quantized checkpoints.",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        default="checkpoints/evaluation_results.json",
        help="Path to incremental evaluation JSON.",
    )
    parser.add_argument(
        "--results-table",
        type=str,
        default="checkpoints/evaluation_table.md",
        help="Path to Markdown table with aggregated results.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Do not delete quantized checkpoints after evaluation.",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Pass through to evaluation script: do not evaluate base model.",
    )

    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    results_json_path = Path(args.results_json)

    # Load existing evaluation results to support resume.
    eval_data = load_existing_results(results_json_path)
    evaluated_names = set(get_evaluated_model_names(eval_data))
    print(f"Found {len(evaluated_names)} successfully evaluated model(s) in JSON.")

    # Define quantization jobs.
    # To add more, just append new dicts here.
    quant_jobs = [
        {
            "id": "quantized_eva_dynamic",
            "script": "quantize_eva.py",
            "output_name": "quantized_eva_dynamic.pth",
            "extra_args": ["--mode", "dynamic"],
        },
        {
            "id": "quantized_eva_static",
            "script": "quantize_eva.py",
            "output_name": "quantized_eva_static.pth",
            "extra_args": ["--mode", "static", "--per-channel"],
        },
        {
            "id": "quantized_fp_default",
            "script": "quantize_fp.py",
            "output_name": "quantized_fp_default.pth",
            "extra_args": [],
        },
        {
            "id": "quantized_manual_default",
            "script": "quantize_manual.py",
            # Manual quantization script appends .weights.pth / .scales.pth itself.
            "output_name": "quantized_manual_default",
            "extra_args": [],
        },
    ]

    # Process each quantization job.
    for job in quant_jobs:
        job_id = job["id"]
        if job_id in evaluated_names:
            print(f"Skipping job '{job_id}' (already evaluated according to JSON).")
            continue

        # Check whether corresponding checkpoint already exists; if not, quantize.
        output_path = work_dir / job["output_name"]
        needs_quant = True

        if job_id == "quantized_manual_default":
            # Manual quantization creates .weights.pth
            weights_path = output_path.with_suffix(".weights.pth")
            if weights_path.exists():
                needs_quant = False
        else:
            if output_path.exists():
                needs_quant = False

        if needs_quant:
            run_quantization_job(job, args.base_model, work_dir, token=args.token)
        else:
            print(f"Quantized checkpoint for '{job_id}' already exists, skipping quantization.")

        # Run evaluation for all pending models (including this one).
        run_evaluation(args.base_model, work_dir, results_json_path, args)

        # Reload evaluation JSON to update the set of evaluated models.
        eval_data = load_existing_results(results_json_path)
        evaluated_names = set(get_evaluated_model_names(eval_data))

        # Optionally delete checkpoint files for this job.
        if not args.keep_checkpoints and job_id in evaluated_names:
            remove_checkpoints(work_dir, job_id)

    # Final: build/update Markdown table from JSON.
    eval_data = load_existing_results(results_json_path)
    table_md = build_markdown_table_from_json(eval_data)
    table_path = Path(args.results_table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w") as f:
        f.write(table_md)
    print(f"\nAggregated results table written to: {table_path}")


if __name__ == "__main__":
    main()


