# EVA Compression Project

This project provides tools for quantizing and evaluating EVA-transformer models for image classification tasks.

## Project Structure

```
eva_compression/
├── scripts/              # Main executable scripts
│   ├── quantize_eva.py          # Quantize EVA models
│   ├── quantize_fp.py           # Fixed-point quantization
│   ├── quantize_manual.py       # Manual quantization
│   ├── eval_classification.py   # Evaluate models on datasets
│   ├── evaluate_all_models.py   # Batch evaluation of multiple models
│   ├── build_results_table.py   # Build comparison tables from results
│   ├── download_imagenet.py     # Download and organize ImageNet dataset
│   └── compare_qtorch_fixedpoint.py  # Compare quantization methods
├── utils/                # Utility modules
│   ├── dataloaders/            # Data loading utilities
│   │   ├── image_dataloaders.py
│   │   └── video_dataloaders.py
│   ├── config.py               # Configuration settings
│   ├── create_synsets_from_standard.py
│   ├── create_synsets_txt.py
│   ├── extract_synsets.py
│   ├── organize_val_images.py
│   └── fix_val_organization*.py
├── data/                 # Dataset storage
│   ├── imagenet/         # ImageNet dataset
│   └── cifar-10-batches-py/  # CIFAR-10 dataset
├── checkpoints/          # Saved model checkpoints
│   ├── quantized_*.pth   # Quantized model files
│   └── evaluation_results.json  # Evaluation results
└── requirements.txt     # Python dependencies
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ImageNet-1k Dataset Setup

ImageNet-1k requires registration at [ImageNet website](https://www.image-net.org/download.php). After registration, download:
- `ILSVRC2012_img_train.tar` (training images, ~138GB) - **Required for training**
- `ILSVRC2012_img_val.tar` (validation images, ~6.3GB) - **Required for validation**
- `ILSVRC2012_devkit_t12.tar.gz` (development kit with labels) - **Optional but recommended** (needed to organize validation images by class)

Place the downloaded files in the `./data/` folder, then organize the dataset:

### Extract all components (recommended):
```bash
python scripts/download_imagenet.py \
  --train-tar ./data/ILSVRC2012_img_train.tar \
  --val-tar ./data/imagenet/ILSVRC2012_img_val.tar \
  --devkit-tar ./data/imagenet/ILSVRC2012_devkit_t12.tar.gz \
  --output-dir ./data/imagenet
```

### Extract only validation set (devkit required for class organization):
```bash
python scripts/download_imagenet.py \
  --val-tar ./data/imagenet/ILSVRC2012_img_val.tar \
  --devkit-tar ./data/imagenet/ILSVRC2012_devkit_t12.tar.gz \
  --output-dir ./data/imagenet
```

### Organize already-extracted validation images:
If validation images are already extracted but not organized into class folders:
```bash
# First, install scipy (required to read meta.mat):
pip install scipy

# Then organize the images:
python scripts/download_imagenet.py \
  --organize-existing-val \
  --val-dir ./data/imagenet/val \
  --devkit-dir ./data/imagenet/ILSVRC2012_devkit_t12
```

This will create the following structure:
```
data/imagenet/
  train/
    n01440764/  (class folders with training images)
    n01443537/
    ...
  val/
    n01440764/  (class folders with validation images)
    n01443537/
    ...
```

## Usage


### Evaluate Models

#### Evaluate on ImageNet-1k:

**With validation data only (recommended for evaluation):**
```bash
python scripts/eval_classification.py \
  --model timm/eva02_small_patch14_336.mim_in22k_ft_in1k \
  --val-dir ./data/imagenet/val \
  --dataset imagenet \
  --device cuda
```

**With both training and validation data:**
```bash
python scripts/eval_classification.py \
  --model timm/eva02_small_patch14_336.mim_in22k_ft_in1k \
  --train-dir ./data/imagenet/train \
  --val-dir ./data/imagenet/val \
  --dataset imagenet \
  --device cuda
```


#### Evaluate on CIFAR-10:
```bash
python scripts/eval_classification.py \
  --model timm/eva02_small_patch14_336.mim_in22k_ft_in1k \
  --val-dir ./data/cifar-10-batches-py \
  --dataset cifar10 \
  --device cuda
```

#### Evaluate on CIFAR-10 with ImageNet labels (for ImageNet-trained models):
```bash
python scripts/eval_classification.py \
  --model timm/eva02_small_patch14_336.mim_in22k_ft_in1k \
  --val-dir ./data/cifar-10-batches-py \
  --dataset cifar10 \
  --use-imagenet-labels \
  --device cuda
```

When using `--use-imagenet-labels`, CIFAR-10 labels are mapped to corresponding ImageNet class indices. This allows you to use ImageNet-trained models directly on CIFAR-10 data without fine-tuning. The model will output predictions for all 1000 ImageNet classes, and accuracy is computed based on the mapped ImageNet class indices.


Reading `eval_quant.py` to extract arguments, usage, and purpose for the README.


[2 tools called]


Here’s a README you can use for `eval_quant.py`. In **Ask mode** I can’t create or edit files; you can paste this into `scripts/EVAL_QUANT_README.md` (or add it to the main `readme.md`), or switch to **Agent mode** if you want it written to a file automatically.

---

## `eval_quant.py` — Evaluate quantized EVA on ImageNet

Evaluates quantized EVA (PyTorch) models on the ImageNet validation set. It loads a model by HuggingFace/timm id (or path), applies the chosen quantization backend in memory, runs validation with the model’s image processor, and reports Top-1 and Top-5 accuracy.

### Requirements

- ImageNet validation data organized by class (e.g. `./data/imagenet/val/n01440764/`, …).
- `PYTHONPATH` set to the repo root when running the script (or run from repo root and set `PYTHONPATH=.`).

### Basic usage

```bash
# From repo root, with PYTHONPATH so scripts and utils resolve
PYTHONPATH=. python scripts/eval_quant.py \
  --model timm/eva02_tiny_patch14_336.mim_in22k_ft_in1k \
  --val-dir ./data/imagenet/val
```

### Required arguments

| Argument     | Description |
|-------------|-------------|
| `--model`   | HuggingFace/timm model id (e.g. `timm/eva02_tiny_patch14_336.mim_in22k_ft_in1k`) or path to a local model. Used for both loading weights and the image processor. |
| `--val-dir` | Path to ImageNet validation directory (class subfolders, e.g. `./data/imagenet/val`). |

### Quantization type: `--quant_type`

Controls which quantization path is used (all run in memory; no separate “saved quantized file” is required for evaluation):

| Value       | Description |
|------------|-------------|
| `torch`    | PyTorch dynamic/static quantization (`quantize_torch`). |
| `fixed`    | Fixed-point quantization via `quantize_fp`. |
| `fixed_op` | Fixed-point op-level quantization via `quantize_fp_op`. |
| `manual`   | Manual quantization (`quantize_manual`). |
| `origin`   | No quantization; load and evaluate the original model. |

### Example: fixed-op quantization (matches your launch config)

```bash
PYTHONPATH=. python scripts/eval_quant.py \
  --model timm/eva02_tiny_patch14_336.mim_in22k_ft_in1k \
  --val-dir ./data/imagenet/val \
  --device cpu \
  --quant_type fixed_op \
  --quantize-all \
  --forward-format fixed \
  --forward-wl 16 \
  --forward-fl 8
```

### Data and device

| Argument        | Default | Description |
|----------------|--------|-------------|
| `--val-dir`    | *(required)* | ImageNet validation root (class subfolders). |
| `--batch-size` | `1`    | Batch size for evaluation. |
| `--num-workers`| `4`    | DataLoader workers. |
| `--device`     | `cuda` if available, else `cpu` | Device to run the model. |
| `--num-classes`| `1000` | Number of classes (ImageNet-1k). |

### Quantization scope (for `fixed` / `fixed_op` / `manual`)

- `--quantize-all` — quantize all supported layers (simplest).
- Or choose modules: `--attention`, `--mlp`, `--embedding`, `--norm`, `--head`, `--other`.

### Fixed-point format (for `fixed` and `fixed_op`)

- **Forward:** `--forward-format` (e.g. `fixed`), `--forward-wl`, `--forward-fl`, `--forward-exp`, `--forward-man`, `--forward-rounding`.
- **Backward:** `--backward-exp`, `--backward-man`, `--backward-rounding`.

Typical example: `--forward-format fixed --forward-wl 16 --forward-fl 8`.

### PyTorch quantization (for `--quant_type torch`)

- `--mode` — e.g. `dynamic`.
- `--dtype` — e.g. `qint8`.
- `--weight-bits`, `--activation-bits` — bit width.

### HuggingFace

- `--base-model` — override model id/path used for the **image processor** (if different from `--model`).
- `--token` — HuggingFace token; else uses `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or HF cache.
