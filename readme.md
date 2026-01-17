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

### Quantize EVA Models

Quantize an EVA model with selective layer quantization:

```bash
python scripts/quantize_eva.py \
  --model timm/eva02_base_patch14_224.mim_in22k \
  --output checkpoints/quantized_eva.pth \
  --mode dynamic \
  --num-labels 1000
```

**Options:**
- `--mode`: Quantization mode - `dynamic` (weights only), `static` (weights + activations), or `qat` (quantization-aware training)
- `--quantize-attention`: Enable quantization of attention layers (default: enabled)
- `--quantize-mlp`: Enable quantization of MLP layers (default: enabled)
- `--quantize-embedding`: Enable quantization of embedding layers (default: enabled)
- `--quantize-head`: Enable quantization of classification head (default: enabled)
- `--bits`: Quantization bit-width (4 or 8, default: 8)
- `--dtype`: Quantization dtype (`qint8` or `quint8`, default: `qint8`)

**Example: Quantize only attention and MLP layers:**
```bash
python scripts/quantize_eva.py \
  --model timm/eva02_base_patch14_224.mim_in22k \
  --output checkpoints/quantized_eva.pth \
  --mode dynamic \
  --no-quantize-embedding \
  --no-quantize-head
```

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

**Evaluate quantized model:**
```bash
python scripts/eval_classification.py \
  --model timm/eva02_small_patch14_336.mim_in22k_ft_in1k \
  --val-dir ./data/imagenet/val \
  --dataset imagenet \
  --quantized-model checkpoints/quantized_model.pth \
  --device cuda
```

Note: The `--train-dir` parameter is optional. For evaluation purposes, you only need the validation set. The dataloader will work with just `--val-dir` specified.

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

# TODO: check class labels after changes

**CIFAR-10 to ImageNet mapping:**
- airplane → airliner (class 404)
- automobile → convertible (class 511)
- bird → bird (class 80)
- cat → tabby cat (class 281)
- deer → elk (class 345)
- dog → dog (class 151)
- frog → tree frog (class 30)
- horse → horse (class 340)
- ship → container ship (class 510)
- truck → pickup truck (class 609)

### Batch Evaluation

Evaluate all quantized models in the checkpoints directory:

```bash
python scripts/evaluate_all_models.py \
  --base-model timm/eva02_small_patch14_336.mim_in22k_ft_in1k \
  --val-dir ./data/imagenet/val \
  --dataset imagenet \
  --work-dir checkpoints \
  --output-json checkpoints/evaluation_results.json \
  --device cuda
```

**Options:**
- `--work-dir`: Directory containing quantized model files (default: `checkpoints`)
- `--output-json`: Path to save evaluation results (default: `checkpoints/evaluation_results.json`)
- `--skip-base`: Skip evaluation of the base (non-quantized) model

### Build Results Table

Generate a comparison table from evaluation results:

```bash
python scripts/build_results_table.py \
  --input checkpoints/evaluation_results.json \
  --output results_table.txt
```

## Data Paths

All scripts use the following default data paths:
- **ImageNet validation**: `./data/imagenet/val`
- **ImageNet training**: `./data/imagenet/train`
- **CIFAR-10**: `./data/cifar-10-batches-py`

You can override these paths using the `--val-dir` and `--train-dir` arguments.

## Model Checkpoints

Quantized models are saved to the `checkpoints/` directory by default. The quantization scripts create:
- `*.pth`: Quantized model state dict
- `*.wrapper.pth`: Full wrapper model (recommended for loading)
- `*.metadata.pth`: Quantization metadata (for some quantization methods)
- `*.scales.pth`: Quantization scales (for manual quantization)

When evaluating quantized models, use the `*.wrapper.pth` file with the `--quantized-model` argument.

## Notes

- All scripts should be run from the project root directory
- Make sure to set the `PYTHONPATH` if running scripts from other directories:
  ```bash
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  ```
- For HuggingFace models, you may need to set a token via `--token` or environment variables (`HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`)
