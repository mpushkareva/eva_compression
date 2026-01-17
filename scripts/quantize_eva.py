#!/usr/bin/env python3
"""
Quantization script for EVA-transformer models.

This script quantizes EVA-transformer models with optional quantization
for different layer types (attention, MLP, embedding, etc.).
Supports both weight and activation quantization.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Type
import torch
import torch.nn as nn
from torchao.quantization import (
    quantize_, 
    Int4DynamicActivationInt4WeightConfig,
    Int8DynamicActivationInt4WeightConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8DynamicActivationIntxWeightConfig,
    Int4WeightOnlyConfig,
    Int8WeightOnlyConfig,
    Float8WeightOnlyConfig,
    Float8DynamicActivationFloat8WeightConfig,
    Float8StaticActivationFloat8WeightConfig,
    Float8DynamicActivationFloat8SemiSparseWeightConfig,
)
from transformers import AutoImageProcessor, TimmWrapperForImageClassification

GROUP_SIZE = 32


class SelectiveQuantizer:
    
    def __init__(
        self,
        quantization_mode: str = 'static',
        dtype: torch.dtype = torch.qint8,
        weight_bits: int = 8,
        activation_bits: int = 8,
    ):
        self.quantization_mode = quantization_mode
        self.dtype = dtype
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
    
    def quantize_weights_only(self, model: nn.Module) -> nn.Module:
        if self.weight_bits == 4:
            quantize_(model, Int4WeightOnlyConfig(group_size=GROUP_SIZE))
        else:
            quantize_(model, Int8WeightOnlyConfig(group_size=GROUP_SIZE))
        return model
    
    def quantize_dynamic(
        self,
        model: nn.Module,
    ) -> nn.Module:
        if self.weight_bits == 4:
            if self.activation_bits == 4:
                quantize_(model, Int4DynamicActivationInt4WeightConfig(group_size=GROUP_SIZE))
            elif self.activation_bits == 8:
                quantize_(model, Int8DynamicActivationInt4WeightConfig(group_size=GROUP_SIZE))
            else:
                raise ValueError(f"Unsupported activation bit width: {self.activation_bits}")
        elif self.weight_bits == 8:
            if self.activation_bits == 8:
                quantize_(model, Int8DynamicActivationInt8WeightConfig(group_size=GROUP_SIZE))
            else:
                raise ValueError(f"Unsupported activation bit width: {self.activation_bits}")
        else:
            raise ValueError(f"Unsupported weight bit width: {self.weight_bits}")
        return model
    
    def quantize(
        self,
        model: nn.Module
    ) -> nn.Module:
        if self.quantization_mode == 'weights_only':
            return self.quantize_weights_only(model)
        elif self.quantization_mode == 'dynamic':
            return self.quantize_dynamic(model)
        else:
            raise ValueError(f"Unknown quantization mode: {self.quantization_mode}")


def load_eva_model(
    model_name_or_path: str,
    num_labels: int = 1000,
    token: Optional[str] = None,
    return_wrapper: bool = False
) -> nn.Module:
    """Load EVA transformer model using the same method as eval_classification.py.
    
    Args:
        model_name_or_path: HuggingFace model ID or local path
        num_labels: Number of classification labels
        token: HuggingFace token (optional)
        return_wrapper: If True, return the TimmWrapperForImageClassification wrapper.
                       If False, return the underlying timm model.
    
    Returns:
        The model (wrapper or underlying timm model depending on return_wrapper)
    """
    if token is None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token is None:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            pass
    
    model = TimmWrapperForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        token=token,
    )
    
    if return_wrapper:
        return model
    
    if hasattr(model, 'model'):
        # The underlying timm model is typically at model.model
        return model.model
    elif hasattr(model, 'timm_model'):
        return model.timm_model
    else:
        # Fallback: return the wrapper itself
        print("Warning: Could not find underlying timm model, using wrapper directly")
        return model


def print_model_structure(model: nn.Module, max_depth: int = 3):
    """Print model structure to help identify layer types."""
    print("\n=== Model Structure (first 3 levels) ===")
    for name, module in list(model.named_modules())[:50]:  # Show first 50 modules
        depth = name.count('.')
        if depth < max_depth:
            indent = "  " * depth
            print(f"{indent}{name}: {type(module).__name__}")
    print("...\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path to EVA model"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path to save quantized model"
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=1000,
        help="Number of classification labels (default: 1000 for ImageNet)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dynamic",
        choices=["dynamic", "weights_only"],
        help="Quantization mode: dynamic (quantizes weights+activations, requires calibration), "
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="qint8",
        choices=["qint8", "quint8"],
        help="Quantization dtype (qint8 for signed, quint8 for unsigned)"
    )
    parser.add_argument(
        "--weight-bits",
        type=int,
        default=8,
        choices=[4, 8]
    )
    parser.add_argument(
        "--activation-bits",
        type=int,
        default=8,
        choices=[4, 8]
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional)"
    )
    
    args = parser.parse_args()
            
    print(f"Loading model from: {args.model}")
    model_wrapper = load_eva_model(args.model, args.num_labels, args.token, return_wrapper=True)
    model_wrapper.eval()
    
    # Get the underlying timm model for quantization
    if hasattr(model_wrapper, 'model'):
        model = model_wrapper.model
    elif hasattr(model_wrapper, 'timm_model'):
        model = model_wrapper.timm_model
    else:
        model = model_wrapper
    
    model.eval()
    print(f"Model type: {type(model).__name__}")
    print(f"Wrapper type: {type(model_wrapper).__name__}")
    
    print_model_structure(model)
    
    dtype_map = {
        'qint8': torch.qint8,
        'quint8': torch.quint8
    }
    torch_dtype = dtype_map[args.dtype]
    
    quantizer = SelectiveQuantizer(
        quantization_mode=args.mode,
        dtype=torch_dtype,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits
    )
    
    print(f"\nApplying {args.mode} quantization...")
    quantized_model = quantizer.quantize(model)
    print("Quantization completed successfully!")

    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving quantized model to: {output_path}")
    quantized_model.to('cpu')
    torch.save(quantized_model, output_path)
    print("\nQuantization script completed!")


if __name__ == "__main__":
    main()

