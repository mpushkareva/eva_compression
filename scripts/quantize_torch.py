#!/usr/bin/env python3
"""
Quantization script for EVA-transformer models.

This script quantizes EVA-transformer models with optional quantization
for different layer types (attention, MLP, embedding, etc.).
Supports both weight and activation quantization.
"""

import argparse
from copy import deepcopy
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

#  EvaAttention
#       blocks.0.attn.qkv: Linear
#       blocks.0.attn.q_norm: Identity
#       blocks.0.attn.k_norm: Identity
#       blocks.0.attn.attn_drop: Dropout
#       blocks.0.attn.norm: Identity
#       blocks.0.attn.proj: Linear
#       blocks.0.attn.proj_drop: Dropout

class EvaAttentionQuantized(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.q_linear = nn.Linear(in_features=in_features, out_features=in_features)
        self.k_linear = nn.Linear(in_features=in_features, out_features=in_features)
        self.v_linear = nn.Linear(in_features=in_features, out_features=in_features)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout()
        self.norm = nn.Identity()
        self.proj = nn.Linear(in_features=in_features, out_features=in_features * 3)
        self.proj_drop = nn.Dropout()

    def _copy_weights(self, qkv_module: nn.Module):
        self.q_linear.weight.data = qkv_module.q_linear.weight.data
        self.q_linear.bias.data = qkv_module.q_linear.bias.data
        self.k_linear.weight.data = qkv_module.k_linear.weight.data
        self.k_linear.bias.data = qkv_module.k_linear.bias.data
        self.v_linear.weight.data = qkv_module.v_linear.weight.data
        self.v_linear.bias.data = qkv_module.v_linear.bias.data
        self.proj.weight.data = qkv_module.proj.weight.data
        self.proj.bias.data = qkv_module.proj.bias.data
        self.proj_drop.bias.data = qkv_module.proj_drop.bias.data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_norm(self.q_linear(x))
        k = self.k_norm(self.k_linear(x))
        v = self.v_norm(self.v_linear(x))
        attn = self.attn_drop(q @ k.transpose(-2, -1))
        attn = self.norm(attn)
        return self.proj(attn)

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
        
        quantized_model = deepcopy(model)

        
        # Collect modules to quantize with their parent info
        module_info = {}
        for name, module in quantized_model.named_modules():
            # Only quantize modules that have weights (Linear, Conv2d, etc.)
            if hasattr(module, 'weight') and module.weight is not None:
                # Skip if it's not a quantizable layer type
                if not isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
                    continue
                
                # Split name to get parent and attribute name
                name_parts = name.split('.')
                parent_name = '.'.join(name_parts[:-1]) if len(name_parts) > 1 else ''
                attr_name = name_parts[-1]
                
                module_info[name] = {
                    'module': module,
                    'parent_name': parent_name,
                    'attr_name': attr_name
                }
        
        # Quantize each module
        for name, info in module_info.items():
            # Navigate to parent module in quantized_model
            if info['parent_name']:
                parent = quantized_model
                for part in info['parent_name'].split('.'):
                    parent = getattr(parent, part)
            else:
                parent = quantized_model

            group_size = info['module'].weight.size(1)
            if info['attr_name'] == 'qkv':
                module = EvaAttentionQuantized(in_features=group_size)
                module._copy_weights(info['module'])
                setattr(parent, info['attr_name'], module)
                continue
            
            if self.weight_bits == 4:
                quantize_(info['module'], Int4WeightOnlyConfig(group_size=group_size))
            elif self.weight_bits == 8:
                quantize_(info['module'], Int8WeightOnlyConfig(group_size=group_size))
            else:
                raise ValueError(f"Unsupported weight bit width: {self.weight_bits}")
            
            setattr(parent, info['attr_name'], info['module'])
        
        return quantized_model
    
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


def quantize_torch(
    model_name_or_path: str,
    num_labels: int = 1000,
    mode: str = "dynamic",
    dtype: str = "qint8",
    weight_bits: int = 8,
    activation_bits: int = 8,
    verbose: bool = False,
):
            
    print(f"Loading, model from: {model_name_or_path}")
    model_wrapper = load_eva_model(model_name_or_path, num_labels, return_wrapper=True)
    model_wrapper.eval()

    
    model = model_wrapper.timm_model
    
    model.eval()
    if verbose:
        print(f"Model type: {type(model).__name__}")
        print(f"Wrapper type: {type(model_wrapper).__name__}")
        print_model_structure(model)
    
    dtype_map = {
        'qint8': torch.qint8,
        'quint8': torch.quint8
    }
    torch_dtype = dtype_map[dtype]
    
    quantizer = SelectiveQuantizer(
        quantization_mode=mode,
        dtype=torch_dtype,
        weight_bits=weight_bits,
        activation_bits=activation_bits
    )
    
    print(f"\nApplying {mode} quantization...")
    quantized_model = quantizer.quantize(model)
    print("Quantization completed successfully!")

    return quantized_model
