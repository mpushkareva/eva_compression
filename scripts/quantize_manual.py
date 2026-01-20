#!/usr/bin/env python3
"""
Manual quantization script for EVA-transformer models.

This script manually quantizes EVA-transformer models by converting weights
to int8 format and saving float scales for each weight tensor.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, TimmWrapperForImageClassification


class LayerTypeIdentifier:
    ATTENTION_PATTERNS = [
        'attn', 
    ]
    
    MLP_PATTERNS = [
        'mlp',
    ]
    
    EMBEDDING_PATTERNS = [
        'patch_embed',
        'rope'
    ]
    
    NORM_PATTERNS = [
        'norm1', 'norm2', 
    ]
    
    HEAD_PATTERTERNS = [
        'head'
    ]
    
    
    @classmethod
    def identify_layer_type(cls, module_name: str, module: nn.Module) -> str:
        """Identify the type of a layer based on its name and type."""
        name_lower = module_name.lower()
        
        # Check for attention layers
        if any(pattern in name_lower for pattern in cls.ATTENTION_PATTERNS):
            return 'attention'
        
        # Check for MLP/feed-forward layers
        if any(pattern in name_lower for pattern in cls.MLP_PATTERNS):
            return 'mlp'
        
        # Check for embedding layers
        if any(pattern in name_lower for pattern in cls.EMBEDDING_PATTERNS):
            return 'embedding'
        
        # Check for normalization layers
        if any(pattern in name_lower for pattern in cls.NORM_PATTERNS):
            return 'norm'
        
        # Check for classification head
        if any(pattern in name_lower for pattern in cls.HEAD_PATTERTERNS):
            return 'head'
        
        # Check module type
        if isinstance(module, (nn.MultiheadAttention,)):
            return 'attention'
        elif isinstance(module, (nn.Linear,)) and 'attn' not in name_lower:
            # Linear layers that are not attention are likely MLP
            return 'mlp'
        elif isinstance(module, (nn.Embedding, nn.Conv2d)):
            return 'embedding'
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            return 'norm'
        
        return 'other'


class ManualQuantizer:
    def __init__(
        self,
        quantize_config: Dict[str, bool],
        dtype: str = 'qint8',
        per_channel: bool = False,
    ):
        self.quantize_config = quantize_config
        self.dtype = dtype
        self.per_channel = per_channel
        self.layer_identifier = LayerTypeIdentifier()
        self.is_signed = (dtype == 'qint8')
        
    def should_quantize(self, module_name: str, module: nn.Module) -> bool:
        layer_type = self.layer_identifier.identify_layer_type(module_name, module)
        return self.quantize_config.get(layer_type, False)
    
    def quantize_tensor(
        self,
        weight: torch.Tensor,
        per_channel: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if per_channel is None:
            per_channel = self.per_channel
        
        if per_channel and weight.dim() > 1:
            if weight.dim() == 2:
                abs_max = torch.abs(weight).max(dim=1, keepdim=True)[0]
            elif weight.dim() == 4:
                abs_max = torch.abs(weight).view(weight.size(0), -1).max(dim=1, keepdim=True)[0]
            else:
                abs_max = torch.abs(weight).max()
                scale = abs_max / (127.0 if self.is_signed else 255.0)
                scale = scale.clamp(min=1e-8)
                
                quantized = weight / scale
                quantized = quantized.round().clamp(
                    -128 if self.is_signed else 0,
                    127 if self.is_signed else 255
                ).to(torch.int8)
                
                return quantized, scale
            
            scale = abs_max / (127.0 if self.is_signed else 255.0)
            scale = scale.clamp(min=1e-8)
            
            if weight.dim() == 2:
                quantized = weight / scale
            elif weight.dim() == 4:
                scale_expanded = scale.view(-1, 1, 1, 1)
                quantized = weight / scale_expanded
            
            quantized = quantized.round().clamp(
                -128 if self.is_signed else 0,
                127 if self.is_signed else 255
            ).to(torch.int8)
            
            return quantized, scale.squeeze()
        else:
            abs_max = torch.abs(weight).max()
            scale = abs_max / (127.0 if self.is_signed else 255.0)
            scale = scale.clamp(min=1e-8)
            
            quantized = weight / scale
            quantized = quantized.round().clamp(
                -128 if self.is_signed else 0,
                127 if self.is_signed else 255
            ).to(torch.int8)
            
            return quantized, scale
    
    def quantize_model(
        self,
        model: nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        
        quantized_weights = {}
        scales = {}
        
        for name, module in model.named_modules():
            if not self.should_quantize(name, module):
                continue
            
            if isinstance(module, nn.Linear) and hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.data
                quantized_weight, scale = self.quantize_tensor(weight)
                quantized_weights[name + '.weight'] = quantized_weight
                scales[name + '.weight'] = scale
                print(f"Quantized {name}.weight: shape={weight.shape}, scale_shape={scale.shape if isinstance(scale, torch.Tensor) else 'scalar'}")
            
            if isinstance(module, nn.Conv2d) and hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.data
                quantized_weight, scale = self.quantize_tensor(weight)
                quantized_weights[name + '.weight'] = quantized_weight
                scales[name + '.weight'] = scale
                print(f"Quantized {name}.weight: shape={weight.shape}, scale_shape={scale.shape if isinstance(scale, torch.Tensor) else 'scalar'}")
            
            if isinstance(module, nn.Embedding) and hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.data
                quantized_weight, scale = self.quantize_tensor(weight)
                quantized_weights[name + '.weight'] = quantized_weight
                scales[name + '.weight'] = scale
                print(f"Quantized {name}.weight: shape={weight.shape}, scale_shape={scale.shape if isinstance(scale, torch.Tensor) else 'scalar'}")
        
        return quantized_weights, scales


def load_eva_model(
    model_name_or_path: str,
    num_labels: int = 1000,
    token: Optional[str] = None,
    return_wrapper: bool = False
) -> nn.Module:
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
        return model.model
    elif hasattr(model, 'timm_model'):
        return model.timm_model
    else:
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


def analyze_layer_types(model: nn.Module) -> Dict[str, List[str]]:
    """Analyze and group layers by type."""
    layer_groups = {
        'attention': [],
        'mlp': [],
        'embedding': [],
        'norm': [],
        'head': [],
        'other': []
    }
    
    identifier = LayerTypeIdentifier()
    
    for name, module in model.named_modules():
        layer_type = identifier.identify_layer_type(name, module)
        if layer_type in layer_groups:
            layer_groups[layer_type].append(name)
    
    return layer_groups


def print_layer_analysis(layer_groups: Dict[str, List[str]]):
    """Print analysis of layer types."""
    print("\n=== Layer Type Analysis ===")
    for layer_type, names in layer_groups.items():
        if names:
            print(f"\n{layer_type.upper()} layers ({len(names)}):")
            for name in names[:10]:  # Show first 10
                print(f"  - {name}")
            if len(names) > 10:
                print(f"  ... and {len(names) - 10} more")


def quantize_manual(
    model_name_or_path: str,
    num_labels: int = 1000,
    attention: bool = False,
    mlp: bool = False,
    embedding: bool = False,
    norm: bool = False,
    head: bool = False,
    other: bool = False,
    quantize_all: bool = False,
    dtype: str = 'qint8',
    per_channel: bool = False,
    verbose: bool = False,
):   
    if quantize_all:
        quantize_config = {
            'attention': True,
            'mlp': True,
            'embedding': True,
            'norm': True,
            'head': True,
            'other': True
        }
    else:
        quantize_config = {
            'attention': attention,
            'mlp': mlp,
            'embedding': embedding,
            'norm': norm,
            'head': head,
            'other': other
        }
    
    
    # Load model (same way as eval_classification.py)
    print(f"Loading model from: {model_name_or_path}")
    model = load_eva_model(model_name_or_path, num_labels)
    model.eval()
    
    # Print model structure
    if verbose:
        print_model_structure(model)
        layer_groups = analyze_layer_types(model)
        print_layer_analysis(layer_groups)
    
    # Check if any layer type is selected for quantization
    if not any(quantize_config.values()):
        print("\nWARNING: No layer types selected for quantization!")
        print("Use --quantize-* flags or --quantize-all to enable quantization.")
        return
    
    print(f"\n=== Manual Quantization Configuration ===")
    for layer_type, enabled in quantize_config.items():
        status = "ENABLED" if enabled else "DISABLED"
        print(f"  {layer_type}: {status}")
    print(f"Quantization dtype: {dtype}")
    print(f"Per-channel: {per_channel}")
    
    # Create quantizer
    quantizer = ManualQuantizer(
        quantize_config=quantize_config,
        dtype=dtype,
        per_channel=per_channel
    )
    
    # TODO: create class with forward and backward pass that quantized weights and scales
    quantized_weights, scales = quantizer.quantize_model(model)
    quantized_model = ...
    return quantized_model

