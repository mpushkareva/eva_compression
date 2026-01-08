#!/usr/bin/env python3
"""
Manual quantization script for EVA-transformer models.

This script manually quantizes EVA-transformer models by converting weights
to int8 format and saving float scales for each weight tensor.
Analogous to quantize_eva.py but with manual quantization implementation.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, TimmWrapperForImageClassification


class LayerTypeIdentifier:
    """Identifies different layer types in EVA-transformer models."""
    
    # Common layer type patterns in transformer models
    ATTENTION_PATTERNS = [
        'attn', 'attention', 'self_attn', 'self_attention',
        'qkv', 'q_proj', 'k_proj', 'v_proj', 'out_proj'
    ]
    
    MLP_PATTERNS = [
        'mlp', 'feed_forward', 'ffn', 'fc1', 'fc2',
        'gate_proj', 'up_proj', 'down_proj'
    ]
    
    EMBEDDING_PATTERNS = [
        'embed', 'patch_embed', 'pos_embed', 'cls_token'
    ]
    
    NORM_PATTERNS = [
        'norm', 'ln', 'layer_norm', 'group_norm'
    ]
    
    HEAD_PATTERTERNS = [
        'head', 'classifier', 'fc', 'proj'
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
    """Manually quantizes model weights to int8 format with float scales."""
    
    def __init__(
        self,
        quantize_config: Dict[str, bool],
        dtype: str = 'qint8',
        per_channel: bool = False,
    ):
        """
        Initialize the manual quantizer.
        
        Args:
            quantize_config: Dictionary mapping layer types to boolean flags
                (e.g., {'attention': True, 'mlp': True, 'embedding': False})
            dtype: Quantization dtype ('qint8' for signed, 'quint8' for unsigned)
            per_channel: Whether to use per-channel quantization
        """
        self.quantize_config = quantize_config
        self.dtype = dtype
        self.per_channel = per_channel
        self.layer_identifier = LayerTypeIdentifier()
        self.is_signed = (dtype == 'qint8')
        
    def should_quantize(self, module_name: str, module: nn.Module) -> bool:
        """Check if a module should be quantized based on its type."""
        layer_type = self.layer_identifier.identify_layer_type(module_name, module)
        return self.quantize_config.get(layer_type, False)
    
    def quantize_tensor(
        self,
        weight: torch.Tensor,
        per_channel: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Manually quantize a weight tensor to int8.
        
        Args:
            weight: Float weight tensor to quantize
            per_channel: Whether to use per-channel quantization (overrides self.per_channel)
        
        Returns:
            Tuple of (quantized_int8_tensor, scale_tensor)
            - quantized_int8_tensor: int8 quantized weights
            - scale_tensor: float scale(s) for dequantization
        """
        if per_channel is None:
            per_channel = self.per_channel
        
        if per_channel and weight.dim() > 1:
            # Per-channel quantization: compute scale for each output channel
            # For 2D tensors (Linear): scale per output channel (first dimension)
            # For 4D tensors (Conv2d): scale per output channel (first dimension)
            if weight.dim() == 2:
                # Linear layer: [out_features, in_features]
                # Compute scale per output channel
                abs_max = torch.abs(weight).max(dim=1, keepdim=True)[0]
            elif weight.dim() == 4:
                # Conv2d layer: [out_channels, in_channels, kernel_h, kernel_w]
                # Compute scale per output channel
                abs_max = torch.abs(weight).view(weight.size(0), -1).max(dim=1, keepdim=True)[0]
            else:
                # Fallback to per-tensor
                abs_max = torch.abs(weight).max()
                scale = abs_max / (127.0 if self.is_signed else 255.0)
                scale = scale.clamp(min=1e-8)  # Avoid division by zero
                
                quantized = weight / scale
                quantized = quantized.round().clamp(
                    -128 if self.is_signed else 0,
                    127 if self.is_signed else 255
                ).to(torch.int8)
                
                return quantized, scale
            
            # Per-channel scale
            scale = abs_max / (127.0 if self.is_signed else 255.0)
            scale = scale.clamp(min=1e-8)  # Avoid division by zero
            
            # Quantize per channel
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
            # Per-tensor quantization
            abs_max = torch.abs(weight).max()
            scale = abs_max / (127.0 if self.is_signed else 255.0)
            scale = scale.clamp(min=1e-8)  # Avoid division by zero
            
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
        """
        Manually quantize model weights.
        
        Args:
            model: Model to quantize
        
        Returns:
            Tuple of (quantized_weights_dict, scales_dict)
            - quantized_weights_dict: Dictionary mapping parameter names to int8 tensors
            - scales_dict: Dictionary mapping parameter names to float scale tensors
        """
        quantized_weights = {}
        scales = {}
        
        for name, module in model.named_modules():
            if not self.should_quantize(name, module):
                continue
            
            # Quantize Linear layer weights
            if isinstance(module, nn.Linear) and hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.data
                quantized_weight, scale = self.quantize_tensor(weight)
                quantized_weights[name + '.weight'] = quantized_weight
                scales[name + '.weight'] = scale
                print(f"Quantized {name}.weight: shape={weight.shape}, scale_shape={scale.shape if isinstance(scale, torch.Tensor) else 'scalar'}")
            
            # Quantize Conv2d layer weights
            if isinstance(module, nn.Conv2d) and hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.data
                quantized_weight, scale = self.quantize_tensor(weight)
                quantized_weights[name + '.weight'] = quantized_weight
                scales[name + '.weight'] = scale
                print(f"Quantized {name}.weight: shape={weight.shape}, scale_shape={scale.shape if isinstance(scale, torch.Tensor) else 'scalar'}")
            
            # Quantize Embedding layer weights
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
    
    # Load model exactly as in eval_classification.py
    model = TimmWrapperForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        token=token,
    )
    
    if return_wrapper:
        return model
    
    # Get the underlying timm model (same structure as used in eval_classification.py)
    # TimmWrapperForImageClassification wraps the timm model
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


def main():
    parser = argparse.ArgumentParser(
        description="Manually quantize EVA-transformer model weights to int8 with float scales"
    )
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
        help="Output path to save quantized weights and scales"
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=1000,
        help="Number of classification labels (default: 1000 for ImageNet)"
    )
    # Set defaults for layer quantization (enabled by default for most layers)
    parser.set_defaults(
        quantize_attention=True,
        quantize_mlp=True,
        quantize_embedding=True,
        quantize_norm=False,
        quantize_head=True
    )
    
    parser.add_argument(
        "--quantize-attention",
        dest="quantize_attention",
        action="store_true",
        help="Quantize attention layers (default: enabled)"
    )
    parser.add_argument(
        "--no-quantize-attention",
        dest="quantize_attention",
        action="store_false",
        help="Disable quantization of attention layers"
    )
    parser.add_argument(
        "--quantize-mlp",
        dest="quantize_mlp",
        action="store_true",
        help="Quantize MLP/feed-forward layers (default: enabled)"
    )
    parser.add_argument(
        "--no-quantize-mlp",
        dest="quantize_mlp",
        action="store_false",
        help="Disable quantization of MLP/feed-forward layers"
    )
    parser.add_argument(
        "--quantize-embedding",
        dest="quantize_embedding",
        action="store_true",
        help="Quantize embedding layers (default: enabled)"
    )
    parser.add_argument(
        "--no-quantize-embedding",
        dest="quantize_embedding",
        action="store_false",
        help="Disable quantization of embedding layers"
    )
    parser.add_argument(
        "--quantize-norm",
        dest="quantize_norm",
        action="store_true",
        help="Quantize normalization layers (default: disabled)"
    )
    parser.add_argument(
        "--quantize-head",
        dest="quantize_head",
        action="store_true",
        help="Quantize classification head (default: enabled)"
    )
    parser.add_argument(
        "--no-quantize-head",
        dest="quantize_head",
        action="store_false",
        help="Disable quantization of classification head"
    )
    parser.add_argument(
        "--quantize-all",
        action="store_true",
        help="Quantize all layer types (overrides individual flags)"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable all quantization (useful for analysis only)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="qint8",
        choices=["qint8", "quint8"],
        help="Quantization dtype (qint8 for signed, quint8 for unsigned)"
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Use per-channel quantization (more accurate but larger scale storage)"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze layer types without quantizing"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional)"
    )
    
    args = parser.parse_args()
    
    # Build quantization config
    if args.no_quantize:
        # Disable all quantization
        quantize_config = {
            'attention': False,
            'mlp': False,
            'embedding': False,
            'norm': False,
            'head': False,
            'other': False
        }
    elif args.quantize_all:
        # Quantize everything
        quantize_config = {
            'attention': True,
            'mlp': True,
            'embedding': True,
            'norm': True,
            'head': True,
            'other': True
        }
    else:
        # Use individual flags with defaults
        quantize_config = {
            'attention': getattr(args, 'quantize_attention', True),
            'mlp': getattr(args, 'quantize_mlp', True),
            'embedding': getattr(args, 'quantize_embedding', True),
            'norm': getattr(args, 'quantize_norm', False),
            'head': getattr(args, 'quantize_head', True),
            'other': False
        }
    
    # Load model (same way as eval_classification.py)
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
    
    # Print model structure
    print_model_structure(model)
    
    # Analyze layer types
    layer_groups = analyze_layer_types(model)
    print_layer_analysis(layer_groups)
    
    if args.analyze_only:
        print("\nAnalysis complete. Exiting without quantization.")
        return
    
    # Check if any layer type is selected for quantization
    if not any(quantize_config.values()):
        print("\nWARNING: No layer types selected for quantization!")
        print("Use --quantize-* flags or --quantize-all to enable quantization.")
        return
    
    print(f"\n=== Manual Quantization Configuration ===")
    for layer_type, enabled in quantize_config.items():
        status = "ENABLED" if enabled else "DISABLED"
        print(f"  {layer_type}: {status}")
    print(f"Quantization dtype: {args.dtype}")
    print(f"Per-channel: {args.per_channel}")
    
    # Create quantizer
    quantizer = ManualQuantizer(
        quantize_config=quantize_config,
        dtype=args.dtype,
        per_channel=args.per_channel
    )
    
    # Apply manual quantization
    print(f"\nApplying manual quantization...")
    try:
        quantized_weights, scales = quantizer.quantize_model(model)
        print(f"Quantization completed successfully! Quantized {len(quantized_weights)} weight tensors.")
    except Exception as e:
        print(f"Error during quantization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save quantized weights and scales
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save quantized weights (int8)
    weights_path = output_path.with_suffix('.weights.pth')
    print(f"\nSaving quantized int8 weights to: {weights_path}")
    torch.save(quantized_weights, weights_path)
    
    # Save scales (float)
    scales_path = output_path.with_suffix('.scales.pth')
    print(f"Saving float scales to: {scales_path}")
    torch.save(scales, scales_path)
    
    # Save metadata (quantization config and info)
    metadata = {
        'quantize_config': quantize_config,
        'dtype': args.dtype,
        'per_channel': args.per_channel,
        'num_quantized_tensors': len(quantized_weights),
        'model_name': args.model,
    }
    metadata_path = output_path.with_suffix('.metadata.pth')
    print(f"Saving metadata to: {metadata_path}")
    torch.save(metadata, metadata_path)
    
    # Print summary
    print(f"\n=== Quantization Summary ===")
    print(f"Quantized weights saved to: {weights_path}")
    print(f"Scales saved to: {scales_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Total quantized tensors: {len(quantized_weights)}")
    
    # Calculate size reduction
    total_float_size = sum(w.numel() * 4 for w in quantized_weights.values())  # 4 bytes per float32
    total_int8_size = sum(w.numel() * 1 for w in quantized_weights.values())  # 1 byte per int8
    total_scale_size = sum(
        (s.numel() if isinstance(s, torch.Tensor) else 1) * 4 
        for s in scales.values()
    )  # 4 bytes per float32 scale
    
    print(f"\nSize comparison:")
    print(f"  Original (float32): {total_float_size / 1024 / 1024:.2f} MB")
    print(f"  Quantized (int8): {total_int8_size / 1024 / 1024:.2f} MB")
    print(f"  Scales (float32): {total_scale_size / 1024 / 1024:.2f} MB")
    print(f"  Total quantized: {(total_int8_size + total_scale_size) / 1024 / 1024:.2f} MB")
    print(f"  Compression ratio: {total_float_size / (total_int8_size + total_scale_size):.2f}x")
    
    print("\nManual quantization script completed!")


if __name__ == "__main__":
    main()

