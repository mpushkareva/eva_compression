#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedPointFormat:
    """Fixed-point number format specification."""
    def __init__(self, wl: int, fl: int, rounding: str = "nearest"):
        self.wl = wl  # Word length (total bits)
        self.fl = fl  # Fractional length
        self.rounding = rounding
        self.scale = 2.0 ** (-fl)
        # Calculate range (signed fixed-point: 1 sign bit, wl-1-fl integer bits, fl fractional bits)
        self.min_val = -(2.0 ** (wl - fl - 1))
        self.max_val = 2.0 ** (wl - fl - 1) - self.scale
        
    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Quantize tensor to fixed-point representation.
        Simulates: round(tensor / scale) * scale, clamped to representable range.
        """
        if tensor is None:
            return None
            
        # Scale up to integer domain
        scaled = tensor / self.scale
        
        # Round according to specified mode
        if self.rounding == "nearest":
            rounded = torch.round(scaled)
        elif self.rounding == "floor":
            rounded = torch.floor(scaled)
        elif self.rounding == "ceil":
            rounded = torch.ceil(scaled)
        elif self.rounding == "trunc":
            rounded = torch.trunc(scaled)
        else:
            rounded = torch.round(scaled)
        
        # Clamp to integer range representable with wl bits (signed)
        min_int = -(2 ** (self.wl - 1))
        max_int = 2 ** (self.wl - 1) - 1
        rounded = torch.clamp(rounded, min_int, max_int)
        
        # Scale back
        quantized = rounded * self.scale
        
        return quantized
    
    def quantize_to_int(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize to integer representation (for storage)."""
        scaled = tensor / self.scale
        if self.rounding == "nearest":
            rounded = torch.round(scaled)
        else:
            rounded = torch.floor(scaled) if self.rounding == "floor" else torch.round(scaled)
        min_int = -(2 ** (self.wl - 1))
        max_int = 2 ** (self.wl - 1) - 1
        return torch.clamp(rounded, min_int, max_int).to(torch.int32)
    
    def dequantize_from_int(self, int_tensor: torch.Tensor) -> torch.Tensor:
        """Dequantize from integer representation."""
        return int_tensor.float() * self.scale


class FixedPointLinear(nn.Module):
    """
    Linear layer with fixed-point weights, biases, and arithmetic.
    Simulates fixed-point matrix multiplication where intermediate results
    are truncated back to fixed-point format.
    """
    def __init__(self, linear: nn.Linear, fp_format: FixedPointFormat):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.fp_format = fp_format
        
        # Store quantization scale for potential debugging/analysis
        self.register_buffer('weight_scale', torch.tensor(fp_format.scale))
        
        # Quantize and store weights as dequantized floats (simulating fixed-point values)
        quantized_weight = fp_format.quantize(linear.weight.data)
        self.register_buffer('weight', quantized_weight)
        
        if linear.bias is not None:
            quantized_bias = fp_format.quantize(linear.bias.data)
            self.register_buffer('bias', quantized_bias)
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize input activations to fixed-point
        x_fixed = self.fp_format.quantize(x)
        
        # Perform matrix multiplication in floating point, but both operands are fixed-point values
        # In true hardware: integer multiplication with scaling
        out = F.linear(x_fixed, self.weight, self.bias)
        
        # Quantize output back to fixed-point (simulates truncation/rounding of extended precision)
        out_fixed = self.fp_format.quantize(out)
        
        return out_fixed
    
    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, wl={self.fp_format.wl}, fl={self.fp_format.fl}')


class FixedPointConv2d(nn.Module):
    """
    Conv2d layer with fixed-point weights, biases, and arithmetic.
    """
    def __init__(self, conv: nn.Conv2d, fp_format: FixedPointFormat):
        super().__init__()
        self.fp_format = fp_format
        
        # Copy convolution parameters
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.padding_mode = conv.padding_mode
        
        # Quantize weights (mimic fixed-point representation)
        quantized_weight = fp_format.quantize(conv.weight.data)
        self.register_buffer('weight', quantized_weight)
        
        if conv.bias is not None:
            quantized_bias = fp_format.quantize(conv.bias.data)
            self.register_buffer('bias', quantized_bias)
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize input to fixed-point
        x_fixed = self.fp_format.quantize(x)
        
        # Handle padding mode
        if self.padding_mode != 'zeros':
            x_fixed = F.pad(x_fixed, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        
        # Perform convolution (operands are fixed-point values)
        out = F.conv2d(x_fixed, self.weight, self.bias, self.stride, 
                      self.padding, self.dilation, self.groups)
        
        # Rescale/quantize output to fixed-point format
        return self.fp_format.quantize(out)
    
    @property
    def _reversed_padding_repeated_twice(self):
        # Helper for padding
        padding = self.padding if isinstance(self.padding, tuple) else (self.padding,) * 2
        return tuple(reversed(padding)) * 2
    
    def extra_repr(self) -> str:
        s = (f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}'
             f', stride={self.stride}')
        if self.padding != (0,) * len(self.padding):
            s += f', padding={self.padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += f', dilation={self.dilation}'
        if self.groups != 1:
            s += f', groups={self.groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += f', padding_mode={self.padding_mode}'
        s += f', wl={self.fp_format.wl}, fl={self.fp_format.fl}'
        return s


class FixedPointConv1d(nn.Module):
    """
    Conv1d layer with fixed-point weights, biases, and arithmetic.
    Used by QKFormer and other models that use 1D convolutions in attention/MLP.
    """
    def __init__(self, conv: nn.Conv1d, fp_format: FixedPointFormat):
        super().__init__()
        self.fp_format = fp_format
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.padding_mode = conv.padding_mode
        quantized_weight = fp_format.quantize(conv.weight.data)
        self.register_buffer('weight', quantized_weight)
        if conv.bias is not None:
            quantized_bias = fp_format.quantize(conv.bias.data)
            self.register_buffer('bias', quantized_bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fixed = self.fp_format.quantize(x)
        if self.padding_mode != 'zeros':
            x_fixed = F.pad(x_fixed, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        out = F.conv1d(x_fixed, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return self.fp_format.quantize(out)

    @property
    def _reversed_padding_repeated_twice(self):
        padding = self.padding if isinstance(self.padding, tuple) else (self.padding,) * 2
        return tuple(reversed(padding)) * 2

    def extra_repr(self) -> str:
        s = (f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}'
             f', stride={self.stride}, wl={self.fp_format.wl}, fl={self.fp_format.fl}')
        return s


class FixedPointLayerNorm(nn.Module):
    """
    LayerNorm with fixed-point parameters and arithmetic simulation.
    Note: LayerNorm involves division (inverse sqrt) which is expensive in fixed-point.
    Here we quantize parameters and outputs to simulate the effect.
    """
    def __init__(self, norm: nn.LayerNorm, fp_format: FixedPointFormat):
        super().__init__()
        self.fp_format = fp_format
        self.normalized_shape = norm.normalized_shape
        self.eps = norm.eps
        self.elementwise_affine = norm.elementwise_affine
        
        if self.elementwise_affine:
            # Quantize weight and bias
            self.register_buffer('weight', fp_format.quantize(norm.weight.data))
            if norm.bias is not None:
                self.register_buffer('bias', fp_format.quantize(norm.bias.data))
            else:
                self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize input
        x_fixed = self.fp_format.quantize(x)
        
        # Perform normalization (in full precision, but input is quantized)
        # Note: True fixed-point LayerNorm is complex due to division/sqrt
        # We quantize the output to simulate limited precision accumulation
        mean = x_fixed.mean(dim=-1, keepdim=True)
        var = ((x_fixed - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x_fixed - mean) / torch.sqrt(var + self.eps)
        
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias
        
        # Quantize final output
        return self.fp_format.quantize(x_norm)


class FixedPointBatchNorm2d(nn.Module):
    """
    BatchNorm2d with fixed-point parameters and arithmetic simulation.
    Quantizes weight (gamma), bias (beta), running statistics, and output.
    """
    def __init__(self, bn: nn.BatchNorm2d, fp_format: FixedPointFormat):
        super().__init__()
        self.fp_format = fp_format
        self.num_features = bn.num_features
        self.eps = bn.eps
        self.momentum = bn.momentum
        self.affine = bn.affine
        self.track_running_stats = bn.track_running_stats

        if self.affine:
            self.register_buffer('weight', fp_format.quantize(bn.weight.data))
            self.register_buffer('bias', fp_format.quantize(bn.bias.data))

        if self.track_running_stats:
            self.register_buffer('running_mean', fp_format.quantize(bn.running_mean.data))
            self.register_buffer('running_var', fp_format.quantize(bn.running_var.data))
            self.register_buffer('num_batches_tracked', bn.num_batches_tracked.clone())
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fixed = self.fp_format.quantize(x)
        out = F.batch_norm(
            x_fixed, self.running_mean, self.running_var,
            self.weight, self.bias, False, 0.0, self.eps
        )
        return self.fp_format.quantize(out)

    def extra_repr(self) -> str:
        return (f'{self.num_features}, eps={self.eps}, affine={self.affine}, '
                f'wl={self.fp_format.wl}, fl={self.fp_format.fl}')


class FixedPointBatchNorm1d(nn.Module):
    """
    BatchNorm1d with fixed-point parameters and arithmetic simulation.
    Quantizes weight (gamma), bias (beta), running statistics, and output.
    """
    def __init__(self, bn: nn.BatchNorm1d, fp_format: FixedPointFormat):
        super().__init__()
        self.fp_format = fp_format
        self.num_features = bn.num_features
        self.eps = bn.eps
        self.momentum = bn.momentum
        self.affine = bn.affine
        self.track_running_stats = bn.track_running_stats

        if self.affine:
            self.register_buffer('weight', fp_format.quantize(bn.weight.data))
            self.register_buffer('bias', fp_format.quantize(bn.bias.data))

        if self.track_running_stats:
            self.register_buffer('running_mean', fp_format.quantize(bn.running_mean.data))
            self.register_buffer('running_var', fp_format.quantize(bn.running_var.data))
            self.register_buffer('num_batches_tracked', bn.num_batches_tracked.clone())
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fixed = self.fp_format.quantize(x)
        out = F.batch_norm(
            x_fixed, self.running_mean, self.running_var,
            self.weight, self.bias, False, 0.0, self.eps
        )
        return self.fp_format.quantize(out)

    def extra_repr(self) -> str:
        return (f'{self.num_features}, eps={self.eps}, affine={self.affine}, '
                f'wl={self.fp_format.wl}, fl={self.fp_format.fl}')


class FixedPointEmbedding(nn.Module):
    """
    Embedding layer with fixed-point weights.
    """
    def __init__(self, embedding: nn.Embedding, fp_format: FixedPointFormat):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.padding_idx = embedding.padding_idx
        self.fp_format = fp_format
        
        # Quantize embedding weights
        quantized_weight = fp_format.quantize(embedding.weight.data)
        self.register_buffer('weight', quantized_weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fp_format.quantize(x)
        out = F.embedding(x, self.weight, self.padding_idx, None, 2.0, False, False)
        # Ensure output is within fixed-point range
        return self.fp_format.quantize(out)


class LayerTypeIdentifier:
    """Identifies different layer types in EVA-transformer and QKFormer-style models."""
    
    # Include q_conv, k_conv, v_conv for QKFormer/SpikingTransformer attention
    ATTENTION_PATTERNS = ['attn', 'qkv', 'proj', 'head', 'q_conv', 'k_conv', 'v_conv', 'tssa', 'ssa']
    MLP_PATTERNS = ['mlp', 'fc1', 'fc2', 'dense', 'fc_conv']
    EMBEDDING_PATTERNS = ['patch_embed', 'pos_embed', 'embed'] # 'rope'
    NORM_PATTERNS = ['norm1', 'norm2', 'ln', 'bn', 'gn']
    HEAD_PATTERNS = ['head', 'classifier', 'fc', 'linear']
    
    @classmethod
    def identify_layer_type(cls, module_name: str, module: nn.Module) -> str:
        """Identify the type of a layer based on its name and type."""
        name_lower = module_name.lower()
        
        if any(pattern in name_lower for pattern in cls.ATTENTION_PATTERNS):
            return 'attention'
        if any(pattern in name_lower for pattern in cls.MLP_PATTERNS):
            return 'mlp'
        if any(pattern in name_lower for pattern in cls.EMBEDDING_PATTERNS):
            return 'embedding'
        if any(pattern in name_lower for pattern in cls.NORM_PATTERNS):
            return 'norm'
        if any(pattern in name_lower for pattern in cls.HEAD_PATTERNS):
            return 'head'
        
        # Check module type
        if isinstance(module, (nn.MultiheadAttention,)):
            return 'attention'
        elif isinstance(module, nn.Conv1d):
            if any(p in name_lower for p in cls.ATTENTION_PATTERNS):
                return 'attention'
            if any(p in name_lower for p in cls.MLP_PATTERNS):
                return 'mlp'
            return 'other'
        elif isinstance(module, (nn.Linear,)):
            if any(p in name_lower for p in cls.HEAD_PATTERNS):
                return 'head'
            elif any(p in name_lower for p in cls.MLP_PATTERNS):
                return 'mlp'
            elif any(p in name_lower for p in cls.ATTENTION_PATTERNS):
                return 'attention'
            return 'other'
        elif isinstance(module, (nn.Embedding, nn.Conv2d)):
            if 'patch' in name_lower or 'embed' in name_lower:
                return 'embedding'
            if any(p in name_lower for p in cls.MLP_PATTERNS):
                return 'mlp'
            return 'other'
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            return 'norm'
        
        return 'other'


class FixedPointModelConverter:
    """
    Converts a PyTorch model to use fixed-point arithmetic by replacing
    standard layers with fixed-point equivalents.
    """
    
    def __init__(
        self,
        quantize_config: Dict[str, bool],
        wl: int = 8,
        fl: int = 4,
        rounding: str = "nearest",
        precision_by_layer_type: Optional[Dict[str, Tuple[int, int]]] = None
    ):
        self.quantize_config = quantize_config
        self.layer_identifier = LayerTypeIdentifier()
        self.default_fp_format = FixedPointFormat(wl=wl, fl=fl, rounding=rounding)
        self.format_by_layer_type: Dict[str, FixedPointFormat] = {}
        if precision_by_layer_type:
            for layer_type, (w, f) in precision_by_layer_type.items():
                self.format_by_layer_type[layer_type] = FixedPointFormat(wl=w, fl=f, rounding=rounding)
        
    def _get_format_for_layer_type(self, layer_type: str) -> FixedPointFormat:
        """Return fixed-point format for the given layer type (override or default)."""
        return self.format_by_layer_type.get(layer_type, self.default_fp_format)
        
    def should_quantize(self, module_name: str, module: nn.Module) -> bool:
        """Check if this module should be converted to fixed-point."""
        layer_type = self.layer_identifier.identify_layer_type(module_name, module)
        return self.quantize_config.get(layer_type, False)
    
    def convert_layer(self, module: nn.Module, layer_type: str) -> Optional[nn.Module]:
        """
        Convert a standard layer to its fixed-point equivalent.
        Returns None if no conversion is available for this layer type.
        """
        fp_format = self._get_format_for_layer_type(layer_type)
        if isinstance(module, nn.Linear):
            return FixedPointLinear(module, fp_format)
        elif isinstance(module, nn.Conv2d):
            return FixedPointConv2d(module, fp_format)
        elif isinstance(module, nn.Conv1d):
            return FixedPointConv1d(module, fp_format)
        elif isinstance(module, nn.LayerNorm):
            return FixedPointLayerNorm(module, fp_format)
        elif isinstance(module, nn.Embedding):
            return FixedPointEmbedding(module, fp_format)
        elif isinstance(module, nn.BatchNorm2d):
            return FixedPointBatchNorm2d(module, fp_format)
        elif isinstance(module, nn.BatchNorm1d):
            return FixedPointBatchNorm1d(module, fp_format)
        return None
    
    def convert_model(self, model: nn.Module) -> nn.Module:
        """
        Traverse the model and replace configurable layers with fixed-point versions.
        """
        module_info = {}
        
        # First pass: identify modules to replace
        for name, module in model.named_modules():
            if self.should_quantize(name, module):
                layer_type = self.layer_identifier.identify_layer_type(name, module)
                converted = self.convert_layer(module, layer_type)
                if converted is not None:
                    fp_format = self._get_format_for_layer_type(layer_type)
                    name_parts = name.split('.')
                    parent_name = '.'.join(name_parts[:-1]) if len(name_parts) > 1 else ''
                    attr_name = name_parts[-1]
                    
                    module_info[name] = {
                        'parent_name': parent_name,
                        'attr_name': attr_name,
                        'replacement': converted,
                        'original_type': type(module).__name__,
                        'fp_format': fp_format
                    }
        
        # Second pass: perform replacements
        for name, info in module_info.items():
            if info['parent_name']:
                parent = model
                for part in info['parent_name'].split('.'):
                    if part:  # Handle empty string for root
                        parent = getattr(parent, part)
            else:
                parent = model
            
            setattr(parent, info['attr_name'], info['replacement'])
            fmt = info['fp_format']
            print(f"Converted {name} ({info['original_type']}) -> FixedPoint (wl={fmt.wl}, fl={fmt.fl})")
        
        return model


def load_eva_model(
    model_name_or_path: str,
    num_labels: int = 1000,
    token: Optional[str] = None
) -> nn.Module:
    """Load EVA model from HuggingFace or local path. Requires transformers."""
    from transformers import TimmWrapperForImageClassification
    if token is None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token is None:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            token = None

    model = TimmWrapperForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        token=token,
    )
    if hasattr(model, 'model'):
        return model.model
    elif hasattr(model, 'timm_model'):
        return model.timm_model
    else:
        print("Warning: Could not find underlying timm model, using wrapper directly")
        return model


def print_model_structure(model: nn.Module, max_depth: int = 3):
    """Print model structure for debugging."""
    print("\n=== Model Structure (first 3 levels) ===")
    for name, module in list(model.named_modules())[:50]:
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
            for name in names[:10]:
                print(f"  - {name}")
            if len(names) > 10:
                print(f"  ... and {len(names) - 10} more")


def apply_fixed_op(
    model: nn.Module,
    quantize_config: Dict[str, bool],
    forward_wl: int = 16,
    forward_fl: int = 8,
    forward_rounding: str = "nearest",
    precision_by_layer_type: Optional[Dict[str, Tuple[int, int]]] = None,
) -> nn.Module:
    """
    Apply fixed-point op-level quantization to an existing nn.Module (e.g. QKFormer).
    Does not load a model; use this when the model is already built (e.g. in STEP/cls/train.py).
    """
    if not any(quantize_config.values()):
        return model
    converter = FixedPointModelConverter(
        quantize_config=quantize_config,
        wl=forward_wl,
        fl=forward_fl,
        rounding=forward_rounding,
        precision_by_layer_type=precision_by_layer_type,
    )
    return converter.convert_model(model)


def parse_precision_by_layer(tokens: List[str]) -> Dict[str, Tuple[int, int]]:
    """
    Parse per-layer precision from CLI tokens like 'attention:8,4' 'mlp:16,8'.
    Returns dict mapping layer_type -> (wl, fl). Invalid tokens raise ValueError.
    """
    result: Dict[str, Tuple[int, int]] = {}
    valid_types = {'attention', 'mlp', 'embedding', 'norm', 'head', 'other'}
    for token in tokens:
        if ':' not in token:
            raise ValueError(f"Invalid precision token '{token}': expected layer_type:wl,fl")
        layer_type, rest = token.split(':', 1)
        layer_type = layer_type.strip().lower()
        if layer_type not in valid_types:
            raise ValueError(f"Unknown layer type '{layer_type}'; valid: {sorted(valid_types)}")
        if ',' not in rest:
            raise ValueError(f"Invalid precision token '{token}': expected layer_type:wl,fl")
        wl_str, fl_str = rest.split(',', 1)
        try:
            wl, fl = int(wl_str.strip()), int(fl_str.strip())
        except ValueError as e:
            raise ValueError(f"Invalid wl,fl in '{token}': {e}") from e
        result[layer_type] = (wl, fl)
    return result


def quantize_fp_op(
    model_name_or_path: str,
    num_labels: int = 1000,
    attention: bool = False,
    mlp: bool = False,
    embedding: bool = False,
    norm: bool = False,
    head: bool = False,
    other: bool = False,
    quantize_all: bool = False,
    forward_format: str = "fixed",
    forward_wl: int = 8,
    forward_fl: int = 4,
    forward_exp: int = 5,
    forward_man: int = 2,
    backward_exp: int = 5,
    backward_man: int = 2,
    forward_rounding: str = "nearest",
    backward_rounding: str = "nearest",
    verbose: bool = False,
    precision_by_layer_type: Optional[Dict[str, Tuple[int, int]]] = None,
):
    """
    Quantize model to fixed-point representation.
    
    Args follow the signature from eval_quant.py.
    backward_* parameters are kept for API compatibility but not used in inference.
    precision_by_layer_type: optional dict mapping layer type (e.g. 'attention', 'mlp')
        to (wl, fl). Unspecified types use forward_wl/forward_fl.
    """
    # Build quantization config
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

    print(f"Loading model from: {model_name_or_path}")
    model = load_eva_model(model_name_or_path, num_labels)
    model.eval()
    
    if verbose:
        print_model_structure(model)
        layer_groups = analyze_layer_types(model)
        print_layer_analysis(layer_groups)
        return
    
    if not any(quantize_config.values()):
        print("\nWARNING: No layer types selected for quantization!")
        print("Use --quantize-* flags or --quantize-all to enable quantization.")
        return model
    
    # Configure fixed-point format (only forward format is used for inference)
    if forward_format == "fixed":
        print(f"Configuring FixedPoint default (wl={forward_wl}, fl={forward_fl}, rounding={forward_rounding})")
    else:
        print(f"Warning: forward_format={forward_format} not 'fixed', but proceeding with fixed-point")
        print(f"Using FixedPoint(wl={forward_wl}, fl={forward_fl})")
    if precision_by_layer_type:
        print("Per-layer precision overrides:")
        for layer_type, (wl, fl) in sorted(precision_by_layer_type.items()):
            print(f"  {layer_type}: wl={wl}, fl={fl}")
    
    # Create converter and transform model
    converter = FixedPointModelConverter(
        quantize_config=quantize_config,
        wl=forward_wl,
        fl=forward_fl,
        rounding=forward_rounding,
        precision_by_layer_type=precision_by_layer_type
    )
    
    quantized_model = converter.convert_model(model)
    print(f"\nFixed-point quantization completed for {model_name_or_path}")
    print(f"Quantized layers: attention={attention}, mlp={mlp}, embedding={embedding}, "
          f"norm={norm}, head={head}, other={other} (or all={quantize_all})")
    
    return quantized_model


if __name__ == "__main__":
    # CLI for testing quantization standalone
    parser = argparse.ArgumentParser(description="Fixed-Point Quantization Tool")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF identifier")
    parser.add_argument("--num-labels", type=int, default=1000)
    parser.add_argument("--wl", type=int, default=8, help="Word length (total bits)")
    parser.add_argument("--fl", type=int, default=4, help="Fractional length")
    parser.add_argument(
        "--precision-by-layer",
        type=str,
        nargs="*",
        default=None,
        metavar="KEY:WL,FL",
        help="Per-layer precision (e.g. attention:8,4 mlp:16,8). Unspecified types use --wl/--fl.",
    )
    parser.add_argument("--forward-rounding", type=str, default="nearest", help="Rounding mode")
    parser.add_argument("--quantize-all", action="store_true")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--mlp", action="store_true")
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--head", action="store_true")
    parser.add_argument("--other", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    precision_by_layer_type = None
    if args.precision_by_layer:
        precision_by_layer_type = parse_precision_by_layer(args.precision_by_layer)

    model = quantize_fp_op(
        args.model,
        args.num_labels,
        attention=args.attention or args.quantize_all,
        mlp=args.mlp or args.quantize_all,
        embedding=args.embedding or args.quantize_all,
        norm=args.norm or args.quantize_all,
        head=args.head or args.quantize_all,
        other=args.other or args.quantize_all,
        quantize_all=args.quantize_all,
        forward_wl=args.wl,
        forward_fl=args.fl,
        forward_rounding=args.forward_rounding,
        precision_by_layer_type=precision_by_layer_type,
        verbose=args.verbose,
    )

    if model and not args.verbose:
        print("\nModel quantized successfully. Testing forward pass...")
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output shape: {output.shape}")
