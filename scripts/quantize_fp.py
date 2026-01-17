#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, TimmWrapperForImageClassification

import qtorch
from qtorch import FixedPoint, FloatingPoint
from qtorch.quant import Quantizer


class LayerTypeIdentifier:
    """Identifies different layer types in EVA-transformer models."""
    
    # Common layer type patterns in transformer models
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


class QPyTorchQuantizer:
    
    def __init__(
        self,
        quantize_config: Dict[str, bool],
        forward_num: Optional[FixedPoint] = FixedPoint(wl=8, fl=2),
        backward_num: Optional[FloatingPoint] = FloatingPoint(exp=5, man=2),
        forward_rounding: str = "nearest",
        backward_rounding: str = "nearest",
    ):
        self.quantize_config = quantize_config
        self.layer_identifier = LayerTypeIdentifier()
        
        self.forward_num = forward_num
        self.backward_num = backward_num
        self.forward_rounding = forward_rounding
        self.backward_rounding = backward_rounding
        
        # Create quantizer instances for different layer types
        self.quantizers = {}
        for layer_type in quantize_config.keys():
            if quantize_config.get(layer_type, False):
                self.quantizers[layer_type] = Quantizer(
                    forward_number=forward_num,
                    backward_number=backward_num,
                    forward_rounding=forward_rounding,
                    backward_rounding=backward_rounding
                )
    
    def should_quantize(self, module_name: str, module: nn.Module) -> bool:
        layer_type = self.layer_identifier.identify_layer_type(module_name, module)
        return self.quantize_config.get(layer_type, False)
    
    def get_quantizer(self, module_name: str, module: nn.Module) -> Optional[Quantizer]:
        layer_type = self.layer_identifier.identify_layer_type(module_name, module)
        return self.quantizers.get(layer_type)
    
    def wrap_module(self, module: nn.Module, quantizer: Quantizer) -> nn.Module:
        class QuantizedWrapper(nn.Module):
            def __init__(self, module, quantizer):
                super().__init__()
                self.module = module
                self.quantizer = quantizer
            
            def forward(self, *args, **kwargs):
                out = self.module(*args, **kwargs)
                return self.quantizer(out)
            
            def __getattr__(self, name):
                # Forward attribute access to the wrapped module
                # This preserves access to weight, bias, etc.
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.module, name)
        
        return QuantizedWrapper(module, quantizer)
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        module_info = {}
        
        for name, module in model.named_modules():
            if self.should_quantize(name, module):
                # Check if it's a quantizable layer type
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
                    # Find parent and attribute name
                    name_parts = name.split('.')
                    parent_name = '.'.join(name_parts[:-1]) if len(name_parts) > 1 else ''
                    attr_name = name_parts[-1]
                    
                    module_info[name] = {
                        'module': module,
                        'parent_name': parent_name,
                        'attr_name': attr_name
                    }
        
        for name, info in module_info.items():
            quantizer = self.get_quantizer(name, info['module'])
            if quantizer is not None:
                print(f"Wrapping {name} with Quantizer")
                
                # Get parent module
                if info['parent_name']:
                    # Navigate to parent module
                    parent = model
                    for part in info['parent_name'].split('.'):
                        parent = getattr(parent, part)
                else:
                    parent = model
                
                # Wrap the module
                wrapped = self.wrap_module(info['module'], quantizer)
                setattr(parent, info['attr_name'], wrapped)
        
        return model


def load_eva_model(
    model_name_or_path: str,
    num_labels: int = 1000,
    token: Optional[str] = None
) -> nn.Module:
    if token is None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token is None:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
    
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
        description="Quantize EVA-transformer model using QPyTorch wrapper approach"
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
        help="Output path to save quantized model"
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=1000,
        help="Number of classification labels (default: 1000 for ImageNet)"
    )
    # Set defaults for layer quantization (enabled by default for most layers)
    parser.set_defaults(
        quantize_attention=False,
        quantize_mlp=False,
        quantize_embedding=False,
        quantize_norm=False,
        quantize_head=False
    )
    
    parser.add_argument(
        "--quantize-attention",
        dest="quantize_attention",
        action="store_true",
        help="Quantize attention layers (default: disabled)"
    )
    parser.add_argument(
        "--quantize-mlp",
        dest="quantize_mlp",
        action="store_true",
        help="Quantize MLP/feed-forward layers (default: disabled)"
    )
    parser.add_argument(
        "--quantize-embedding",
        dest="quantize_embedding",
        action="store_true",
        help="Quantize embedding layers (default: disabled)"
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
        help="Quantize classification head (default: disabled)"
    )
    parser.add_argument(
        "--quantize-all",
        action="store_true",
        help="Quantize all layer types (overrides individual flags, default: disabled)"
    )
    # QPyTorch specific parameters
    parser.add_argument(
        "--forward-wl",
        type=int,
        default=8,
        help="Forward quantization word length for FixedPoint (default: 8)"
    )
    parser.add_argument(
        "--forward-fl",
        type=int,
        default=4,
        help="Forward quantization fractional length for FixedPoint (default: 2)"
    )
    parser.add_argument(
        "--forward-exp",
        type=int,
        default=4,
        help="Forward quantization exponent bits for FloatingPoint (default: 5)"
    )
    parser.add_argument(
        "--forward-man",
        type=int,
        default=4,
        help="Forward quantization mantissa bits for FloatingPoint (default: 2)"
    )
    parser.add_argument(
        "--backward-exp",
        type=int,
        default=5,
        help="Backward quantization exponent bits for FloatingPoint (default: 5)"
    )
    parser.add_argument(
        "--backward-man",
        type=int,
        default=2,
        help="Backward quantization mantissa bits for FloatingPoint (default: 2)"
    )
    parser.add_argument(
        "--forward-rounding",
        type=str,
        default="nearest",
        choices=["nearest", "stochastic"],
        help="Forward rounding mode (default: nearest)"
    )
    parser.add_argument(
        "--backward-rounding",
        type=str,
        default="nearest",
        choices=["nearest", "stochastic"],
        help="Backward rounding mode (default: nearest)"
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
    
    if args.quantize_all:
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
            'attention': getattr(args, 'quantize_attention', True),
            'mlp': getattr(args, 'quantize_mlp', True),
            'embedding': getattr(args, 'quantize_embedding', True),
            'norm': getattr(args, 'quantize_norm', False),
            'head': getattr(args, 'quantize_head', True),
            'other': False
        }
    
    print(f"Loading model from: {args.model}")
    model_wrapper = load_eva_model(args.model, args.num_labels, args.token, return_wrapper=True)
    model_wrapper.eval()
    
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
    
    layer_groups = analyze_layer_types(model)
    print_layer_analysis(layer_groups)
    
    if args.analyze_only:
        print("\nAnalysis complete. Exiting without quantization.")
        return
    
    if not any(quantize_config.values()):
        print("\nWARNING: No layer types selected for quantization!")
        print("Use --quantize-* flags or --quantize-all to enable quantization.")
        return
    
    # Create number formats for QPyTorch
    if args.forward_format == "fixed":
        forward_num = FixedPoint(wl=args.forward_wl, fl=args.forward_fl)
        print(f"Forward format: FixedPoint(wl={args.forward_wl}, fl={args.forward_fl})")
    else:
        forward_num = FloatingPoint(exp=args.forward_exp, man=args.forward_man)
        print(f"Forward format: FloatingPoint(exp={args.forward_exp}, man={args.forward_man})")
    
    backward_num = FloatingPoint(exp=args.backward_exp, man=args.backward_man)
    print(f"Backward format: FloatingPoint(exp={args.backward_exp}, man={args.backward_man})")
    
    quantizer = QPyTorchQuantizer(
        quantize_config=quantize_config,
        forward_num=forward_num,
        backward_num=backward_num,
        forward_rounding=args.forward_rounding,
        backward_rounding=args.backward_rounding
    )
    
    quantized_model = quantizer.quantize_model(model)
    print(f"Quantization for {args.model} completed successfully!")
    
    # if hasattr(model_wrapper, 'model'):
    #     model_wrapper.model = quantized_model
    # elif hasattr(model_wrapper, 'timm_model'):
    #     model_wrapper.timm_model = quantized_model
    # else:
    #     model_wrapper = quantized_model
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving quantized model to: {output_path}")
    torch.save(quantized_model, output_path)
    print("\nQuantization script completed!")

if __name__ == "__main__":
    main()

