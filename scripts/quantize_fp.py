#!/usr/bin/env python3
"""
QPyTorch quantization script for EVA-transformer models.

This script quantizes EVA-transformer models using QPyTorch's wrapper approach
for layers. Uses the same parameters as quantize_manual.py but applies
QPyTorch's Quantizer modules to wrap layers.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, TimmWrapperForImageClassification

# QPyTorch imports
try:
    import qtorch
    from qtorch import FixedPoint, FloatingPoint
    from qtorch.quant import Quantizer
except ImportError as e:
    if "setuptools" in str(e).lower():
        print("ERROR: setuptools is required for QPyTorch but is not installed.")
        print("Please install it by running:")
        print("  pip install setuptools")
        print("Or if using the virtual environment:")
        print("  eva/bin/python -m pip install setuptools")
        raise ImportError(
            "setuptools is required for QPyTorch. "
            "Install it with: pip install setuptools"
        ) from e
    else:
        raise


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


class QPyTorchQuantizer:
    """Quantizes model using QPyTorch's Quantizer wrapper approach."""
    
    def __init__(
        self,
        quantize_config: Dict[str, bool],
        forward_num: Optional[FixedPoint] = None,
        backward_num: Optional[FloatingPoint] = None,
        forward_rounding: str = "nearest",
        backward_rounding: str = "nearest",
    ):
        """
        Initialize the QPyTorch quantizer.
        
        Args:
            quantize_config: Dictionary mapping layer types to boolean flags
                (e.g., {'attention': True, 'mlp': True, 'embedding': False})
            forward_num: Forward quantization number format (FixedPoint or FloatingPoint)
            backward_num: Backward quantization number format (FixedPoint or FloatingPoint)
            forward_rounding: Forward rounding mode ("nearest" or "stochastic")
            backward_rounding: Backward rounding mode ("nearest" or "stochastic")
        """
        self.quantize_config = quantize_config
        self.layer_identifier = LayerTypeIdentifier()
        
        # Default number formats if not provided
        if forward_num is None:
            # Default: 8-bit fixed point with 2 fractional bits
            forward_num = FixedPoint(wl=8, fl=2)
        if backward_num is None:
            # Default: 8-bit floating point with 5 exp bits and 2 mantissa bits
            backward_num = FloatingPoint(exp=5, man=2)
        
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
        """Check if a module should be quantized based on its type."""
        layer_type = self.layer_identifier.identify_layer_type(module_name, module)
        return self.quantize_config.get(layer_type, False)
    
    def get_quantizer(self, module_name: str, module: nn.Module) -> Optional[Quantizer]:
        """Get the appropriate quantizer for a module."""
        layer_type = self.layer_identifier.identify_layer_type(module_name, module)
        return self.quantizers.get(layer_type)
    
    def wrap_module(self, module: nn.Module, quantizer: Quantizer) -> nn.Module:
        """
        Wrap a module with a quantizer.
        
        Creates a wrapper that preserves the original module's interface
        but applies quantization to the output.
        """
        class QuantizedWrapper(nn.Module):
            """Wrapper that applies quantization to module output."""
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
        """
        Quantize model by wrapping selected layers with QPyTorch Quantizers.
        
        This method modifies the model in-place by replacing quantizable layers
        with wrappers that apply quantization to the output.
        
        Args:
            model: Model to quantize (modified in-place)
            
        Returns:
            The same model (modified in-place)
        """
        # Build a mapping of module names to their parent modules and attribute names
        # We need to do this before modifying the model structure
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
        
        # Now wrap modules (we need to get parents fresh each time since structure may change)
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
        default=2,
        help="Forward quantization fractional length for FixedPoint (default: 2)"
    )
    parser.add_argument(
        "--forward-exp",
        type=int,
        default=5,
        help="Forward quantization exponent bits for FloatingPoint (default: 5)"
    )
    parser.add_argument(
        "--forward-man",
        type=int,
        default=2,
        help="Forward quantization mantissa bits for FloatingPoint (default: 2)"
    )
    parser.add_argument(
        "--forward-format",
        type=str,
        default="fixed",
        choices=["fixed", "float"],
        help="Forward quantization format: fixed or float (default: fixed)"
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
    
    # Create number formats for QPyTorch
    if args.forward_format == "fixed":
        forward_num = FixedPoint(wl=args.forward_wl, fl=args.forward_fl)
        print(f"Forward format: FixedPoint(wl={args.forward_wl}, fl={args.forward_fl})")
    else:
        forward_num = FloatingPoint(exp=args.forward_exp, man=args.forward_man)
        print(f"Forward format: FloatingPoint(exp={args.forward_exp}, man={args.forward_man})")
    
    backward_num = FloatingPoint(exp=args.backward_exp, man=args.backward_man)
    print(f"Backward format: FloatingPoint(exp={args.backward_exp}, man={args.backward_man})")
    
    print(f"\n=== QPyTorch Quantization Configuration ===")
    for layer_type, enabled in quantize_config.items():
        status = "ENABLED" if enabled else "DISABLED"
        print(f"  {layer_type}: {status}")
    print(f"Forward rounding: {args.forward_rounding}")
    print(f"Backward rounding: {args.backward_rounding}")
    
    # Create quantizer
    quantizer = QPyTorchQuantizer(
        quantize_config=quantize_config,
        forward_num=forward_num,
        backward_num=backward_num,
        forward_rounding=args.forward_rounding,
        backward_rounding=args.backward_rounding
    )
    
    # Apply QPyTorch quantization
    print(f"\nApplying QPyTorch quantization wrappers...")
    try:
        quantized_model = quantizer.quantize_model(model)
        print(f"Quantization completed successfully!")
    except Exception as e:
        print(f"Error during quantization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save quantized model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the quantized model
    print(f"\nSaving quantized model to: {output_path}")
    torch.save(quantized_model.state_dict(), output_path)
    
    # Save metadata
    metadata = {
        'quantize_config': quantize_config,
        'forward_format': args.forward_format,
        'forward_wl': args.forward_wl if args.forward_format == "fixed" else None,
        'forward_fl': args.forward_fl if args.forward_format == "fixed" else None,
        'forward_exp': args.forward_exp if args.forward_format == "float" else None,
        'forward_man': args.forward_man if args.forward_format == "float" else None,
        'backward_exp': args.backward_exp,
        'backward_man': args.backward_man,
        'forward_rounding': args.forward_rounding,
        'backward_rounding': args.backward_rounding,
        'model_name': args.model,
    }
    metadata_path = output_path.with_suffix('.metadata.pth')
    print(f"Saving metadata to: {metadata_path}")
    torch.save(metadata, metadata_path)
    
    # Print summary
    print(f"\n=== Quantization Summary ===")
    print(f"Quantized model saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")
    print("\nQPyTorch quantization script completed!")


if __name__ == "__main__":
    main()

