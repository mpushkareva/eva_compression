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
from torch.ao.quantization import (
    QConfigMapping,
    prepare,
    convert,
    prepare_qat,
)
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


class SelectiveQuantizer:
    """Quantizes model layers selectively based on layer type."""
    
    def __init__(
        self,
        quantize_config: Dict[str, bool],
        quantization_mode: str = 'static',
        dtype: torch.dtype = torch.qint8,
        per_channel: bool = False,
        bits: int = 8,
    ):
        """
        Initialize the quantizer.
        
        Args:
            quantize_config: Dictionary mapping layer types to boolean flags
                (e.g., {'attention': True, 'mlp': True, 'embedding': False})
            quantization_mode: 'static', 'dynamic', or 'qat'
            dtype: Quantization dtype (torch.qint8 or torch.quint8)
            per_channel: Whether to use per-channel quantization
            bits: Quantization bit-width (4 or 8). Note: 4-bit requires special handling
        """
        self.quantize_config = quantize_config
        self.quantization_mode = quantization_mode
        self.dtype = dtype
        self.per_channel = per_channel
        self.bits = bits
        self.layer_identifier = LayerTypeIdentifier()
        
        if bits not in [4, 8]:
            raise ValueError(f"Unsupported bit-width: {bits}. Supported values: 4, 8")
        if bits == 4 and quantization_mode == 'dynamic':
            raise ValueError("4-bit quantization is not supported for dynamic mode. Use static or QAT mode.")
        
    def should_quantize(self, module_name: str, module: nn.Module) -> bool:
        """Check if a module should be quantized based on its type."""
        layer_type = self.layer_identifier.identify_layer_type(module_name, module)
        return self.quantize_config.get(layer_type, False)
    
    def get_qconfig_mapping(self) -> QConfigMapping:
        """Get quantization configuration mapping for static/QAT quantization.
        
        Note: Dynamic quantization doesn't use qconfig mapping as it only
        quantizes weights, not activations.
        """
        if self.quantization_mode == 'dynamic':
            # Dynamic quantization doesn't require qconfig mapping
            return None
        
        # Get qconfig based on bit-width and per-channel settings
        if self.bits == 8:
            # 8-bit quantization (standard PyTorch support)
            if self.per_channel:
                from torch.ao.quantization.qconfig import default_per_channel_qconfig
                qconfig = default_per_channel_qconfig
            else:
                from torch.ao.quantization.qconfig import default_qconfig
                qconfig = default_qconfig
        elif self.bits == 4:
            # 4-bit quantization (requires custom qconfig)
            # Note: PyTorch doesn't have built-in 4-bit support, so we'll use
            # a workaround with custom observers or fall back to 8-bit with scaling
            print("Warning: 4-bit quantization is experimental. Using custom qconfig.")
            try:
                # Try to create a 4-bit qconfig using custom observers
                from torch.ao.quantization.observer import MinMaxObserver
                from torch.ao.quantization.fake_quantize import FakeQuantize
                from torch.ao.quantization.qconfig import QConfig
                
                # Create 4-bit observer (16 levels: -8 to 7 for signed, 0 to 15 for unsigned)
                if self.dtype == torch.qint8:
                    # For signed 4-bit: use 8-bit observer but scale to 4-bit range
                    weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
                    activation_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
                else:
                    weight_observer = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
                    activation_observer = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
                
                # Note: This is a simplified 4-bit approach. For true 4-bit quantization,
                # you may need additional libraries like bitsandbytes or custom implementations
                qconfig = QConfig(activation=activation_observer, weight=weight_observer)
                print("Note: Using experimental 4-bit quantization. Results may vary.")
            except Exception as e:
                print(f"Warning: Could not create 4-bit qconfig: {e}")
                print("Falling back to 8-bit quantization.")
                from torch.ao.quantization.qconfig import default_qconfig
                qconfig = default_qconfig
                self.bits = 8  # Update to reflect actual quantization
        else:
            raise ValueError(f"Unsupported bit-width: {self.bits}")
        
        # Create custom qconfig mapping
        qconfig_mapping = QConfigMapping()
        
        # Set qconfig for different layer types
        for layer_type, should_quantize in self.quantize_config.items():
            if should_quantize:
                if layer_type == 'attention':
                    # Attention layers - quantize both weights and activations
                    qconfig_mapping.set_object_type(nn.MultiheadAttention, qconfig)
                    qconfig_mapping.set_object_type(nn.Linear, qconfig)
                elif layer_type == 'mlp':
                    # MLP layers - quantize both weights and activations
                    qconfig_mapping.set_object_type(nn.Linear, qconfig)
                elif layer_type == 'embedding':
                    # Embedding layers - quantize both weights and activations
                    qconfig_mapping.set_object_type(nn.Embedding, qconfig)
                    qconfig_mapping.set_object_type(nn.Conv2d, qconfig)
                elif layer_type == 'head':
                    # Classification head - quantize both weights and activations
                    qconfig_mapping.set_object_type(nn.Linear, qconfig)
        
        return qconfig_mapping
    
    def quantize_dynamic(
        self,
        model: nn.Module,
        layer_types_to_quantize: Optional[Set[str]] = None
    ) -> nn.Module:
        """Apply dynamic quantization to selected layers.
        
        Note: Dynamic quantization only quantizes weights, not activations.
        For activation quantization, use static or QAT mode.
        """
        if layer_types_to_quantize is None:
            layer_types_to_quantize = {
                k for k, v in self.quantize_config.items() if v
            }
        
        # Dynamic quantization for Linear layers (weights only)
        modules_to_quantize = {}
        for name, module in model.named_modules():
            layer_type = self.layer_identifier.identify_layer_type(name, module)
            if layer_type in layer_types_to_quantize:
                if isinstance(module, nn.Linear):
                    modules_to_quantize[name] = module
        
        # Apply dynamic quantization
        # Note: We need to quantize from bottom-up to avoid replacing parent modules
        sorted_names = sorted(modules_to_quantize.keys(), key=lambda x: x.count('.'), reverse=True)
        
        for name in sorted_names:
            module = modules_to_quantize[name]
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            try:
                if parent_name:
                    parent_module = model
                    for part in parent_name.split('.'):
                        parent_module = getattr(parent_module, part)
                    quantized = torch.quantization.quantize_dynamic(
                        module, {nn.Linear}, dtype=self.dtype
                    )
                    setattr(parent_module, child_name, quantized)
                else:
                    quantized = torch.quantization.quantize_dynamic(
                        module, {nn.Linear}, dtype=self.dtype
                    )
                    setattr(model, child_name, quantized)
            except Exception as e:
                print(f"Warning: Could not quantize {name}: {e}")
                continue
        
        return model
    
    def quantize_static(
        self,
        model: nn.Module,
        example_inputs: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """Apply static quantization to selected layers."""
        qconfig_mapping = self.get_qconfig_mapping()
        
        # Prepare model for quantization
        # Handle different PyTorch API versions
        import inspect
        prepare_sig = inspect.signature(prepare)
        
        # Check if prepare accepts qconfig_mapping as keyword argument
        if 'qconfig_mapping' in prepare_sig.parameters:
            # Newer API: use keyword argument
            if example_inputs is not None and 'example_inputs' in prepare_sig.parameters:
                model_prepared = prepare(model, qconfig_mapping=qconfig_mapping, example_inputs=example_inputs)
            else:
                model_prepared = prepare(model, qconfig_mapping=qconfig_mapping)
        else:
            # Older API: might need positional argument or different approach
            # Try positional first
            try:
                if example_inputs is not None:
                    model_prepared = prepare(model, qconfig_mapping, example_inputs=example_inputs)
                else:
                    model_prepared = prepare(model, qconfig_mapping)
            except TypeError:
                # If that fails, the API might be completely different
                # Fall back to using prepare_fx or suggest dynamic mode
                raise RuntimeError(
                    "Static quantization is not supported with your PyTorch version. "
                    "Please use --mode dynamic instead, or update PyTorch to version 2.0+."
                )
        
        # Calibrate with example inputs (if provided)
        if example_inputs is not None:
            with torch.no_grad():
                _ = model_prepared(example_inputs)
        
        # Convert to quantized model
        model_quantized = convert(model_prepared)
        
        return model_quantized
    
    def quantize_qat(
        self,
        model: nn.Module,
        example_inputs: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """Apply quantization-aware training setup."""
        qconfig_mapping = self.get_qconfig_mapping()
        
        # Prepare model for QAT
        # Handle different PyTorch API versions
        import inspect
        prepare_qat_sig = inspect.signature(prepare_qat)
        
        # Check if prepare_qat accepts qconfig_mapping as keyword argument
        if 'qconfig_mapping' in prepare_qat_sig.parameters:
            # Newer API: use keyword argument
            if example_inputs is not None and 'example_inputs' in prepare_qat_sig.parameters:
                model_prepared = prepare_qat(model, qconfig_mapping=qconfig_mapping, example_inputs=example_inputs)
            else:
                model_prepared = prepare_qat(model, qconfig_mapping=qconfig_mapping)
        else:
            # Older API: try positional argument
            try:
                if example_inputs is not None:
                    model_prepared = prepare_qat(model, qconfig_mapping, example_inputs=example_inputs)
                else:
                    model_prepared = prepare_qat(model, qconfig_mapping)
            except TypeError:
                raise RuntimeError(
                    "QAT quantization is not supported with your PyTorch version. "
                    "Please use --mode dynamic instead, or update PyTorch to version 2.0+."
                )
        
        return model_prepared
    
    def quantize(
        self,
        model: nn.Module,
        example_inputs: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """Apply quantization based on the selected mode."""
        if self.quantization_mode == 'dynamic':
            return self.quantize_dynamic(model)
        elif self.quantization_mode == 'static':
            return self.quantize_static(model, example_inputs)
        elif self.quantization_mode == 'qat':
            return self.quantize_qat(model, example_inputs)
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
        description="Quantize EVA-transformer model with optional layer-type quantization"
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
    parser.add_argument(
        "--mode",
        type=str,
        default="dynamic",
        choices=["static", "dynamic", "qat"],
        help="Quantization mode: static (quantizes weights+activations, requires calibration), "
             "dynamic (weights only, no calibration needed), or qat (quantization-aware training)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="qint8",
        choices=["qint8", "quint8"],
        help="Quantization dtype (qint8 for signed, quint8 for unsigned)"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Quantization bit-width (4 or 8 bits). "
             "Note: 4-bit is experimental and only works with static/QAT mode. "
             "8-bit is the standard and recommended option."
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Use per-channel quantization (only for static/QAT, more accurate but slower)"
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
    parser.add_argument(
        "--example-input-shape",
        type=int,
        nargs=4,
        default=[1, 3, 224, 224],
        metavar=("B", "C", "H", "W"),
        help="Example input shape for static/QAT quantization [B C H W]"
    )
    
    args = parser.parse_args()
    
    # Build quantization config
    # Default: quantize attention, MLP, embedding, and head layers (layer-by-layer quantization)
    # This enables quantization by default for the most important layer types
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
        # Use individual flags with defaults: attention=True, mlp=True, embedding=True, head=True, norm=False
        # Default behavior: enable quantization for attention, MLP, embedding, and head
        # If --no-quantize-* flag is used, it will set the value to False
        # If --quantize-* flag is used, it will set the value to True
        # If neither is used, we default to True for most layers
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
    
    print(f"\n=== Quantization Configuration ===")
    for layer_type, enabled in quantize_config.items():
        status = "ENABLED" if enabled else "DISABLED"
        print(f"  {layer_type}: {status}")
    print(f"Mode: {args.mode}")
    if args.mode == 'dynamic':
        print("  Note: Dynamic mode quantizes WEIGHTS ONLY (not activations)")
    elif args.mode in ['static', 'qat']:
        print("  Note: Static/QAT mode quantizes BOTH WEIGHTS AND ACTIVATIONS")
    print(f"Quantization level: {args.bits}-bit")
    print(f"Dtype: {args.dtype}")
    print(f"Per-channel: {args.per_channel}")
    
    # Validate bit-width compatibility
    if args.bits == 4 and args.mode == 'dynamic':
        print("\nERROR: 4-bit quantization is not supported for dynamic mode!")
        print("Please use --mode static or --mode qat for 4-bit quantization.")
        return
    
    # Convert dtype string to torch dtype
    dtype_map = {
        'qint8': torch.qint8,
        'quint8': torch.quint8
    }
    torch_dtype = dtype_map[args.dtype]
    
    # Create quantizer
    quantizer = SelectiveQuantizer(
        quantize_config=quantize_config,
        quantization_mode=args.mode,
        dtype=torch_dtype,
        per_channel=args.per_channel,
        bits=args.bits
    )
    
    # Prepare example inputs if needed
    example_inputs = None
    if args.mode in ['static', 'qat']:
        B, C, H, W = args.example_input_shape
        example_inputs = torch.randn(B, C, H, W)
        print(f"\nUsing example input shape: {args.example_input_shape}")
    
    # Apply quantization
    print(f"\nApplying {args.mode} quantization...")
    try:
        quantized_model = quantizer.quantize(model, example_inputs)
        print("Quantization completed successfully!")
    except Exception as e:
        print(f"Error during quantization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Replace the underlying model in the wrapper with the quantized version
    if hasattr(model_wrapper, 'model'):
        model_wrapper.model = quantized_model
    elif hasattr(model_wrapper, 'timm_model'):
        model_wrapper.timm_model = quantized_model
    else:
        # If we couldn't find the underlying model, use the quantized model directly
        model_wrapper = quantized_model
    
    # Save quantized model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving quantized model to: {output_path}")
    try:
        if args.mode == 'qat':
            # For QAT, save the prepared model (needs training before conversion)
            torch.save(quantized_model.state_dict(), output_path)
            print("Note: QAT model saved. Train it before converting to quantized model.")
            # Also save the wrapper with the prepared model
            wrapper_path = output_path.with_suffix('.wrapper.pth')
            torch.save(model_wrapper.state_dict(), wrapper_path)
            print(f"Wrapper with QAT model saved to: {wrapper_path}")
        else:
            # For static/dynamic, save the quantized model
            torch.save(quantized_model.state_dict(), output_path)
            # Also save the full wrapper model
            wrapper_path = output_path.with_suffix('.wrapper.pth')
            torch.save(model_wrapper.state_dict(), wrapper_path)
            print(f"Quantized model state dict saved to: {output_path}")
            print(f"Full wrapper model state dict saved to: {wrapper_path}")
            print("Note: To use the quantized model, load the wrapper state dict and use it with TimmWrapperForImageClassification")
    except Exception as e:
        print(f"Error saving model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nQuantization script completed!")


if __name__ == "__main__":
    main()

