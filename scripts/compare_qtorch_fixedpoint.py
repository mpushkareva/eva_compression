#!/usr/bin/env python3
"""
Compare layer outputs between qtorch-quantized model and fixed-point math model.

This script loads a model quantized with qtorch and compares its layer outputs
with the same model using the same weights but with manual fixed-point math operations.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, TimmWrapperForImageClassification
import json
import numpy as np

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
        raise ImportError(
            "setuptools is required for QPyTorch. "
            "Install it with: pip install setuptools"
        ) from e
    else:
        raise


class LayerTypeIdentifier:
    """Identifies different layer types in EVA-transformer models."""
    
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
        
        if any(pattern in name_lower for pattern in cls.ATTENTION_PATTERNS):
            return 'attention'
        if any(pattern in name_lower for pattern in cls.MLP_PATTERNS):
            return 'mlp'
        if any(pattern in name_lower for pattern in cls.EMBEDDING_PATTERNS):
            return 'embedding'
        if any(pattern in name_lower for pattern in cls.NORM_PATTERNS):
            return 'norm'
        if any(pattern in name_lower for pattern in cls.HEAD_PATTERTERNS):
            return 'head'
        
        if isinstance(module, (nn.MultiheadAttention,)):
            return 'attention'
        elif isinstance(module, (nn.Linear,)) and 'attn' not in name_lower:
            return 'mlp'
        elif isinstance(module, (nn.Embedding, nn.Conv2d)):
            return 'embedding'
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            return 'norm'
        
        return 'other'


class FixedPointMath:
    """Manual fixed-point math operations."""
    
    def __init__(self, wl: int = 8, fl: int = 2):
        """
        Initialize fixed-point math.
        
        Args:
            wl: Word length (total bits)
            fl: Fractional length (fractional bits)
        """
        self.wl = wl
        self.fl = fl
        self.int_bits = wl - fl - 1  # Signed: 1 bit for sign
        self.scale = 2.0 ** fl
        
        # Compute quantization range
        self.max_val = (2 ** (wl - 1) - 1) / self.scale
        self.min_val = -(2 ** (wl - 1)) / self.scale
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize tensor to fixed-point representation.
        
        Args:
            x: Input tensor
            
        Returns:
            Quantized tensor (still in float, but represents fixed-point values)
        """
        # Scale to fixed-point range
        x_scaled = x * self.scale
        
        # Round to nearest integer
        x_rounded = torch.round(x_scaled)
        
        # Clamp to valid range
        x_clamped = torch.clamp(x_rounded, 
                               min=-(2 ** (self.wl - 1)),
                               max=(2 ** (self.wl - 1) - 1))
        
        # Convert back to float representation
        x_quantized = x_clamped / self.scale
        
        return x_quantized
    
    def apply_fixed_point_ops(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fixed-point quantization to tensor."""
        return self.quantize(x)


class FixedPointWrapper(nn.Module):
    """Wrapper that applies fixed-point math to module output."""
    
    def __init__(self, module: nn.Module, fixed_point: FixedPointMath):
        super().__init__()
        self.module = module
        self.fixed_point = fixed_point
    
    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        return self.fixed_point.apply_fixed_point_ops(out)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def load_eva_model(
    model_name_or_path: str,
    num_labels: int = 1000,
    token: Optional[str] = None,
    return_wrapper: bool = False
) -> nn.Module:
    """Load EVA transformer model."""
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


def create_fixed_point_model(
    model: nn.Module,
    quantize_config: Dict[str, bool],
    forward_wl: int = 8,
    forward_fl: int = 2
) -> nn.Module:
    """
    Create a model with fixed-point math operations.
    
    Args:
        model: Base model
        quantize_config: Dictionary mapping layer types to boolean flags
        forward_wl: Forward word length
        forward_fl: Forward fractional length
        
    Returns:
        Model with fixed-point wrappers applied
    """
    layer_identifier = LayerTypeIdentifier()
    fixed_point = FixedPointMath(wl=forward_wl, fl=forward_fl)
    
    # Build module info
    module_info = {}
    for name, module in model.named_modules():
        layer_type = layer_identifier.identify_layer_type(name, module)
        if quantize_config.get(layer_type, False):
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
                name_parts = name.split('.')
                parent_name = '.'.join(name_parts[:-1]) if len(name_parts) > 1 else ''
                attr_name = name_parts[-1]
                
                module_info[name] = {
                    'module': module,
                    'parent_name': parent_name,
                    'attr_name': attr_name
                }
    
    # Wrap modules
    for name, info in module_info.items():
        print(f"Wrapping {name} with FixedPointMath")
        
        if info['parent_name']:
            parent = model
            for part in info['parent_name'].split('.'):
                parent = getattr(parent, part)
        else:
            parent = model
        
        wrapped = FixedPointWrapper(info['module'], fixed_point)
        setattr(parent, info['attr_name'], wrapped)
    
    return model


def load_qtorch_model(
    model_path: str,
    metadata_path: str,
    base_model_name: Optional[str] = None,
    num_labels: int = 1000,
    token: Optional[str] = None
) -> nn.Module:
    """
    Load qtorch-quantized model.
    
    Note: The qtorch model should have been saved with Quantizer wrappers already applied.
    We need to recreate the quantization structure before loading weights.
    
    Args:
        model_path: Path to quantized model state dict
        metadata_path: Path to metadata file
        base_model_name: Base model name for loading structure (optional, will use metadata if available)
        num_labels: Number of labels
        token: HuggingFace token
        
    Returns:
        Quantized model with qtorch wrappers
    """
    # Load metadata
    metadata = torch.load(metadata_path, map_location='cpu')
    quantize_config = metadata.get('quantize_config', {})
    
    # Use model name from metadata if available, otherwise use provided base_model_name
    if base_model_name is None:
        base_model_name = metadata.get('model_name')
        if base_model_name is None:
            raise ValueError(
                "base_model_name must be provided either as argument or in metadata file. "
                "Metadata file does not contain 'model_name'."
            )
        print(f"Using model name from metadata: {base_model_name}")
    else:
        # Check if metadata has a different model name and warn
        metadata_model_name = metadata.get('model_name')
        if metadata_model_name and metadata_model_name != base_model_name:
            print(f"WARNING: Model name mismatch!")
            print(f"  Provided: {base_model_name}")
            print(f"  In metadata: {metadata_model_name}")
            print(f"  Using model name from metadata: {metadata_model_name}")
            base_model_name = metadata_model_name
    
    # Recreate quantization parameters
    forward_format = metadata.get('forward_format', 'fixed')
    if forward_format == 'fixed':
        forward_num = FixedPoint(
            wl=metadata.get('forward_wl', 8),
            fl=metadata.get('forward_fl', 2)
        )
    else:
        forward_num = FloatingPoint(
            exp=metadata.get('forward_exp', 5),
            man=metadata.get('forward_man', 2)
        )
    
    backward_num = FloatingPoint(
        exp=metadata.get('backward_exp', 5),
        man=metadata.get('backward_man', 2)
    )
    
    forward_rounding = metadata.get('forward_rounding', 'nearest')
    backward_rounding = metadata.get('backward_rounding', 'nearest')
    
    # Load base model
    model_wrapper = load_eva_model(base_model_name, num_labels, token, return_wrapper=True)
    model_wrapper.eval()
    
    if hasattr(model_wrapper, 'model'):
        model = model_wrapper.model
    elif hasattr(model_wrapper, 'timm_model'):
        model = model_wrapper.timm_model
    else:
        model = model_wrapper
    
    model.eval()
    
    # Recreate qtorch quantization structure
    # Import QPyTorchQuantizer from quantize_fp
    try:
        import quantize_fp
        QPyTorchQuantizer = quantize_fp.QPyTorchQuantizer
    except ImportError:
        # Fallback: try importing from current directory
        import sys
        from pathlib import Path
        quantize_fp_path = Path(__file__).parent / 'quantize_fp.py'
        if quantize_fp_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("quantize_fp", quantize_fp_path)
            quantize_fp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(quantize_fp_module)
            QPyTorchQuantizer = quantize_fp_module.QPyTorchQuantizer
        else:
            raise FileNotFoundError(f"Could not find quantize_fp.py at {quantize_fp_path}")
    
    quantizer = QPyTorchQuantizer(
        quantize_config=quantize_config,
        forward_num=forward_num,
        backward_num=backward_num,
        forward_rounding=forward_rounding,
        backward_rounding=backward_rounding
    )
    
    # Apply quantization wrappers
    quantized_model = quantizer.quantize_model(model)
    
    # Load quantized state dict
    state_dict = torch.load(model_path, map_location='cpu')
    quantized_model.load_state_dict(state_dict, strict=False)
    
    return quantized_model


class LayerOutputHook:
    """Hook to capture layer outputs."""
    
    def __init__(self):
        self.outputs = {}
        self.hooks = []
    
    def register_hook(self, name: str, module: nn.Module):
        """Register a forward hook for a module."""
        def hook_fn(module, input, output):
            self.outputs[name] = output.detach().clone()
        
        hook = module.register_forward_hook(hook_fn)
        self.hooks.append(hook)
    
    def clear(self):
        """Clear all outputs and remove hooks."""
        self.outputs.clear()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def compare_outputs(
    output1: torch.Tensor,
    output2: torch.Tensor,
    name: str
) -> Dict[str, float]:
    """
    Compare two tensor outputs.
    
    Args:
        output1: First output tensor
        output2: Second output tensor
        name: Layer name for reporting
        
    Returns:
        Dictionary with comparison metrics
    """
    # Ensure same shape
    if output1.shape != output2.shape:
        return {
            'name': name,
            'error': f'Shape mismatch: {output1.shape} vs {output2.shape}'
        }
    
    # Flatten for comparison
    flat1 = output1.flatten().cpu()
    flat2 = output2.flatten().cpu()
    
    # Compute metrics
    mse = torch.mean((flat1 - flat2) ** 2).item()
    mae = torch.mean(torch.abs(flat1 - flat2)).item()
    max_diff = torch.max(torch.abs(flat1 - flat2)).item()
    
    # Relative error
    abs1 = torch.abs(flat1)
    relative_error = torch.where(
        abs1 > 1e-8,
        torch.abs(flat1 - flat2) / (abs1 + 1e-8),
        torch.abs(flat1 - flat2)
    )
    mean_relative_error = torch.mean(relative_error).item()
    max_relative_error = torch.max(relative_error).item()
    
    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        flat1.unsqueeze(0), flat2.unsqueeze(0)
    ).item()
    
    # Correlation
    if len(flat1) > 1:
        correlation = torch.corrcoef(torch.stack([flat1, flat2]))[0, 1].item()
    else:
        correlation = 1.0 if torch.allclose(flat1, flat2) else 0.0
    
    return {
        'name': name,
        'shape': list(output1.shape),
        'mse': mse,
        'mae': mae,
        'max_diff': max_diff,
        'mean_relative_error': mean_relative_error,
        'max_relative_error': max_relative_error,
        'cosine_similarity': cos_sim,
        'correlation': correlation,
        'mean_output1': torch.mean(flat1).item(),
        'std_output1': torch.std(flat1).item(),
        'mean_output2': torch.mean(flat2).item(),
        'std_output2': torch.std(flat2).item(),
    }


def get_model_input_size(model: nn.Module) -> int:
    """
    Get the expected input image size from the model.
    
    Args:
        model: The model to inspect (can be wrapped or unwrapped)
        
    Returns:
        Expected input size (height/width, assuming square images)
    """
    # Handle wrapped models (TimmWrapperForImageClassification)
    actual_model = model
    if hasattr(model, 'model'):
        actual_model = model.model
    elif hasattr(model, 'timm_model'):
        actual_model = model.timm_model
    
    # Try to get from patch_embed (common in vision transformers)
    if hasattr(actual_model, 'patch_embed') and hasattr(actual_model.patch_embed, 'img_size'):
        img_size = actual_model.patch_embed.img_size
        if isinstance(img_size, (tuple, list)):
            return img_size[0] if len(img_size) > 0 else 224
        return int(img_size)
    
    # Try to get from model attributes
    if hasattr(actual_model, 'img_size'):
        img_size = actual_model.img_size
        if isinstance(img_size, (tuple, list)):
            return img_size[0] if len(img_size) > 0 else 224
        return int(img_size)
    
    # Try to get from default_cfg if available (timm models)
    if hasattr(actual_model, 'default_cfg') and actual_model.default_cfg is not None:
        if 'input_size' in actual_model.default_cfg:
            input_size = actual_model.default_cfg['input_size']
            if isinstance(input_size, (tuple, list)):
                if len(input_size) >= 3:
                    # Usually (3, H, W) format
                    return int(input_size[-1])
                elif len(input_size) >= 2:
                    # Usually (H, W) format
                    return int(input_size[-1])
            return int(input_size)
    
    # Default fallback
    return 224


def compare_models(
    qtorch_model: nn.Module,
    fixed_point_model: nn.Module,
    input_tensor: torch.Tensor,
    layer_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare outputs of two models layer by layer.
    
    Args:
        qtorch_model: Model quantized with qtorch
        fixed_point_model: Model with fixed-point math
        input_tensor: Input tensor
        layer_names: Optional list of layer names to compare (if None, compares all)
        
    Returns:
        Dictionary mapping layer names to comparison metrics
    """
    # Register hooks for both models
    hook1 = LayerOutputHook()
    hook2 = LayerOutputHook()
    
    # Get all quantizable layers from both models
    layer_identifier = LayerTypeIdentifier()
    all_layers = set()
    
    # Collect layer names from qtorch model (may have wrappers)
    for name, module in qtorch_model.named_modules():
        # Check if it's a wrapped module (qtorch QuantizedWrapper)
        if hasattr(module, 'module') and hasattr(module, 'quantizer'):
            # This is a qtorch wrapper - use the wrapper itself to capture quantized output
            all_layers.add(name)
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
            # Regular module
            all_layers.add(name)
    
    # Also collect from fixed-point model
    for name, module in fixed_point_model.named_modules():
        # Check if it's a wrapped module (FixedPointWrapper)
        if hasattr(module, 'module') and hasattr(module, 'fixed_point'):
            # This is a fixed-point wrapper - use the wrapper itself
            all_layers.add(name)
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
            # Regular module
            all_layers.add(name)
    
    # Register hooks on matching layers
    for name in all_layers:
        if layer_names is not None and name not in layer_names:
            continue
        
        # Find corresponding module in qtorch model
        try:
            module1 = qtorch_model
            for part in name.split('.'):
                module1 = getattr(module1, part)
            hook1.register_hook(name, module1)
        except AttributeError:
            continue
        
        # Find corresponding module in fixed-point model
        try:
            module2 = fixed_point_model
            for part in name.split('.'):
                module2 = getattr(module2, part)
            hook2.register_hook(name, module2)
        except AttributeError:
            continue
    
    # Run forward pass on both models
    with torch.no_grad():
        _ = qtorch_model(input_tensor)
        _ = fixed_point_model(input_tensor)
    
    # Compare outputs
    results = {}
    for name in hook1.outputs.keys():
        if name in hook2.outputs:
            output1 = hook1.outputs[name]
            output2 = hook2.outputs[name]
            results[name] = compare_outputs(output1, output2, name)
    
    # Cleanup
    hook1.clear()
    hook2.clear()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare qtorch-quantized model with fixed-point math model"
    )
    parser.add_argument(
        "--qtorch-model",
        type=str,
        required=True,
        help="Path to qtorch-quantized model state dict (.pth file)"
    )
    parser.add_argument(
        "--qtorch-metadata",
        type=str,
        required=True,
        help="Path to qtorch model metadata (.metadata.pth file)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=False,
        default=None,
        help="Base model name or path (for loading model structure). "
             "If not provided, will use model name from metadata file."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for comparison results (JSON file)"
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=1000,
        help="Number of classification labels (default: 1000)"
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=4,
        default=[1, 3, 224, 224],
        metavar=("B", "C", "H", "W"),
        help="Input shape for comparison [B C H W]"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run comparison on"
    )
    
    args = parser.parse_args()
    
    # Load metadata
    print(f"Loading metadata from: {args.qtorch_metadata}")
    metadata = torch.load(args.qtorch_metadata, map_location='cpu')
    
    quantize_config = metadata.get('quantize_config', {})
    forward_wl = metadata.get('forward_wl', 8)
    forward_fl = metadata.get('forward_fl', 2)
    
    # Determine which model to use (from args or metadata)
    base_model_name = args.base_model
    if base_model_name is None:
        base_model_name = metadata.get('model_name')
        if base_model_name is None:
            raise ValueError(
                "base_model must be provided either via --base-model argument or in metadata file. "
                "Metadata file does not contain 'model_name'."
            )
        print(f"Using model name from metadata: {base_model_name}")
    else:
        # Check if metadata has a different model name and warn
        metadata_model_name = metadata.get('model_name')
        if metadata_model_name and metadata_model_name != base_model_name:
            print(f"WARNING: Model name mismatch!")
            print(f"  Provided: {base_model_name}")
            print(f"  In metadata: {metadata_model_name}")
            print(f"  Using model name from metadata: {metadata_model_name}")
            base_model_name = metadata_model_name
    
    print(f"Quantization config: {quantize_config}")
    print(f"Forward format: FixedPoint(wl={forward_wl}, fl={forward_fl})")
    
    # Load qtorch model
    print(f"\nLoading qtorch-quantized model from: {args.qtorch_model}")
    qtorch_model = load_qtorch_model(
        args.qtorch_model,
        args.qtorch_metadata,
        base_model_name,
        args.num_labels,
        args.token
    )
    qtorch_model.eval()
    qtorch_model = qtorch_model.to(args.device)
    
    # Load base model for fixed-point version
    print(f"\nLoading base model for fixed-point version: {base_model_name}")
    model_wrapper = load_eva_model(base_model_name, args.num_labels, args.token, return_wrapper=True)
    model_wrapper.eval()
    
    if hasattr(model_wrapper, 'model'):
        base_model = model_wrapper.model
    elif hasattr(model_wrapper, 'timm_model'):
        base_model = model_wrapper.timm_model
    else:
        base_model = model_wrapper
    
    base_model.eval()
    
    # Copy weights from qtorch model to base model
    print("\nCopying weights from qtorch model to base model...")
    qtorch_state = qtorch_model.state_dict()
    base_state = base_model.state_dict()
    
    # Extract actual weights from wrapped modules
    # QPyTorch wrappers store weights in module.weight, module.bias, etc.
    copied_count = 0
    for key in qtorch_state.keys():
        if key in base_state:
            # Check if shapes match
            if qtorch_state[key].shape == base_state[key].shape:
                base_state[key] = qtorch_state[key].clone()
                copied_count += 1
        else:
            # Try to find corresponding key (might be nested in wrapper)
            # QPyTorch wrappers might have keys like "module.weight" instead of just "weight"
            if key.endswith('.weight') or key.endswith('.bias'):
                # Try without the wrapper prefix
                base_key = key
                if '.module.' in key:
                    # Remove wrapper prefix
                    base_key = key.replace('.module.', '.')
                if base_key in base_state and qtorch_state[key].shape == base_state[base_key].shape:
                    base_state[base_key] = qtorch_state[key].clone()
                    copied_count += 1
    
    base_model.load_state_dict(base_state, strict=False)
    print(f"Copied {copied_count} weight parameters")
    
    # Create fixed-point model
    print("\nCreating fixed-point model...")
    fixed_point_model = create_fixed_point_model(
        base_model,
        quantize_config,
        forward_wl,
        forward_fl
    )
    fixed_point_model.eval()
    fixed_point_model = fixed_point_model.to(args.device)
    
    # Detect model input size and update input shape if needed
    detected_size = get_model_input_size(base_model)
    B, C, H, W = args.input_shape
    
    # If user provided custom input shape, use it; otherwise use detected size
    if H == 224 and W == 224:  # Default values
        H = W = detected_size
        print(f"\nDetected model input size: {detected_size}x{detected_size}")
        print(f"Updating input shape from default 224x224 to {detected_size}x{detected_size}")
    else:
        print(f"\nUsing provided input shape: {H}x{W}")
        if H != detected_size or W != detected_size:
            print(f"WARNING: Provided input size ({H}x{W}) doesn't match model's expected size ({detected_size}x{detected_size})")
    
    # Create input tensor
    input_tensor = torch.randn(B, C, H, W, device=args.device)
    print(f"Final input shape: [{B}, {C}, {H}, {W}]")
    
    # Compare models
    print("\nComparing layer outputs...")
    results = compare_models(qtorch_model, fixed_point_model, input_tensor)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert tensors to lists for JSON serialization
    json_results = {}
    for name, metrics in results.items():
        json_results[name] = {k: (float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v) 
                             for k, v in metrics.items()}
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n=== Comparison Summary ===")
    print(f"Total layers compared: {len(results)}")
    
    if results:
        mse_values = [r['mse'] for r in results.values() if 'mse' in r]
        mae_values = [r['mae'] for r in results.values() if 'mae' in r]
        cos_sim_values = [r['cosine_similarity'] for r in results.values() if 'cosine_similarity' in r]
        
        if mse_values:
            print(f"Mean MSE: {np.mean(mse_values):.6e}")
            print(f"Max MSE: {np.max(mse_values):.6e}")
        if mae_values:
            print(f"Mean MAE: {np.mean(mae_values):.6e}")
            print(f"Max MAE: {np.max(mae_values):.6e}")
        if cos_sim_values:
            print(f"Mean Cosine Similarity: {np.mean(cos_sim_values):.6f}")
            print(f"Min Cosine Similarity: {np.min(cos_sim_values):.6f}")
        
        # Show top 5 layers with highest differences
        print("\nTop 5 layers with highest MSE:")
        sorted_by_mse = sorted(results.items(), key=lambda x: x[1].get('mse', 0), reverse=True)[:5]
        for name, metrics in sorted_by_mse:
            print(f"  {name}: MSE={metrics.get('mse', 0):.6e}, MAE={metrics.get('mae', 0):.6e}")
    
    print("\nComparison completed!")


if __name__ == "__main__":
    main()

