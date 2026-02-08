"""
Integer-only EVA-02 Tiny using I-ViT quantization primitives.

Replaces all floating-point nonlinear ops (LayerNorm, Softmax, SiLU)
with integer-arithmetic equivalents from I-ViT (ICCV 2023).

Key adaptations from standard I-ViT (designed for vanilla ViT):
  - IntSiLU instead of IntGELU (SwiGLU activation)
  - IntGluMlp with gated structure: SiLU(x1) * x2
  - RoPE on Q,K (applied in float, norm-preserving)
  - No QKV bias (EVA-02 uses q_bias/v_bias trick)
  - Decomposed attention (no fused SDPA)
"""

import sys
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from functools import partial

# ── I-ViT imports ──────────────────────────────────────────────────────────
# I-ViT uses `from .quant_utils import *` (relative imports), so we need its
# package hierarchy in sys.modules.  To avoid clashing with *our* `models/`
# package we temporarily swap sys.path, import what we need under the
# `ivit_models` alias, then restore.

import importlib
import types

_ivit_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'I-ViT'))

# Save any existing 'models' entry (ours) and temporarily remove it
_saved_models = sys.modules.pop('models', None)
_saved_models_qu = sys.modules.pop('models.quantization_utils', None)
_saved_models_qm = sys.modules.pop('models.quantization_utils.quant_modules', None)
_saved_models_qutil = sys.modules.pop('models.quantization_utils.quant_utils', None)

sys.path.insert(0, _ivit_root)
try:
    import models.quantization_utils.quant_modules as _qm
    import models.quantization_utils.quant_utils as _qu
finally:
    sys.path.remove(_ivit_root)
    # Remove I-ViT's 'models' from sys.modules so ours can be found again
    sys.modules.pop('models', None)
    sys.modules.pop('models.quantization_utils', None)
    sys.modules.pop('models.quantization_utils.quant_modules', None)
    sys.modules.pop('models.quantization_utils.quant_utils', None)
    # Restore our modules
    if _saved_models is not None:
        sys.modules['models'] = _saved_models
    if _saved_models_qu is not None:
        sys.modules['models.quantization_utils'] = _saved_models_qu
    if _saved_models_qm is not None:
        sys.modules['models.quantization_utils.quant_modules'] = _saved_models_qm
    if _saved_models_qutil is not None:
        sys.modules['models.quantization_utils.quant_utils'] = _saved_models_qutil

QuantLinear = _qm.QuantLinear
QuantAct = _qm.QuantAct
QuantConv2d = _qm.QuantConv2d
QuantMatMul = _qm.QuantMatMul
IntLayerNorm = _qm.IntLayerNorm
IntSoftmax = _qm.IntSoftmax
floor_ste = _qu.floor_ste

# ── timm imports (RoPE) ───────────────────────────────────────────────────
from timm.layers.pos_embed_sincos import apply_rot_embed_cat


# ═══════════════════════════════════════════════════════════════════════════
# IntSiLU — integer-only SiLU via shift-based sigmoid
# ═══════════════════════════════════════════════════════════════════════════

class IntSiLU(nn.Module):
    """
    SiLU(x) = x * sigmoid(x).

    Identical to I-ViT's IntGELU (which approximates GELU ≈ x·σ(1.702x))
    except we drop the 1.702 constant: σ(x) instead of σ(1.702x).
    """

    def __init__(self, output_bit=8):
        super().__init__()
        self.output_bit = output_bit
        self.n = 23  # integer precision for shift-based exp
        self.register_buffer('act_scaling_factor', torch.zeros(1))

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2 ** 4)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r / 2 - x0_int
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2 ** self.n
        return exp_int, scaling_factor

    def forward(self, x, scaling_factor=None):
        pre_x_int = x / scaling_factor
        # SiLU: sigmoid(x), not sigmoid(1.702*x) as in GELU
        scaling_factor_sig = scaling_factor * 1.0  # no constant

        x_int_max, _ = pre_x_int.max(dim=-1, keepdim=True)
        x_int = pre_x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor_sig)
        exp_int_max, _ = self.int_exp_shift(-x_int_max, scaling_factor_sig)
        exp_int_sum = exp_int + exp_int_max

        exp_int_sum.clamp_max_(2 ** 31 - 1)
        factor = floor_ste.apply((2 ** 31 - 1) / exp_int_sum)
        sigmoid_int = floor_ste.apply(
            exp_int * factor / 2 ** (31 - self.output_bit + 1)
        )
        sigmoid_scaling_factor = torch.Tensor(
            [1 / 2 ** (self.output_bit - 1)]
        ).to(x.device)

        x_int = pre_x_int * sigmoid_int
        scaling_factor = scaling_factor * sigmoid_scaling_factor
        self.act_scaling_factor = scaling_factor
        return x_int * scaling_factor, scaling_factor


# ═══════════════════════════════════════════════════════════════════════════
# IntGluMlp — integer-only SwiGLU MLP
# ═══════════════════════════════════════════════════════════════════════════

class IntGluMlp(nn.Module):
    """
    Integer-only GluMlp (SwiGLU).

    EVA-02 Tiny structure (gate_last=False):
        fc1(192→1024) → chunk → SiLU(x1) * x2 → fc2(512→192)
    """

    def __init__(self, in_features, hidden_features, out_features=None,
                 weight_bit=8, bias_bit=32):
        super().__init__()
        out_features = out_features or in_features
        assert hidden_features % 2 == 0
        half_hidden = hidden_features // 2

        self.fc1 = QuantLinear(in_features, hidden_features,
                               bias=True, weight_bit=weight_bit,
                               bias_bit=bias_bit)
        self.qact_fc1 = QuantAct()

        self.act = IntSiLU()
        self.qact_act = QuantAct()   # quantize SiLU(x1)
        self.qact_gate = QuantAct()  # quantize x2 (gate)
        self.qact_mul = QuantAct()   # quantize after element-wise multiply

        self.fc2 = QuantLinear(half_hidden, out_features,
                               bias=True, weight_bit=weight_bit,
                               bias_bit=bias_bit)
        self.qact_fc2 = QuantAct(16)  # 16-bit for residual addition

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, scaling_factor):
        # fc1
        x, scaling_factor = self.fc1(x, scaling_factor)
        x, scaling_factor = self.qact_fc1(x, scaling_factor)

        # chunk into two halves
        x1, x2 = x.chunk(2, dim=-1)
        sf1 = scaling_factor
        sf2 = scaling_factor

        # gate_last=False: SiLU(x1) * x2
        x1, sf1 = self.act(x1, sf1)
        x1, sf1 = self.qact_act(x1, sf1)

        x2, sf2 = self.qact_gate(x2, sf2)

        # element-wise multiply (same pattern as QuantMatMul but element-wise)
        x1_int = x1 / sf1
        x2_int = x2 / sf2
        sf_mul = sf1 * sf2
        x = (x1_int * x2_int) * sf_mul

        x, scaling_factor = self.qact_mul(x, sf_mul)

        # fc2
        x, scaling_factor = self.fc2(x, scaling_factor)
        x, scaling_factor = self.qact_fc2(x, scaling_factor)
        return x, scaling_factor


# ═══════════════════════════════════════════════════════════════════════════
# IntEvaAttention — decomposed SDPA + RoPE
# ═══════════════════════════════════════════════════════════════════════════

class IntEvaAttention(nn.Module):
    """
    Integer-only multi-head attention for EVA-02.

    Differences from I-ViT's IntAttention:
      - No QKV bias (EVA-02 uses separate q_bias/v_bias)
      - RoPE applied to patch tokens only (not cls)
      - Decomposed attention (Q@K^T → softmax → @V)
    """

    def __init__(self, dim, num_heads=3, num_prefix_tokens=1,
                 weight_bit=8, bias_bit=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_prefix_tokens = num_prefix_tokens

        # QKV: EVA-02 has bias=False on the fused Linear, uses separate q/v bias
        self.qkv = QuantLinear(dim, 3 * dim, bias=False,
                               weight_bit=weight_bit, bias_bit=bias_bit)
        self.qact_qkv = QuantAct()

        # q_bias and v_bias as parameters, k_bias as zero buffer
        self.q_bias = Parameter(torch.zeros(dim))
        self.register_buffer('k_bias', torch.zeros(dim))
        self.v_bias = Parameter(torch.zeros(dim))

        # Attention matmuls
        self.matmul_qk = QuantMatMul()
        self.qact_attn = QuantAct()
        self.int_softmax = IntSoftmax(output_bit=16)
        self.matmul_av = QuantMatMul()
        self.qact_v = QuantAct()

        # Output projection
        self.proj = QuantLinear(dim, dim, bias=True,
                                weight_bit=weight_bit, bias_bit=bias_bit)
        self.qact_proj = QuantAct(16)  # 16-bit for residual

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, scaling_factor, rope=None):
        B, N, C = x.shape

        # ── QKV projection ─────────────────────────────────────────────
        # qkv Linear has bias=False; EVA-02 uses separate q_bias/v_bias
        qkv, sf_qkv = self.qkv(x, scaling_factor)
        # Add bias in float domain, then re-quantize
        qkv_bias = torch.cat([self.q_bias, self.k_bias, self.v_bias])
        qkv = qkv + qkv_bias
        qkv, sf_qkv = self.qact_qkv(qkv, sf_qkv)

        # ── Reshape to (B, num_heads, N, head_dim) ─────────────────────
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv.unbind(0)  # each (B, H, N, D)

        # ── RoPE (in float, norm-preserving) ───────────────────────────
        if rope is not None:
            npt = self.num_prefix_tokens
            # Only rotate patch tokens, leave cls/prefix tokens untouched
            q_t = apply_rot_embed_cat(q[:, :, npt:, :], rope)
            q = torch.cat([q[:, :, :npt, :], q_t], dim=2).type_as(v)

            k_t = apply_rot_embed_cat(k[:, :, npt:, :], rope)
            k = torch.cat([k[:, :, :npt, :], k_t], dim=2).type_as(v)

        # ── Q @ K^T ──────────────────────────────────────────────────
        attn, sf_attn = self.matmul_qk(
            q, sf_qkv, k.transpose(-2, -1), sf_qkv
        )
        # Apply 1/sqrt(d) scaling to attention logits (NOT to Q before matmul)
        attn = attn * self.scale
        sf_attn = sf_attn * self.scale
        attn, sf_attn = self.qact_attn(attn, sf_attn)

        # ── Softmax ──────────────────────────────────────────────────
        attn, sf_attn = self.int_softmax(attn, sf_attn)

        # ── attn @ V ─────────────────────────────────────────────────
        v, sf_v = self.qact_v(v, sf_qkv)
        x, sf_x = self.matmul_av(attn, sf_attn, v, sf_v)

        # ── Reshape and project ──────────────────────────────────────
        x = x.transpose(1, 2).reshape(B, N, C)
        sf_x = sf_x

        x, scaling_factor = self.proj(x, sf_x)
        x, scaling_factor = self.qact_proj(x, scaling_factor)
        return x, scaling_factor


# ═══════════════════════════════════════════════════════════════════════════
# IntEvaBlock — single transformer block
# ═══════════════════════════════════════════════════════════════════════════

class IntEvaBlock(nn.Module):
    """
    Integer-only EVA-02 transformer block.

    Pre-norm architecture:
      x = x + attn(norm1(x))
      x = x + mlp(norm2(x))
    Residual additions use QuantAct identity path.
    """

    def __init__(self, dim, num_heads, mlp_ratio, num_prefix_tokens=1,
                 weight_bit=8, bias_bit=32):
        super().__init__()
        hidden_features = int(dim * mlp_ratio) * 2  # doubled for GLU

        self.norm1 = IntLayerNorm(dim)
        self.qact1 = QuantAct()
        self.attn = IntEvaAttention(dim, num_heads,
                                    num_prefix_tokens=num_prefix_tokens,
                                    weight_bit=weight_bit, bias_bit=bias_bit)
        self.qact2 = QuantAct(16)  # residual add

        self.norm2 = IntLayerNorm(dim)
        self.qact3 = QuantAct()
        self.mlp = IntGluMlp(dim, hidden_features, dim,
                             weight_bit=weight_bit, bias_bit=bias_bit)
        self.qact4 = QuantAct(16)  # residual add

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, scaling_factor, rope=None):
        # ── Attention branch with residual ─────────────────────────────
        residual = x
        sf_residual = scaling_factor

        x, scaling_factor = self.norm1(x, scaling_factor)
        x, scaling_factor = self.qact1(x, scaling_factor)
        x, scaling_factor = self.attn(x, scaling_factor, rope=rope)
        # Residual add via QuantAct identity
        x, scaling_factor = self.qact2(
            x, scaling_factor,
            identity=residual, identity_scaling_factor=sf_residual
        )

        # ── MLP branch with residual ──────────────────────────────────
        residual = x
        sf_residual = scaling_factor

        x, scaling_factor = self.norm2(x, scaling_factor)
        x, scaling_factor = self.qact3(x, scaling_factor)
        x, scaling_factor = self.mlp(x, scaling_factor)
        # Residual add via QuantAct identity
        x, scaling_factor = self.qact4(
            x, scaling_factor,
            identity=residual, identity_scaling_factor=sf_residual
        )

        return x, scaling_factor


# ═══════════════════════════════════════════════════════════════════════════
# IntEva — full integer-only EVA-02 model
# ═══════════════════════════════════════════════════════════════════════════

class IntEva(nn.Module):
    """
    Integer-only EVA-02 Tiny for ImageNet-1k classification.

    Architecture:
      - QuantConv2d patch embedding (3→192, k=14, s=14)
      - cls_token + absolute pos_embed (added via QuantAct)
      - RotaryEmbeddingCat (reused from timm, computes sin/cos tables)
      - 12× IntEvaBlock
      - IntLayerNorm (fc_norm, the actual final norm)
      - QuantLinear head (192→1000)
    """

    def __init__(self, img_size=336, patch_size=14, in_chans=3,
                 embed_dim=192, depth=12, num_heads=3,
                 mlp_ratio=4 * 2 / 3, num_classes=1000,
                 weight_bit=8, bias_bit=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_prefix_tokens = 1  # cls token
        num_patches = (img_size // patch_size) ** 2

        # ── Input quantization ─────────────────────────────────────────
        self.qact_input = QuantAct()

        # ── Patch embedding ────────────────────────────────────────────
        self.patch_embed = QuantConv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
            padding=0, bias=True,
            weight_bit=weight_bit, bias_bit=bias_bit,
        )
        self.qact_embed = QuantAct()

        # ── Position embedding + cls token ─────────────────────────────
        self.cls_token = Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = Parameter(
            torch.zeros(1, num_patches + self.num_prefix_tokens, embed_dim)
        )
        self.qact_pos = QuantAct(16)  # after adding pos_embed

        # ── RoPE (reuse from timm, stays in float) ────────────────────
        from timm.layers.pos_embed_sincos import RotaryEmbeddingCat
        ref_feat_shape = (img_size // patch_size, img_size // patch_size)
        self.rope = RotaryEmbeddingCat(
            embed_dim // num_heads,
            in_pixels=False,
            feat_shape=ref_feat_shape,
            ref_feat_shape=(16, 16),
        )

        # ── Transformer blocks ─────────────────────────────────────────
        self.blocks = nn.ModuleList([
            IntEvaBlock(embed_dim, num_heads, mlp_ratio,
                        num_prefix_tokens=self.num_prefix_tokens,
                        weight_bit=weight_bit, bias_bit=bias_bit)
            for _ in range(depth)
        ])

        # ── Post-transformer norm (Identity in EVA-02 with global_pool='avg')
        # EVA-02 avg-pool: norm=Identity, fc_norm=LayerNorm.
        # We use nn.Identity here — no normalization after blocks.
        self.norm = nn.Identity()

        # ── Classification head ────────────────────────────────────────
        self.fc_norm = IntLayerNorm(embed_dim)
        self.qact_head = QuantAct()
        self.head = QuantLinear(embed_dim, num_classes, bias=True,
                                weight_bit=weight_bit, bias_bit=bias_bit)

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x):
        # ── Quantize input ─────────────────────────────────────────────
        x, sf = self.qact_input(x)

        # ── Patch embed ────────────────────────────────────────────────
        x, sf = self.patch_embed(x, sf)
        # (B, C, H', W') → (B, N, C)
        x = x.flatten(2).transpose(1, 2)
        sf = sf.reshape(1, 1, -1)

        x, sf = self.qact_embed(x, sf)

        # ── Prepend cls token + absolute position embedding ────────────
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x, sf = self.qact_pos(x)

        # ── Get RoPE embedding ─────────────────────────────────────────
        rot_pos_embed = self.rope.get_embed()

        # ── Transformer blocks ─────────────────────────────────────────
        for blk in self.blocks:
            x, sf = blk(x, sf, rope=rot_pos_embed)

        # ── Post-norm (Identity for EVA-02 avg-pool) ──────────────────
        # self.norm is nn.Identity — no operation needed

        # ── Global average pool (skip cls token) ──────────────────────
        x = x[:, self.num_prefix_tokens:, :].mean(dim=1)
        sf = sf

        # ── FC norm + head ─────────────────────────────────────────────
        # fc_norm expects (B, 1, C) for IntLayerNorm's axis=2 mean
        x = x.unsqueeze(1)
        x, sf = self.fc_norm(x, sf)
        x = x.squeeze(1)
        sf = sf

        x, sf = self.qact_head(x, sf)
        x, sf = self.head(x, sf)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Weight loading utility
# ═══════════════════════════════════════════════════════════════════════════

def load_eva_weights_into_int_eva(int_model, fp_model):
    """
    Copy weights from a timm EVA-02 model (possibly SQ+GPTQ'd) into IntEva.

    QuantLinear/IntLayerNorm/QuantConv2d extend nn.Linear/LayerNorm/Conv2d,
    so weight/bias keys match directly. Extra I-ViT buffers (fc_scaling_factor,
    weight_integer, etc.) are populated during forward passes.
    """
    src_sd = fp_model.state_dict()
    tgt_sd = int_model.state_dict()

    # Key remapping: FP model → IntEva
    key_map = {
        'patch_embed.weight': 'patch_embed.proj.weight',
        'patch_embed.bias': 'patch_embed.proj.bias',
    }

    _ivit_buffers = {
        'fc_scaling_factor', 'weight_integer', 'bias_integer',
        'conv_scaling_factor', 'act_scaling_factor', 'norm_scaling_factor',
    }

    loaded, skipped = [], []
    for tgt_key in tgt_sd:
        # Skip I-ViT buffers (populated during forward)
        if any(buf in tgt_key for buf in _ivit_buffers):
            skipped.append(tgt_key)
            continue

        # Determine source key (direct or remapped)
        src_key = key_map.get(tgt_key, tgt_key)

        if src_key in src_sd and src_sd[src_key].shape == tgt_sd[tgt_key].shape:
            tgt_sd[tgt_key] = src_sd[src_key]
            loaded.append(tgt_key)
        elif tgt_key in src_sd:
            skipped.append(f'{tgt_key} (shape mismatch: '
                          f'{src_sd[tgt_key].shape} vs {tgt_sd[tgt_key].shape})')
        else:
            skipped.append(tgt_key)

    int_model.load_state_dict(tgt_sd, strict=False)
    return loaded, skipped


def freeze_int_model(model):
    """Freeze QuantAct running stats. Works regardless of class identity."""
    for m in model.modules():
        if hasattr(m, 'running_stat') and hasattr(m, 'fix'):
            m.fix()


def unfreeze_int_model(model):
    """Unfreeze QuantAct running stats. Works regardless of class identity."""
    for m in model.modules():
        if hasattr(m, 'running_stat') and hasattr(m, 'unfix'):
            m.unfix()
