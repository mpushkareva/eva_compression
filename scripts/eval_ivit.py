"""
I-ViT integer-only evaluation for EVA-02 Tiny on ImageNet-1k.

Pipeline:
  1. Load EVA-02 Tiny from timm (pretrained FP32)
  2. Apply SmoothQuant α=0.5
  3. Apply GPTQ W4
  4. Build IntEva model, copy SQ+GPTQ weights
  5. Calibrate: unfreeze → run 128 samples → freeze
  6. Evaluate Top-1/Top-5 on ImageNet-1k val
"""

import argparse
import io
import contextlib
import sys
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import timm
import timm.data

# ── project imports ────────────────────────────────────────────────────────
_project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(_project_root))

from models.int_eva import (
    IntEva, load_eva_weights_into_int_eva,
    freeze_int_model, unfreeze_int_model,
)

# Reuse SQ+GPTQ utilities from existing script
_scripts_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(_scripts_dir))

# Import GPTQ dependencies
_gptq_dir = os.path.join(_project_root, 'gptq')
sys.path.insert(0, os.path.abspath(_gptq_dir))
from gptq import GPTQ          # noqa: E402
from quant import Quantizer     # noqa: E402
from modelutils import find_layers  # noqa: E402
sys.path.pop(0)

from torchvision import datasets  # noqa: E402





# ── shared helpers ─────────────────────────────────────────────────────────

class ImageNetFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir()
        )
        class_to_idx = {cls_name: int(cls_name) for cls_name in classes}
        return classes, class_to_idx


def accuracy_from_logits(logits, targets, topk=(1, 5)):
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k * (100.0 / batch_size)).item())
    return res


def get_calibration_loader(train_dir, model_or_cfg, nsamples=128, seed=0):
    if hasattr(model_or_cfg, 'pretrained_cfg'):
        data_cfg = timm.data.resolve_data_config(model_or_cfg.pretrained_cfg)
    else:
        data_cfg = timm.data.resolve_data_config(model_or_cfg)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    train_ds = ImageNetFolder(train_dir, transform=transform)
    rng = torch.Generator()
    rng.manual_seed(seed)
    indices = torch.randperm(len(train_ds), generator=rng)[:nsamples].tolist()
    subset = Subset(train_ds, indices)
    return DataLoader(subset, batch_size=1, shuffle=False, num_workers=4,
                      pin_memory=True)


class Catcher(nn.Module):
    def __init__(self, module, cache):
        super().__init__()
        self.module = module
        self.cache = cache

    def forward(self, x, **kwargs):
        self.cache['inputs'].append(x.detach().cpu())
        self.cache['kwargs'] = kwargs
        raise ValueError


# ── SmoothQuant (reused from eval_smoothquant_gptq.py) ─────────────────────

@torch.no_grad()
def collect_act_scales(model, cal_loader, dev):
    model.eval()
    model.to(dev)
    act_scales = {}

    def make_hook(key):
        def hook(_, _inp, out):
            absmax = out.abs().amax(dim=(0, 1))
            if key not in act_scales:
                act_scales[key] = absmax.clone()
            else:
                act_scales[key] = torch.max(act_scales[key], absmax)
        return hook

    handles = []
    for i, block in enumerate(model.blocks):
        handles.append(block.norm1.register_forward_hook(make_hook(f'{i}.norm1')))
        handles.append(block.norm2.register_forward_hook(make_hook(f'{i}.norm2')))

    for imgs, _ in cal_loader:
        model(imgs.to(dev))

    for h in handles:
        h.remove()
    model.cpu()
    return act_scales


@torch.no_grad()
def apply_smoothquant(model, act_scales, alpha=0.5):
    for i, block in enumerate(model.blocks):
        _smooth_pair(block.norm1, block.attn.qkv, act_scales[f'{i}.norm1'], alpha)
        _smooth_pair(block.norm2, block.mlp.fc1, act_scales[f'{i}.norm2'], alpha)


def _smooth_pair(norm, linear, act_scale, alpha):
    act_scale = act_scale.to(norm.weight.device)
    weight_scale = linear.weight.abs().max(dim=0)[0].clamp(min=1e-8)
    s = (act_scale.pow(alpha) / weight_scale.pow(1 - alpha)).clamp(min=1e-8)
    norm.weight.div_(s)
    if norm.bias is not None:
        norm.bias.div_(s)
    linear.weight.mul_(s.view(1, -1))


# ── GPTQ (reused from eval_smoothquant_gptq.py) ───────────────────────────

@torch.no_grad()
def quantize_blocks(model, cal_loader, dev, wbits, groupsize, actorder, sym,
                    verbose=False):
    model.eval()
    blocks = model.blocks

    model.patch_embed.to(dev)
    model.cls_token.data = model.cls_token.to(dev)
    model.pos_embed.data = model.pos_embed.to(dev)
    if hasattr(model, 'norm_pre'):
        model.norm_pre.to(dev)

    img_size = model.pretrained_cfg.get('input_size', (3, 336, 336))
    dummy = torch.randn(1, *img_size, device=dev)
    x_dummy = model.patch_embed(dummy)
    x_dummy, rot_pos_embed = model._pos_embed(x_dummy)
    rot_pos_embed = rot_pos_embed.detach()
    del x_dummy, dummy

    cache = {'inputs': [], 'kwargs': {}}
    blocks[0] = Catcher(blocks[0], cache)

    _iter = tqdm(cal_loader, desc='Catching') if verbose else cal_loader
    for imgs, _ in _iter:
        try:
            imgs = imgs.to(dev)
            x = model.patch_embed(imgs)
            x, _ = model._pos_embed(x)
            if hasattr(model, 'norm_pre'):
                x = model.norm_pre(x)
            blocks[0](x, rope=rot_pos_embed)
        except ValueError:
            pass

    blocks[0] = blocks[0].module

    model.patch_embed.cpu()
    model.cls_token.data = model.cls_token.cpu()
    model.pos_embed.data = model.pos_embed.cpu()
    if hasattr(model, 'norm_pre'):
        model.norm_pre.cpu()

    inps = cache['inputs']
    nsamples = len(inps)
    torch.cuda.empty_cache()
    outs = [None] * nsamples

    for block_idx in range(len(blocks)):
        if verbose:
            print(f'\n=== Block {block_idx} ===')
        block = blocks[block_idx].to(dev)

        subset = find_layers(block, layers=[nn.Linear])
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                wbits, perchannel=True, sym=sym, mse=False
            )

        def make_add_batch(name):
            def hook(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return hook

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(make_add_batch(name)))

        attn = getattr(block, 'attn', None)
        _saved = None
        if attn is not None and hasattr(attn, 'qkv_bias_separate'):
            _saved = attn.qkv_bias_separate
            attn.qkv_bias_separate = True

        for j in range(nsamples):
            outs[j] = block(
                inps[j].to(dev), rope=rot_pos_embed.to(dev)
            ).detach().cpu()
        for h in handles:
            h.remove()
        if _saved is not None:
            attn.qkv_bias_separate = _saved

        for name in subset:
            if verbose:
                print(f'  {name} ...', end=' ')
                gptq[name].fasterquant(groupsize=groupsize, actorder=actorder)
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    gptq[name].fasterquant(groupsize=groupsize, actorder=actorder)
            gptq[name].free()

        for j in range(nsamples):
            outs[j] = block(
                inps[j].to(dev), rope=rot_pos_embed.to(dev)
            ).detach().cpu()

        blocks[block_idx] = block.cpu()
        del block, gptq
        torch.cuda.empty_cache()
        inps, outs = outs, [None] * nsamples


# ── evaluation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, dev, verbose=False):
    model.eval()
    model.to(dev)
    top1_sum = 0.0
    top5_sum = 0.0
    n_samples = 0

    _iter = tqdm(val_loader, desc='Evaluating', unit='batch') if verbose else val_loader
    for images, targets in _iter:
        images = images.to(dev)
        targets = targets.to(dev)
        logits = model(images)
        top1, top5 = accuracy_from_logits(
            logits.cpu(), targets.cpu(), topk=(1, 5)
        )
        bs = targets.size(0)
        top1_sum += top1 * bs
        top5_sum += top5 * bs
        n_samples += bs

    top1_avg = top1_sum / n_samples
    top5_avg = top5_sum / n_samples
    return top1_avg, top5_avg, n_samples


# ── I-ViT calibration ─────────────────────────────────────────────────────

@torch.no_grad()
def calibrate_int_model(int_model, cal_loader, dev, verbose=False):
    """Run calibration data through IntEva to learn QuantAct ranges."""
    int_model.eval()
    int_model.to(dev)
    unfreeze_int_model(int_model)

    _iter = tqdm(cal_loader, desc='Calibrating I-ViT') if verbose else cal_loader
    for imgs, _ in _iter:
        int_model(imgs.to(dev))

    freeze_int_model(int_model)
    int_model.cpu()


# ── main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='I-ViT integer-only evaluation for EVA-02 Tiny'
    )
    p.add_argument('--model', type=str,
                   default='eva02_tiny_patch14_336.mim_in22k_ft_in1k')
    p.add_argument('--wbits', type=int, default=4,
                   help='Weight quantization bits (GPTQ, default: 4)')
    p.add_argument('--abits', type=int, default=8,
                   help='Activation quantization bits (default: 8)')
    p.add_argument('--alpha', type=float, default=0.5,
                   help='SmoothQuant migration strength (default: 0.5)')
    p.add_argument('--groupsize', type=int, default=128)
    p.add_argument('--actorder', action='store_true')
    p.add_argument('--sym', action='store_true')
    p.add_argument('--nsamples', type=int, default=128)
    p.add_argument('--val-dir', type=str, required=True)
    p.add_argument('--train-dir', type=str, default=None,
                   help='Training data dir for calibration (default: inferred)')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--verbose', '-v', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    dev = torch.device(args.device)

    # Infer train dir from val dir if not given
    if args.train_dir is None:
        args.train_dir = os.path.join(os.path.dirname(args.val_dir), 'train')
        print(f'Inferred train dir: {args.train_dir}')

    # ── Step 1: Load FP32 model ────────────────────────────────────────
    print(f'Loading {args.model} ...')
    fp_model = timm.create_model(args.model, pretrained=True)
    fp_model.eval()
    print(f'Model loaded – {sum(p.numel() for p in fp_model.parameters())/1e6:.1f}M params')

    # ── Step 2: SmoothQuant ────────────────────────────────────────────
    cal_loader = get_calibration_loader(
        args.train_dir, fp_model, nsamples=args.nsamples, seed=args.seed
    )
    print(f'\nSmoothQuant: alpha={args.alpha}')
    act_scales = collect_act_scales(fp_model, cal_loader, dev)
    apply_smoothquant(fp_model, act_scales, alpha=args.alpha)

    # ── Step 3: GPTQ ──────────────────────────────────────────────────
    print(f'GPTQ: wbits={args.wbits}, groupsize={args.groupsize}, '
          f'actorder={args.actorder}, sym={args.sym}')
    cal_loader = get_calibration_loader(
        args.train_dir, fp_model, nsamples=args.nsamples, seed=args.seed
    )
    quantize_blocks(
        fp_model, cal_loader, dev,
        wbits=args.wbits, groupsize=args.groupsize,
        actorder=args.actorder, sym=args.sym,
        verbose=args.verbose,
    )

    # ── Step 4: Build IntEva and copy weights ──────────────────────────
    print('\nBuilding integer-only model ...')
    cfg = fp_model.pretrained_cfg
    img_size = cfg.get('input_size', (3, 336, 336))[1]

    int_model = IntEva(
        img_size=img_size,
        patch_size=14,
        in_chans=3,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4 * 2 / 3,
        num_classes=1000,
        weight_bit=args.wbits,
        bias_bit=32,
    )

    loaded, skipped = load_eva_weights_into_int_eva(int_model, fp_model)
    print(f'  Loaded {len(loaded)} params, skipped {len(skipped)} '
          f'(buffers / shape mismatches)')
    if args.verbose:
        for s in skipped:
            print(f'    skip: {s}')

    del fp_model
    torch.cuda.empty_cache()

    # ── Step 5: Calibrate QuantAct ranges ──────────────────────────────
    print('\nCalibrating integer-only model ...')
    ivit_data_cfg = {'input_size': (3, img_size, img_size)}
    cal_loader = get_calibration_loader(
        args.train_dir, ivit_data_cfg, nsamples=args.nsamples, seed=args.seed
    )
    calibrate_int_model(int_model, cal_loader, dev, verbose=args.verbose)

    # ── Step 6: Evaluate ───────────────────────────────────────────────
    data_cfg = timm.data.resolve_data_config(
        {'input_size': (3, img_size, img_size)}
    )
    val_transform = timm.data.create_transform(**data_cfg, is_training=False)
    val_ds = ImageNetFolder(args.val_dir, transform=val_transform)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f'\nEvaluating on {len(val_ds)} images ...')
    top1, top5, n = evaluate(int_model, val_loader, dev, verbose=args.verbose)

    tag = (f'I-ViT Integer-Only | SQ(α={args.alpha}) + GPTQ W{args.wbits} '
           f'A{args.abits}')
    print(f'\n{"=" * 60}')
    print(f'  [{tag}]')
    print(f'  Top-1 Accuracy: {top1:.2f}%')
    print(f'  Top-5 Accuracy: {top5:.2f}%')
    print(f'  Total Samples:  {n}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
