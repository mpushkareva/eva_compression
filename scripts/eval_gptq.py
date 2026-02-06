"""
GPTQ post-training quantization for EVA-02 Tiny on ImageNet-1k.

Applies GPTQ block-by-block to each EvaBlock using calibration data from
ImageNet train split, then evaluates Top-1/Top-5 accuracy on the val split.
"""

import argparse
import sys
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import timm
import timm.data

# ---------------------------------------------------------------------------
# Import GPTQ classes – the submodule uses bare imports (from quant import *)
# so we add its directory to sys.path before importing.
# ---------------------------------------------------------------------------
_gptq_dir = os.path.join(os.path.dirname(__file__), '..', 'gptq')
sys.path.insert(0, os.path.abspath(_gptq_dir))

from gptq import GPTQ          # noqa: E402
from quant import Quantizer     # noqa: E402
from modelutils import find_layers  # noqa: E402

sys.path.pop(0)

from torchvision import datasets  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────

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


# ── calibration data ──────────────────────────────────────────────────────

def get_calibration_loader(train_dir, model, nsamples=128, seed=0):
    """Build a DataLoader of *nsamples* random images from ImageNet train."""
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    train_ds = datasets.ImageFolder(train_dir, transform=transform)

    rng = torch.Generator()
    rng.manual_seed(seed)
    indices = torch.randperm(len(train_ds), generator=rng)[:nsamples].tolist()
    subset = Subset(train_ds, indices)

    return DataLoader(subset, batch_size=1, shuffle=False, num_workers=4,
                      pin_memory=True)


# ── Catcher (capture inputs to blocks[0]) ─────────────────────────────────

class Catcher(nn.Module):
    """Replace blocks[0] to capture its input activations during calibration."""
    def __init__(self, module, cache):
        super().__init__()
        self.module = module
        self.cache = cache

    def forward(self, x, **kwargs):
        self.cache['inputs'].append(x.detach().cpu())
        self.cache['kwargs'] = kwargs
        raise ValueError  # stop forward early


# ── block-by-block GPTQ quantization ─────────────────────────────────────

@torch.no_grad()
def quantize_blocks(model, cal_loader, dev, wbits, groupsize, actorder, sym):
    """Run GPTQ on every EvaBlock sequentially."""

    model.eval()

    # ── 1. Pre-block feature extraction (patch_embed → _pos_embed → norm_pre)
    #    We also need rot_pos_embed for each block's forward.
    blocks = model.blocks

    # Move pre-block layers to device
    model.patch_embed.to(dev)
    model.cls_token.data = model.cls_token.to(dev)
    model.pos_embed.data = model.pos_embed.to(dev)
    if hasattr(model, 'norm_pre'):
        model.norm_pre.to(dev)

    # Capture rot_pos_embed from _pos_embed()
    # Run a dummy input to get the rope value.
    dummy = torch.randn(1, 3, 336, 336, device=dev)
    x_dummy = model.patch_embed(dummy)
    x_dummy, rot_pos_embed = model._pos_embed(x_dummy)
    rot_pos_embed = rot_pos_embed.detach()  # constant across batches
    del x_dummy, dummy

    # ── 2. Collect inputs to blocks[0] via Catcher
    cache = {'inputs': [], 'kwargs': {}}
    blocks[0] = Catcher(blocks[0], cache)

    for imgs, _ in tqdm(cal_loader, desc='Catching calibration inputs'):
        try:
            imgs = imgs.to(dev)
            x = model.patch_embed(imgs)
            x, _rope = model._pos_embed(x)
            if hasattr(model, 'norm_pre'):
                x = model.norm_pre(x)
            blocks[0](x, rope=rot_pos_embed)
        except ValueError:
            pass

    # Restore original block
    blocks[0] = blocks[0].module

    # Move pre-block layers back to cpu to free memory
    model.patch_embed.cpu()
    model.cls_token.data = model.cls_token.cpu()
    model.pos_embed.data = model.pos_embed.cpu()
    if hasattr(model, 'norm_pre'):
        model.norm_pre.cpu()

    inps = cache['inputs']  # list of [1, seq, dim] cpu tensors
    nsamples = len(inps)
    print(f'Captured {nsamples} calibration samples, '
          f'shape {inps[0].shape}')

    torch.cuda.empty_cache()

    # ── 3. Quantize each block
    outs = [None] * nsamples

    for block_idx in range(len(blocks)):
        print(f'\n=== Quantizing block {block_idx} ===')
        block = blocks[block_idx].to(dev)

        subset = find_layers(block, layers=[nn.Linear])
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                wbits, perchannel=True, sym=sym, mse=False
            )

        # ── forward hooks to accumulate Hessian
        def make_add_batch(name):
            def hook(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return hook

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(make_add_batch(name)))

        # Run calibration through this block
        for j in range(nsamples):
            outs[j] = block(inps[j].to(dev), rope=rot_pos_embed.to(dev)).detach().cpu()
        for h in handles:
            h.remove()

        # ── Quantize weights
        for name in subset:
            print(f'  {name} ...', end=' ')
            gptq[name].fasterquant(
                groupsize=groupsize, actorder=actorder
            )
            gptq[name].free()

        # ── Re-run block with quantized weights to get outputs for next block
        for j in range(nsamples):
            outs[j] = block(inps[j].to(dev), rope=rot_pos_embed.to(dev)).detach().cpu()

        blocks[block_idx] = block.cpu()
        del block, gptq
        torch.cuda.empty_cache()

        # Chain: next block's input = this block's output
        inps, outs = outs, [None] * nsamples

    print('\nGPTQ quantization complete.')


# ── evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, dev):
    model.eval()
    model.to(dev)

    top1_sum = 0.0
    top5_sum = 0.0
    n_samples = 0

    for images, targets in tqdm(val_loader, desc='Evaluating', unit='batch'):
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


# ── main ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='GPTQ quantization for EVA-02 Tiny on ImageNet-1k'
    )
    p.add_argument('--wbits', type=int, default=4,
                    help='Quantization bit-width (default: 4)')
    p.add_argument('--groupsize', type=int, default=128,
                    help='Group size for quantization (default: 128)')
    p.add_argument('--actorder', action='store_true',
                    help='Activation-based column reordering')
    p.add_argument('--sym', action='store_true',
                    help='Symmetric quantization')
    p.add_argument('--nsamples', type=int, default=128,
                    help='Number of calibration samples (default: 128)')
    p.add_argument('--val-dir', type=str, required=True,
                    help='ImageNet validation directory')
    p.add_argument('--train-dir', type=str, required=True,
                    help='ImageNet train directory (for calibration)')
    p.add_argument('--batch-size', type=int, default=64,
                    help='Evaluation batch size (default: 64)')
    p.add_argument('--no-quant', action='store_true',
                    help='Skip quantization – evaluate FP32 baseline')
    p.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    dev = torch.device(args.device)

    # ── Load model
    print('Loading EVA-02 Tiny ...')
    model = timm.create_model(
        'eva02_tiny_patch14_336.mim_in22k_ft_in1k', pretrained=True
    )
    model.eval()
    print(f'Model loaded – {sum(p.numel() for p in model.parameters())/1e6:.1f}M params')

    # ── Quantize (unless --no-quant)
    if not args.no_quant:
        print(f'\nGPTQ config: wbits={args.wbits}, groupsize={args.groupsize}, '
              f'actorder={args.actorder}, sym={args.sym}, nsamples={args.nsamples}')
        cal_loader = get_calibration_loader(
            args.train_dir, model, nsamples=args.nsamples, seed=args.seed
        )
        quantize_blocks(
            model, cal_loader, dev,
            wbits=args.wbits, groupsize=args.groupsize,
            actorder=args.actorder, sym=args.sym,
        )
    else:
        print('Skipping quantization (FP32 baseline)')

    # ── Build evaluation loader using timm's data config
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    val_transform = timm.data.create_transform(**data_cfg, is_training=False)
    val_ds = datasets.ImageFolder(args.val_dir, transform=val_transform)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Evaluate
    print(f'\nEvaluating on {len(val_ds)} images ...')
    top1, top5, n = evaluate(model, val_loader, dev)

    print(f'\n{"="*60}')
    tag = 'FP32 baseline' if args.no_quant else f'GPTQ W{args.wbits} g{args.groupsize}'
    print(f'  [{tag}]')
    print(f'  Top-1 Accuracy: {top1:.2f}%')
    print(f'  Top-5 Accuracy: {top5:.2f}%')
    print(f'  Total Samples:  {n}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
