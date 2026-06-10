"""C-MAPSS robustness suite — the plan's 8 corruptions.

Key difference vs v1: does NOT retrain anything. Loads the exact checkpoints
produced by train.py (all 3 seeds), so corruption numbers correspond 1:1 to
the headline clean numbers. Reports clean RMSE, corrupted RMSE, and the
degradation ratio (corrupted/clean, lower=better for RMSE).

Corruption randomness is seeded per (corruption, seed) so every model sees the
IDENTICAL corrupted test set.

Usage:
  python robustness.py --fd 001 --seq 50 --models gru,tcn,transformer --seeds 42,123,314
"""
import argparse
import json
import math
import os
import zlib

import numpy as np
import torch

from dataset import get_dataloaders
from models import build_model

BASE = os.path.dirname(os.path.abspath(__file__))


def apply_corruption(x, c_type, gen):
    """x: (B,T,C) on device. gen: torch.Generator on same device."""
    x = x.clone()
    B, T, C = x.shape
    dev = x.device

    def rand(*shape):
        return torch.rand(*shape, device=dev, generator=gen)

    def randn(*shape):
        return torch.randn(*shape, device=dev, generator=gen)

    if c_type == 'clean':
        return x

    if c_type == 'sensor_noise':
        return x + randn(B, T, C) * 0.05

    if c_type == 'missing_cycles_zero':
        # 20% of timesteps zeroed (raw packet loss)
        mask = (rand(B, T, 1) > 0.2).float()
        return x * mask

    if c_type == 'missing_cycles_zoh':
        # same 20% loss but with a zero-order-hold driver (standard embedded logic)
        keep = rand(B, T, 1) > 0.2
        keep[:, 0] = True  # first sample always arrives
        idx = torch.where(keep.squeeze(-1), torch.arange(T, device=dev).expand(B, T),
                          torch.zeros(B, T, dtype=torch.long, device=dev))
        idx = torch.cummax(idx, dim=1).values  # forward-fill index
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(B, T, C))

    if c_type == 'channel_dropout':
        # 3 random sensor channels dead per batch draw
        dead = torch.randperm(C, generator=gen, device=dev)[:3]
        x[:, :, dead] = 0.0
        return x

    if c_type == 'channel_dropout_frozen':
        # same 3 dead channels, but a smart driver freezes them at last-known
        # value (here: value at window start) instead of feeding raw zeros.
        # Standard embedded behavior for a stuck/disconnected sensor.
        dead = torch.randperm(C, generator=gen, device=dev)[:3]
        x[:, :, dead] = x[:, :1, dead]
        return x

    if c_type == 'sensor_drift':
        # gradual linear drift on 4 random channels (calibration drift)
        ch = torch.randperm(C, generator=gen, device=dev)[:4]
        drift = torch.linspace(0, 0.15, T, device=dev).view(1, T, 1)
        x[:, :, ch] = x[:, :, ch] + drift
        return x

    if c_type == 'bias_offset':
        # constant bias on one random channel
        ch = int(torch.randint(C, (1,), generator=gen, device=dev).item())
        x[:, :, ch] += 0.3
        return x

    if c_type == 'spike_anomalies':
        # impulse spikes: 2% of cells get +2.0 (electrical glitches)
        mask = rand(B, T, C) < 0.02
        return x + mask.float() * 2.0

    if c_type == 'spike_hampel':
        # identical spikes, but routed through a 3-tap median conditioner —
        # standard embedded signal conditioning (Hampel-style), applied to the
        # whole stream for EVERY model equally.
        mask = rand(B, T, C) < 0.02
        x = x + mask.float() * 2.0
        xp = x.transpose(1, 2)                       # (B,C,T)
        pad = torch.nn.functional.pad(xp, (1, 1), mode='replicate')
        med = pad.unfold(-1, 3, 1).median(dim=-1).values  # (B,C,T)
        return med.transpose(1, 2)

    if c_type == 'delayed_channel':
        # one random channel lags by 5 cycles (fusion timing mismatch)
        ch = int(torch.randint(C, (1,), generator=gen, device=dev).item())
        lag = 5
        shifted = torch.roll(x[:, :, ch], shifts=lag, dims=1)
        shifted[:, :lag] = x[:, :1, ch]  # ZOH the head
        x[:, :, ch] = shifted
        return x

    raise ValueError(c_type)


CORRUPTIONS = ['clean', 'sensor_noise', 'missing_cycles_zero', 'missing_cycles_zoh',
               'channel_dropout', 'channel_dropout_frozen', 'sensor_drift', 'bias_offset',
               'spike_anomalies', 'spike_hampel', 'delayed_channel']


@torch.no_grad()
def eval_rmse(model, dl, device, c_type, corr_seed):
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(corr_seed)
    se, n = 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        x = apply_corruption(x, c_type, gen)
        out = model(x)
        se += torch.sum((out - y) ** 2).item()
        n += x.size(0)
    return math.sqrt(se / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fd', default='001')
    ap.add_argument('--seq', type=int, default=50)
    ap.add_argument('--models', default='gru,tcn,transformer')
    ap.add_argument('--seeds', default='42,123,314')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--tag', default='robustness')
    ap.add_argument('--ckpt-prefix', default='', help="e.g. 'tuned_' to evaluate tuned checkpoints")
    args = ap.parse_args()

    _, _, test_dl, in_ch = get_dataloaders(fd_num=args.fd, seq_len=args.seq)
    model_names = args.models.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]

    out_path = os.path.join(BASE, 'results', f'{args.tag}_fd{args.fd}_seq{args.seq}.jsonl')
    rows = {}
    for mname in model_names:
        per_corr = {c: [] for c in CORRUPTIONS}
        for seed in seeds:
            ck = os.path.join(BASE, 'checkpoints',
                              f'{args.ckpt_prefix}{mname}_fd{args.fd}_seq{args.seq}_s{seed}.pt')
            if not os.path.exists(ck):
                print(f"MISSING checkpoint: {ck} — run train.py first")
                continue
            model = build_model(mname, in_ch).to(args.device)
            model.load_state_dict(torch.load(ck, map_location=args.device))
            for c in CORRUPTIONS:
                # corruption seed fixed per (corruption, model-seed): identical
                # corrupted data across architectures (crc32 = stable across processes,
                # unlike Python's salted hash())
                corr_seed = zlib.crc32(f'{c}_{seed}'.encode()) % (2**31)
                rmse = eval_rmse(model, test_dl, args.device, c, corr_seed=corr_seed)
                per_corr[c].append(rmse)
        rows[mname] = per_corr
        with open(out_path, 'a') as f:
            f.write(json.dumps({'model': mname, 'seq': args.seq,
                                'results': {c: v for c, v in per_corr.items()}}) + '\n')

    # pretty table: mean RMSE and degradation ratio vs own clean
    print(f"\n=== ROBUSTNESS (FD{args.fd}, seq={args.seq}, mean of {len(seeds)} seeds) ===")
    hdr = f"{'corruption':<22}" + ''.join(f"{m:>16}" for m in model_names)
    print(hdr)
    print('-' * len(hdr))
    for c in CORRUPTIONS:
        line = f"{c:<22}"
        for m in model_names:
            v = rows[m][c]
            if not v:
                line += f"{'--':>16}"
                continue
            mean = np.mean(v)
            if c == 'clean':
                line += f"{mean:>11.2f}     "
            else:
                ratio = mean / np.mean(rows[m]['clean'])
                line += f"{mean:>9.2f}({ratio:4.2f})"
        print(line)
    print("\n(ratio = corrupted RMSE / clean RMSE, lower is better)")


if __name__ == '__main__':
    main()
