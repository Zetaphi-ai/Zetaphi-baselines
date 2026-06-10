"""EuRoC v2 reality-damage suite — IMU-specific corruption ladder.

Does NOT retrain anything: loads the exact checkpoints train.py produced (all
3 seeds), so corruption numbers correspond 1:1 to headline clean numbers.
Corruption randomness seeded per (corruption, seed): every model sees the
IDENTICAL corrupted test stream.

IMU channel map: 0-2 gyro (rad/s), 3-5 accel (m/s^2) — normalized units here.

Corruption ladder (raw fault vs same fault behind a standard embedded driver):
  gyro_bias_drift     slow-growing gyro bias (thermal drift) — the classic IMU killer
  accel_bias          constant accelerometer bias step (calibration error)
  dropped_packets     20% samples lost, fed as zeros (raw bus loss)
  dropped_packets_zoh same loss behind zero-order-hold (standard driver)
  spike_bursts        2% cells get +2sigma impulses (electrical/vibration glitches)
  spike_hampel        identical spikes behind 3-tap Hampel median conditioner
  axis_dead           one gyro + one accel axis dead (zeros)
  axis_frozen         same dead axes frozen at last-known value (smart driver)
  timestamp_jitter    5% of samples swapped with neighbor (timing skew)
  sensor_noise        broadband noise, 0.1sigma

Usage:
  python robustness.py --seq 800 --device cuda:0
"""
import argparse
import json
import os

import numpy as np
import torch

from dataset import get_dataloaders
from models import build_model

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(BASE, 'checkpoints')
RESULTS = os.path.join(BASE, 'results')

MODELS = {
    # name on card -> (build name, ckpt prefix, model_kw)
    'gru':         ('gru', 'gru', {}),
    'tcn':         ('tcn', 'tcn', {}),
    'transformer': ('transformer', 'transformer', {}),
    # ZetaPhi entries withheld (proprietary constants); harness identical.
}

CORRUPTIONS = ['clean', 'gyro_bias_drift', 'accel_bias',
               'dropped_packets', 'dropped_packets_zoh',
               'spike_bursts', 'spike_hampel',
               'axis_dead', 'axis_frozen',
               'timestamp_jitter', 'sensor_noise']


def apply_corruption(x, c_type, gen):
    """x: (B,T,6) normalized. gen: torch.Generator on same device."""
    x = x.clone()
    B, T, C = x.shape
    dev = x.device

    def rand(*shape):
        return torch.rand(*shape, device=dev, generator=gen)

    def randn(*shape):
        return torch.randn(*shape, device=dev, generator=gen)

    if c_type == 'clean':
        return x

    if c_type == 'gyro_bias_drift':
        # linearly growing bias on all gyro axes, up to 0.5sigma by window end
        drift = torch.linspace(0, 0.5, T, device=dev).view(1, T, 1)
        sign = torch.sign(randn(B, 1, 3))
        x[:, :, 0:3] = x[:, :, 0:3] + drift * sign
        return x

    if c_type == 'accel_bias':
        # constant 0.3sigma bias step on accel axes
        sign = torch.sign(randn(B, 1, 3))
        x[:, :, 3:6] = x[:, :, 3:6] + 0.3 * sign
        return x

    if c_type == 'dropped_packets':
        mask = (rand(B, T, 1) > 0.2).float()
        return x * mask

    if c_type == 'dropped_packets_zoh':
        keep = rand(B, T, 1) > 0.2
        keep[:, 0] = True
        idx = torch.where(keep.squeeze(-1), torch.arange(T, device=dev).expand(B, T),
                          torch.zeros(B, T, dtype=torch.long, device=dev))
        idx = torch.cummax(idx, dim=1).values
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(B, T, C))

    if c_type == 'spike_bursts':
        mask = rand(B, T, C) < 0.02
        return x + mask.float() * 2.0

    if c_type == 'spike_hampel':
        mask = rand(B, T, C) < 0.02
        x = x + mask.float() * 2.0
        xp = x.transpose(1, 2)
        pad = torch.nn.functional.pad(xp, (1, 1), mode='replicate')
        med = pad.unfold(-1, 3, 1).median(dim=-1).values
        return med.transpose(1, 2)

    if c_type == 'axis_dead':
        g = int(torch.randint(3, (1,), generator=gen, device=dev).item())
        a = 3 + int(torch.randint(3, (1,), generator=gen, device=dev).item())
        x[:, :, [g, a]] = 0.0
        return x

    if c_type == 'axis_frozen':
        g = int(torch.randint(3, (1,), generator=gen, device=dev).item())
        a = 3 + int(torch.randint(3, (1,), generator=gen, device=dev).item())
        x[:, :, [g, a]] = x[:, :1, [g, a]]
        return x

    if c_type == 'timestamp_jitter':
        # 5% of interior samples swapped with their successor
        swap = (rand(B, T - 1) < 0.05)
        idx = torch.arange(T, device=dev).expand(B, T).clone()
        i = torch.arange(T - 1, device=dev)
        nxt = idx[:, 1:]
        curr = idx[:, :-1]
        new_curr = torch.where(swap, nxt, curr)
        new_next = torch.where(swap, curr, nxt)
        idx[:, :-1] = new_curr
        idx[:, 1:] = new_next
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(B, T, C))

    if c_type == 'sensor_noise':
        return x + randn(B, T, C) * 0.1

    raise ValueError(c_type)


@torch.no_grad()
def eval_corrupted(model, dl, device, tgt_std, c_type, base_seed):
    gen = torch.Generator(device=device)
    gen.manual_seed(int(np.uint32(hash((c_type, base_seed)) & 0xffffffff)))
    preds, trues = [], []
    for x, y in dl:
        x = apply_corruption(x.to(device), c_type, gen)
        preds.append(model(x).float().cpu().numpy())
        trues.append(y.numpy())
    p = np.concatenate(preds)
    t = np.concatenate(trues)
    pp, tt = p * tgt_std, t * tgt_std
    dp = float(np.sqrt(np.mean((pp[:, :3] - tt[:, :3]) ** 2)) * 100)
    rot = float(np.sqrt(np.mean((pp[:, 3:] - tt[:, 3:]) ** 2)) * 180 / np.pi)
    return dp, rot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', type=int, default=800)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seeds', default='42,123,314')
    ap.add_argument('--models', default=','.join(MODELS))
    args = ap.parse_args()

    device = args.device
    _, _, test_dl, meta = get_dataloaders(seq_len=args.seq, stride=20)
    tgt_std = np.array(meta['tgt_std'], dtype=np.float32)

    out_path = os.path.join(RESULTS, f'robustness_seq{args.seq}.jsonl')
    seen = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                seen.add((r['model'], r['seed'], r['corruption']))

    for mname in args.models.split(','):
        build_name, prefix, mkw = MODELS[mname]
        for seed in [int(s) for s in args.seeds.split(',')]:
            ck_path = os.path.join(CKPT, f'{prefix}_seq{args.seq}_s{seed}.pt')
            if not os.path.exists(ck_path):
                print(f'MISSING {ck_path}', flush=True)
                continue
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            state = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
            model = build_model(build_name, **mkw).to(device)
            model.load_state_dict(state)
            model.eval()
            if build_name == 'zetaphi':
                torch._dynamo.reset()
                model = torch.compile(model)
            for c in CORRUPTIONS:
                if (mname, seed, c) in seen:
                    continue
                dp, rot = eval_corrupted(model, test_dl, device, tgt_std, c, seed)
                row = {'model': mname, 'seed': seed, 'seq_len': args.seq,
                       'corruption': c, 'dp_rmse_cm': round(dp, 2),
                       'rot_rmse_deg': round(rot, 3)}
                with open(out_path, 'a') as f:
                    f.write(json.dumps(row) + '\n')
                print(f"{mname:<14} s{seed:<4} {c:<22} dp {dp:7.2f}cm rot {rot:7.3f}deg",
                      flush=True)

    print('ROBUSTNESS DONE', out_path, flush=True)


if __name__ == '__main__':
    main()
