"""EuRoC v2 training harness — IMU-only relative odometry.

Protocol (honest-card rules, inherited from cmapss_v2):
  - cross-sequence split: train MH01/02/04, val MH03, test MH05.
  - model selection: best VALIDATION loss epoch -> evaluated on test ONCE.
  - 3 seeds per cell.
  - metrics in PHYSICAL units: dp RMSE (cm), rot RMSE (deg), plus normalized MSE.

Compute card baked in per run (not an afterthought):
  - train wall-clock (s), peak train VRAM (MB)
  - batch-1 inference latency p50/p95 (ms, CUDA-synced, 200 iters) + eval peak VRAM
  - checkpoints are self-describing (model config stored alongside weights).

Usage:
  python train.py --seqs 200,400,800 --models gru,tcn,transformer,zetaphi \
                  --seeds 42,123,314 --device cuda:0 --tag maingrid
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders
from models import build_model, count_params

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, 'results')
CKPT_DIR = os.path.join(BASE, 'checkpoints')

_LOADER_CACHE = {}


def loaders(seq_len, batch_size=64):
    key = (seq_len, batch_size)
    if key not in _LOADER_CACHE:
        _LOADER_CACHE[key] = get_dataloaders(seq_len=seq_len, stride=20,
                                             batch_size=batch_size)
    return _LOADER_CACHE[key]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@torch.no_grad()
def evaluate(model, dl, device, tgt_std):
    model.eval()
    preds, trues = [], []
    for x, y in dl:
        out = model(x.to(device))
        preds.append(out.float().cpu().numpy())
        trues.append(y.numpy())
    p = np.concatenate(preds)
    t = np.concatenate(trues)
    mse_n = float(np.mean((p - t) ** 2))
    # back to physical units
    pp, tt = p * tgt_std, t * tgt_std
    dp_rmse_cm = float(np.sqrt(np.mean((pp[:, :3] - tt[:, :3]) ** 2)) * 100)
    rot_rmse_deg = float(np.sqrt(np.mean((pp[:, 3:] - tt[:, 3:]) ** 2)) * 180 / np.pi)
    return mse_n, dp_rmse_cm, rot_rmse_deg


@torch.no_grad()
def latency_card(model, seq_len, device, iters=200, warmup=30):
    model.eval()
    x1 = torch.randn(1, seq_len, 6, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    for _ in range(warmup):
        model(x1)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x1)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
    return (float(np.percentile(t, 50)), float(np.percentile(t, 95)),
            round(peak_mb, 1))


def run_one(model_name, seq_len, seed, device, epochs=40, lr=1e-3, wd=1e-4,
            batch_size=64, compile_zeta=True, save_ckpt=True, model_kw=None,
            tag=''):
    set_seed(seed)
    train_dl, val_dl, test_dl, meta = loaders(seq_len, batch_size)
    tgt_std = np.array(meta['tgt_std'], dtype=np.float32)

    model_kw = model_kw or {}
    model = build_model(model_name, **model_kw).to(device)
    n_params = count_params(model)

    run_model = model
    if compile_zeta and model_name.startswith('zetaphi'):
        run_model = torch.compile(model)  # vectorized layout fuses under inductor

    opt = torch.optim.AdamW(run_model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()

    torch.cuda.reset_peak_memory_stats(device)
    best_val = float('inf')
    best_state = None
    t0 = time.time()
    for ep in range(epochs):
        run_model.train()
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(run_model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(run_model.parameters(), 1.0)
            opt.step()
        sched.step()
        val_mse, _, _ = evaluate(run_model, val_dl, device, tgt_std)
        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
    train_time = time.time() - t0
    train_peak_mb = torch.cuda.max_memory_allocated(device) / 1e6

    model.load_state_dict(best_state)
    eval_model = (torch.compile(model)
                  if (compile_zeta and model_name.startswith('zetaphi')) else model)
    test_mse, dp_cm, rot_deg = evaluate(eval_model, test_dl, device, tgt_std)
    val_mse_n, val_dp_cm, val_rot_deg = evaluate(eval_model, val_dl, device, tgt_std)
    lat_p50, lat_p95, eval_peak_mb = latency_card(eval_model, seq_len, device)

    if save_ckpt:
        ck = {'state_dict': best_state, 'model': model_name, 'seq_len': seq_len,
              'seed': seed, 'model_kw': model_kw}
        if hasattr(model, 'hparams'):
            ck['hparams'] = model.hparams  # keep checkpoints self-describing
        torch.save(ck, os.path.join(
            CKPT_DIR, f'{model_name}{tag}_seq{seq_len}_s{seed}.pt'))

    return {
        'model': model_name, 'tag': tag, 'seq_len': seq_len, 'seed': seed,
        'params': n_params, 'epochs': epochs, 'lr': lr, 'wd': wd,
        'val_mse_n': best_val,
        'test_mse_n': test_mse, 'test_dp_rmse_cm': round(dp_cm, 2),
        'test_rot_rmse_deg': round(rot_deg, 3),
        'val_dp_rmse_cm': round(val_dp_cm, 2),
        'val_rot_rmse_deg': round(val_rot_deg, 3),
        'train_time_s': round(train_time, 1),
        'train_peak_vram_mb': round(train_peak_mb, 1),
        'lat_b1_p50_ms': round(lat_p50, 3), 'lat_b1_p95_ms': round(lat_p95, 3),
        'eval_peak_vram_mb': eval_peak_mb,
        'model_kw': model_kw,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seqs', default='200,400,800')
    ap.add_argument('--models', default='gru,tcn,transformer,zetaphi')
    ap.add_argument('--seeds', default='42,123,314')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--tag', default='run')
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f'{args.tag}.jsonl')

    seqs = [int(s) for s in args.seqs.split(',')]
    model_names = args.models.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]

    seen = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                seen.add((r['model'], r['seq_len'], r['seed']))

    total = len(seqs) * len(model_names) * len(seeds)
    done = 0
    for seq_len in seqs:
        for mname in model_names:
            for seed in seeds:
                done += 1
                if (mname, seq_len, seed) in seen:
                    print(f"[{done}/{total}] SKIP {mname} seq{seq_len} s{seed}", flush=True)
                    continue
                r = run_one(mname, seq_len, seed, args.device,
                            epochs=args.epochs, batch_size=args.batch)
                with open(out_path, 'a') as f:
                    f.write(json.dumps(r) + '\n')
                print(f"[{done}/{total}] {mname:<12} seq{seq_len:<5} s{seed:<4} | "
                      f"val {r['val_mse_n']:.4f} | test dp {r['test_dp_rmse_cm']:6.2f}cm "
                      f"rot {r['test_rot_rmse_deg']:6.3f}deg | "
                      f"{r['train_time_s']}s, {r['train_peak_vram_mb']}MB train | "
                      f"b1 {r['lat_b1_p50_ms']}ms", flush=True)

    print('ALL DONE', out_path, flush=True)


if __name__ == '__main__':
    main()
