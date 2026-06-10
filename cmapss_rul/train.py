"""C-MAPSS baseline training harness.

Protocol (honest-card rules):
  - model selection: best VALIDATION RMSE epoch -> that checkpoint is evaluated
    on the official test set exactly once. No best-epoch-on-test peeking.
  - 3 seeds per (model, history) cell.
  - NASA asymmetric score reported alongside RMSE/MAE.

Usage:
  python train.py --fd 001 --seqs 30,50,100,150,200 --models gru,tcn,transformer \
                  --seeds 42,123,314 --device cuda:0 --tag maingrid
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders
from models import build_model, count_params

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def nasa_score(pred, true):
    """Official C-MAPSS asymmetric score: late predictions punished harder."""
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    preds, trues = [], []
    for x, y in dl:
        out = model(x.to(device))
        preds.append(out.float().cpu().numpy())
        trues.append(y.numpy())
    p = np.concatenate(preds).ravel()
    t = np.concatenate(trues).ravel()
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae = float(np.mean(np.abs(p - t)))
    return rmse, mae, nasa_score(p, t)


def run_one(model_name, fd, seq_len, seed, device, epochs=60, lr=1e-3, wd=1e-4,
            save_ckpt=True):
    set_seed(seed)
    train_dl, val_dl, test_dl, in_ch = get_dataloaders(fd_num=fd, seq_len=seq_len)
    model = build_model(model_name, in_ch).to(device)
    n_params = count_params(model)

    run_model = model

    opt = torch.optim.AdamW(run_model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    t0 = time.time()
    for ep in range(epochs):
        run_model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(run_model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(run_model.parameters(), 1.0)
            opt.step()
        sched.step()
        val_rmse, _, _ = evaluate(run_model, val_dl, device)
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = copy.deepcopy(model.state_dict())
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    rmse, mae, score = evaluate(model, test_dl, device)

    if save_ckpt:
        ck = os.path.join(os.path.dirname(RESULTS_DIR), 'checkpoints',
                          f'{model_name}_fd{fd}_seq{seq_len}_s{seed}.pt')
        torch.save(best_state, ck)

    return {
        'model': model_name, 'fd': fd, 'seq_len': seq_len, 'seed': seed,
        'params': n_params, 'val_rmse': best_val, 'test_rmse': rmse,
        'test_mae': mae, 'nasa_score': score, 'train_time_s': round(train_time, 1),
        'epochs': epochs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fd', default='001')
    ap.add_argument('--seqs', default='30,50,100,150,200')
    ap.add_argument('--models', default='gru,tcn,transformer')
    ap.add_argument('--seeds', default='42,123,314')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--tag', default='run')
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f'{args.tag}_fd{args.fd}.jsonl')

    seqs = [int(s) for s in args.seqs.split(',')]
    model_names = args.models.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]

    total = len(seqs) * len(model_names) * len(seeds)
    done = 0
    # resume support: skip cells already in the jsonl
    seen = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                seen.add((r['model'], r['seq_len'], r['seed']))

    for seq_len in seqs:
        for mname in model_names:
            for seed in seeds:
                done += 1
                if (mname, seq_len, seed) in seen:
                    print(f"[{done}/{total}] SKIP {mname} seq{seq_len} s{seed} (already done)", flush=True)
                    continue
                r = run_one(mname, args.fd, seq_len, seed, args.device, epochs=args.epochs)
                with open(out_path, 'a') as f:
                    f.write(json.dumps(r) + '\n')
                print(f"[{done}/{total}] {mname:<12} seq{seq_len:<4} s{seed:<5} | "
                      f"val {r['val_rmse']:6.2f} | test RMSE {r['test_rmse']:6.2f} "
                      f"MAE {r['test_mae']:6.2f} NASA {r['nasa_score']:8.1f} | {r['train_time_s']}s",
                      flush=True)

    print('ALL DONE', out_path, flush=True)


if __name__ == '__main__':
    main()
