"""C-MAPSS loader — protocol notes:

1. NO survivor bias in the history ladder: test units shorter than seq_len are
   PADDED by replicating their earliest cycle (zero-order-hold backwards in
   time), so ALL test units are scored at every history length.
2. Unit-level validation split (20% of train units) for model selection and
   tuning. The official test set is touched exactly once per final model.
3. Normalization stats computed on the train-split units only (no val/test
   leakage into min/max).
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SPLIT_SEED = 1337  # fixed across all experiments so every model sees the same split


def load_fd(data_dir, fd_num='001', seq_len=30, max_rul=125, val_frac=0.2):
    cols = ['unit', 'cycle', 'set1', 'set2', 'set3'] + [f's{i}' for i in range(1, 22)]
    train_df = pd.read_csv(os.path.join(data_dir, f'train_FD{fd_num}.txt'), sep=r'\s+', header=None, names=cols)
    test_df = pd.read_csv(os.path.join(data_dir, f'test_FD{fd_num}.txt'), sep=r'\s+', header=None, names=cols)
    rul_df = pd.read_csv(os.path.join(data_dir, f'RUL_FD{fd_num}.txt'), sep=r'\s+', header=None, names=['RUL'])

    # piecewise-linear RUL target, clipped at max_rul (standard recipe)
    mx = train_df.groupby('unit')['cycle'].max().rename('max_cycle').reset_index()
    train_df = train_df.merge(mx, on='unit')
    train_df['RUL'] = (train_df['max_cycle'] - train_df['cycle']).clip(upper=max_rul)
    train_df.drop(columns='max_cycle', inplace=True)

    # unit-level train/val split (fixed seed, independent of model seed)
    units = np.array(sorted(train_df['unit'].unique()))
    rng = np.random.RandomState(SPLIT_SEED)
    rng.shuffle(units)
    n_val = max(1, int(len(units) * val_frac))
    val_units = set(units[:n_val].tolist())
    tr_units = set(units[n_val:].tolist())

    feature_cols = ['set1', 'set2', 'set3'] + [f's{i}' for i in range(1, 22)]
    tr_mask = train_df['unit'].isin(tr_units)
    std = train_df.loc[tr_mask, feature_cols].std()
    feature_cols = std[std > 1e-4].index.tolist()

    mn = train_df.loc[tr_mask, feature_cols].min()
    mxv = train_df.loc[tr_mask, feature_cols].max()
    for df in (train_df, test_df):
        df[feature_cols] = (df[feature_cols] - mn) / (mxv - mn + 1e-7)

    def sliding(df, unit_ids):
        X, Y = [], []
        for unit in unit_ids:
            ud = df[df['unit'] == unit]
            dx = ud[feature_cols].values.astype(np.float32)
            dy = ud['RUL'].values.astype(np.float32)
            if len(dx) < seq_len:  # pad train/val units too (rare, but symmetric)
                pad = np.repeat(dx[:1], seq_len - len(dx), axis=0)
                dx = np.concatenate([pad, dx], 0)
                dy = np.concatenate([np.full(seq_len - len(dy), dy[0], np.float32), dy])
            for i in range(len(dx) - seq_len + 1):
                X.append(dx[i:i + seq_len])
                Y.append(dy[i + seq_len - 1])
        return np.stack(X), np.array(Y, np.float32)

    X_tr, Y_tr = sliding(train_df, sorted(tr_units))
    X_va, Y_va = sliding(train_df, sorted(val_units))

    # test: one window per unit (last seq_len cycles), ZOH-pad short units
    X_te, Y_te = [], []
    for unit in sorted(test_df['unit'].unique()):
        dx = test_df[test_df['unit'] == unit][feature_cols].values.astype(np.float32)
        if len(dx) < seq_len:
            pad = np.repeat(dx[:1], seq_len - len(dx), axis=0)
            dx = np.concatenate([pad, dx], 0)
        X_te.append(dx[-seq_len:])
        Y_te.append(min(max_rul, float(rul_df.iloc[int(unit) - 1].item())))
    X_te = np.stack(X_te)
    Y_te = np.array(Y_te, np.float32)

    return (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te), len(feature_cols)


class WindowDS(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def get_dataloaders(fd_num='001', seq_len=30, batch_size=256, data_dir=DATA_DIR):
    tr, va, te, c = load_fd(data_dir, fd_num, seq_len)
    return (
        DataLoader(WindowDS(*tr), batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(WindowDS(*va), batch_size=batch_size, shuffle=False),
        DataLoader(WindowDS(*te), batch_size=batch_size, shuffle=False),
        c,
    )


if __name__ == '__main__':
    for seq in (30, 50, 100, 150, 200):
        tr, va, te, c = load_fd(DATA_DIR, '001', seq)
        print(f"seq={seq:>3} | train {tr[0].shape} val {va[0].shape} test {te[0].shape} ch={c}")
        assert te[0].shape[0] == 100, "history ladder must keep all 100 test units"
    print("OK: all 100 test units present at every history length")
