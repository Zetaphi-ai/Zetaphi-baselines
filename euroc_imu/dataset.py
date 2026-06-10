"""EuRoC v2 — IMU-only relative-odometry dataset (the honest formulation).

Task: given a window of raw 200Hz IMU (gyro+accel, 6ch), predict the relative
motion over that window:
  - dp_body (3): end-position minus start-position, rotated into the body
    frame at window start (makes the target frame-invariant / learnable)
  - rotvec (3): rotation vector of R_start^T @ R_end (relative orientation)

Split: cross-SEQUENCE (no sliding-window leakage):
  train: MH_01_easy, MH_02_easy, MH_04_difficult
  val:   MH_03_medium
  test:  MH_05_difficult

IMU normalized with train-set per-channel mean/std. Targets standardized with
train-set per-dim std (kept in meta.json so metrics report in physical units).
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

SPLITS = {
    'train': ['MH_01_easy', 'MH_02_easy', 'MH_04_difficult'],
    'val': ['MH_03_medium'],
    'test': ['MH_05_difficult'],
}


def load_sequence(seq_name):
    """Return (imu (N,6) float32, pos (N,3), quat_wxyz (N,4)) synced at IMU rate (200Hz)."""
    mav = os.path.join(DATA, seq_name, 'mav0')
    imu = pd.read_csv(os.path.join(mav, 'imu0', 'data.csv'))
    imu.columns = ['t', 'wx', 'wy', 'wz', 'ax', 'ay', 'az']
    gt = pd.read_csv(os.path.join(mav, 'state_groundtruth_estimate0', 'data.csv'))
    gt = gt.iloc[:, :8]
    gt.columns = ['t', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz']
    imu, gt = imu.sort_values('t'), gt.sort_values('t')
    # GT estimate is at 200Hz roughly aligned with IMU; asof-match each IMU stamp
    m = pd.merge_asof(imu, gt, on='t', direction='nearest',
                      tolerance=int(5e6))  # 5ms tolerance (ns timestamps)
    m = m.dropna().reset_index(drop=True)
    imu_arr = m[['wx', 'wy', 'wz', 'ax', 'ay', 'az']].to_numpy(np.float32)
    pos = m[['px', 'py', 'pz']].to_numpy(np.float64)
    quat = m[['qw', 'qx', 'qy', 'qz']].to_numpy(np.float64)
    return imu_arr, pos, quat


def make_windows(imu, pos, quat, seq_len, stride):
    """Vectorized window extraction + relative-pose targets."""
    n = len(imu)
    starts = np.arange(0, n - seq_len, stride)
    ends = starts + seq_len - 1
    # scipy wants xyzw
    q_xyzw = quat[:, [1, 2, 3, 0]]
    R_start = Rotation.from_quat(q_xyzw[starts])
    R_end = Rotation.from_quat(q_xyzw[ends])
    dp_world = pos[ends] - pos[starts]
    dp_body = R_start.inv().apply(dp_world)                     # (W,3)
    rotvec = (R_start.inv() * R_end).as_rotvec()                # (W,3)
    targets = np.concatenate([dp_body, rotvec], axis=1).astype(np.float32)
    return starts, targets


class EurocIMUWindows(Dataset):
    def __init__(self, seqs, seq_len, stride, imu_stats=None, tgt_std=None):
        xs, starts_all, tgts = [], [], []
        offset = 0
        self.imu_cat = []
        for s in seqs:
            imu, pos, quat = load_sequence(s)
            starts, targets = make_windows(imu, pos, quat, seq_len, stride)
            self.imu_cat.append(imu)
            starts_all.append(starts + offset)
            tgts.append(targets)
            offset += len(imu)
        self.imu = np.concatenate(self.imu_cat, axis=0)
        self.starts = np.concatenate(starts_all)
        self.targets = np.concatenate(tgts, axis=0)
        self.seq_len = seq_len

        if imu_stats is None:
            imu_stats = (self.imu.mean(0, keepdims=True), self.imu.std(0, keepdims=True) + 1e-8)
        self.imu_mean, self.imu_std = imu_stats
        if tgt_std is None:
            tgt_std = self.targets.std(0, keepdims=True) + 1e-8
        self.tgt_std = tgt_std

        self.imu = (self.imu - self.imu_mean) / self.imu_std
        self.targets_n = self.targets / self.tgt_std

    def stats(self):
        return (self.imu_mean, self.imu_std), self.tgt_std

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        x = torch.from_numpy(self.imu[s:s + self.seq_len])
        y = torch.from_numpy(self.targets_n[i])
        return x, y


def get_dataloaders(seq_len=400, stride=20, batch_size=64, num_workers=4):
    train = EurocIMUWindows(SPLITS['train'], seq_len, stride)
    imu_stats, tgt_std = train.stats()
    val = EurocIMUWindows(SPLITS['val'], seq_len, stride, imu_stats, tgt_std)
    test = EurocIMUWindows(SPLITS['test'], seq_len, stride, imu_stats, tgt_std)
    meta = {
        'seq_len': seq_len, 'stride': stride,
        'tgt_std': tgt_std.squeeze().tolist(),
        'imu_mean': imu_stats[0].squeeze().tolist(),
        'imu_std': imu_stats[1].squeeze().tolist(),
        'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
    }
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers, drop_last=sh,
                                   pin_memory=True)
    return mk(train, True), mk(val, False), mk(test, False), meta


if __name__ == '__main__':
    tr, va, te, meta = get_dataloaders()
    print(json.dumps(meta, indent=2))
    x, y = next(iter(tr))
    print('x', x.shape, 'y', y.shape)
