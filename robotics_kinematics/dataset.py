import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class UCIHARDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.data_path = os.path.join(data_path, split)
        
        # Load labels (0-indexed)
        y_path = os.path.join(self.data_path, f'y_{split}.txt')
        self.y = torch.tensor(np.loadtxt(y_path, dtype=np.int64) - 1)
        
        # Load 9 channels of inertial sensor data
        # shape: (samples, 128, 9)
        signals = [
            f'body_acc_x_{split}.txt', f'body_acc_y_{split}.txt', f'body_acc_z_{split}.txt',
            f'body_gyro_x_{split}.txt', f'body_gyro_y_{split}.txt', f'body_gyro_z_{split}.txt',
            f'total_acc_x_{split}.txt', f'total_acc_y_{split}.txt', f'total_acc_z_{split}.txt'
        ]
        
        loaded_signals = []
        for sig in signals:
            path = os.path.join(self.data_path, 'Inertial Signals', sig)
            loaded_signals.append(np.loadtxt(path, dtype=np.float32))
            
        # Stack to (samples, seq_len, channels) -> (N, 128, 9)
        self.X = torch.tensor(np.stack(loaded_signals, axis=-1))
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(data_path='/home/truthtree/.hermes/workspace/uci_har_zetaphi/data/UCI HAR Dataset', batch_size=64):
    train_ds = UCIHARDataset(data_path, split='train')
    test_ds = UCIHARDataset(data_path, split='test')
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_dl, test_dl

if __name__ == '__main__':
    train_dl, test_dl = get_dataloaders()
    for x, y in train_dl:
        print(f"Batch X shape: {x.shape} (Batch, SeqLen, Channels)")
        print(f"Batch Y shape: {y.shape}")
        break
