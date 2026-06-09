import torch
import torch.nn as nn
from dataset import get_dataloaders
from models import SensorTransformer

def apply_corruption(x, c_type, intensity=1.0):
    """
    x shape: (B, 128, 9)
    Channels: 0-2 (Body Acc), 3-5 (Body Gyro), 6-8 (Total Acc)
    """
    x = x.clone()
    B, T, C = x.shape
    device = x.device
    
    if c_type == "gaussian_noise":
        noise = torch.randn_like(x) * 0.1 * intensity
        return x + noise
        
    elif c_type == "gyro_dropout":
        x[:, :, 3:6] = 0.0
        return x
        
    elif c_type == "accel_dropout":
        x[:, :, 0:3] = 0.0
        return x
        
    elif c_type == "random_timestep_dropout":
        mask = torch.rand(B, T, 1, device=device) > (0.2 * intensity)
        return x * mask.float()
        
    elif c_type == "spike_bursts":
        mask = torch.rand(B, T, C, device=device) < (0.05 * intensity)
        x[mask] += 5.0
        return x
        
    elif c_type == "time_shift":
        shift = int(5 * intensity)
        shifted_gyro = torch.roll(x[:, :, 3:6], shifts=shift, dims=1)
        shifted_gyro[:, :shift, :] = 0.0 
        x[:, :, 3:6] = shifted_gyro
        return x
        
    elif c_type == "calibration_drift":
        drift = torch.linspace(0, 0.5 * intensity, steps=T, device=device).unsqueeze(0).unsqueeze(-1)
        x[:, :, 3:6] += drift
        return x
        
    elif c_type == "quantization":
        levels = max(2, int(16 / intensity))
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        normed = (x - x_min) / (x_max - x_min + 1e-5)
        quantized = torch.round(normed * levels) / levels
        return quantized * (x_max - x_min + 1e-5) + x_min

    return x

@torch.no_grad()
def eval_corrupted(model, loader, device, c_type):
    model.eval()
    cor = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if c_type != "clean":
            x = apply_corruption(x, c_type)
            
        out = model(x)
        cor += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return cor / total

def main():
    device = "cuda:0"
    train_dl, test_dl = get_dataloaders(batch_size=64)
    
    # Baseline Transformer
    trans = SensorTransformer(d_model=128, n_layers=1).to(device)
    
    t_opt = torch.optim.AdamW(trans.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    print("Training Baseline Transformer for 10 epochs...")
    for ep in range(10):
        trans.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            t_opt.zero_grad()
            t_loss = crit(trans(x), y)
            t_loss.backward()
            t_opt.step()
            
    print("Training complete. Running Robustness Suite.\n")
    
    corruptions = [
        "clean",
        "gaussian_noise",
        "gyro_dropout",
        "accel_dropout",
        "random_timestep_dropout",
        "spike_bursts",
        "time_shift",
        "calibration_drift",
        "quantization"
    ]
    
    print(f"{'CORRUPTION':<25} | {'TRANSFORMER':<12}")
    print("-" * 40)
    
    for c in corruptions:
        t_acc = eval_corrupted(trans, test_dl, device, c)
        print(f"{c:<25} | {t_acc*100:>10.1f}%")

if __name__ == '__main__':
    main()
