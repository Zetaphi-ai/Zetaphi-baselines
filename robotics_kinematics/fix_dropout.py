import torch
import torch.nn as nn
from dataset import get_dataloaders
from models import SensorTransformer
from corruption_tests import apply_corruption

def forward_fill_filter(x):
    """
    Simulates a smart edge-sensor driver. 
    If a timestep reading is exactly 0.0 (a dropped packet), 
    it holds and carries over the value from the previous timestep.
    """
    x_out = x.clone()
    for t in range(1, x.shape[1]):
        mask = (x_out[:, t, :] == 0.0)
        x_out[:, t, :] = torch.where(mask, x_out[:, t-1, :], x_out[:, t, :])
    return x_out

@torch.no_grad()
def eval_corrupted_fixed(model, loader, device, c_type, apply_fix=False):
    model.eval()
    cor = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        if c_type != "clean":
            x = apply_corruption(x, c_type)
            
        if apply_fix:
            x = forward_fill_filter(x)
            
        out = model(x)
        cor += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return cor / total

def main():
    device = "cuda:0"
    train_dl, test_dl = get_dataloaders(batch_size=64)
    
    trans = SensorTransformer(d_model=128, n_layers=1).to(device)
    
    t_opt = torch.optim.AdamW(trans.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    print("Fast-training Baseline Transformer for 10 epochs...")
    for ep in range(10):
        trans.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            t_opt.zero_grad()
            t_loss = crit(trans(x), y)
            t_loss.backward()
            t_opt.step()
            
    print("Evaluating Random Timestep Dropout (20% packet loss)...")
    
    # Clean baseline
    t_clean = eval_corrupted_fixed(trans, test_dl, device, "clean", False)
    
    # Without fix
    t_drop = eval_corrupted_fixed(trans, test_dl, device, "random_timestep_dropout", False)
    
    # With fix
    t_fix = eval_corrupted_fixed(trans, test_dl, device, "random_timestep_dropout", True)
    
    print("\nRESULTS (20% Packet Loss):")
    print(f"Transformer (Clean Baseline):     {t_clean*100:.1f}%\n")
    print(f"Transformer (Unfiltered Zeros):   {t_drop*100:.1f}%")
    print(f"Transformer (Forward-Filled):     {t_fix*100:.1f}%\n")

if __name__ == '__main__':
    main()
