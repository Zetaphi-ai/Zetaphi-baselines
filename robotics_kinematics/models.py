import torch
import torch.nn as nn

# Baseline Model (Standard 1D Transformer)
class SensorTransformer(nn.Module):
    def __init__(self, in_channels=9, d_model=128, n_heads=4, n_layers=1, num_classes=6):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 128, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = self.proj(x) + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

if __name__ == "__main__":
    t = SensorTransformer(d_model=128, n_layers=1)
    print(f"Transformer Params: {count_params(t):,}")
