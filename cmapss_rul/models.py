"""C-MAPSS baseline models — parameter-matched to a ~70k budget.

These are the public baselines our proprietary O(N) mixer is compared against.
A common pitfall when 'param-matching' a Transformer: nn.TransformerEncoderLayer
defaults to dim_feedforward=2048, which silently quadruples the budget. It is
overridden here (ffn=192).
"""
import torch
import torch.nn as nn


class SensorTransformer(nn.Module):
    def __init__(self, in_channels, d_model=48, n_heads=4, n_layers=2, dim_ff=192, max_len=256):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B, T, C = x.shape
        x = self.proj(x) + self.pos_embed[:, :T, :]
        x = self.transformer(x)
        return self.head(x.mean(dim=1))


class SensorGRU(nn.Module):
    def __init__(self, in_channels, hidden_dim=86, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class SensorTCN(nn.Module):
    """Causal dilated TCN. v1 used symmetric padding (acausal — peeks ahead);
    here padding is left-only so it is a fair streaming baseline."""

    def __init__(self, in_channels, hidden_dim=85, levels=(1, 2, 4, 8)):
        super().__init__()
        chans = [in_channels] + [hidden_dim] * len(levels)
        self.convs = nn.ModuleList([
            nn.Conv1d(chans[i], chans[i + 1], kernel_size=3, dilation=d)
            for i, d in enumerate(levels)
        ])
        self.pads = [2 * d for d in levels]
        self.relu = nn.ReLU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B,C,T)
        for conv, p in zip(self.convs, self.pads):
            x = self.relu(conv(nn.functional.pad(x, (p, 0))))
        return self.head(x.mean(dim=-1))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_model(name, in_channels, **kw):
    if name == 'gru':
        return SensorGRU(in_channels)
    if name == 'tcn':
        return SensorTCN(in_channels)
    if name == 'transformer':
        return SensorTransformer(in_channels)
    raise ValueError(name)


if __name__ == '__main__':
    c = 14
    for n in ['gru', 'tcn', 'transformer']:
        print(f"{n:<12} {count_params(build_model(n, c)):>8,}")
