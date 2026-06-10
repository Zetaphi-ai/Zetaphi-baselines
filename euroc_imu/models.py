"""EuRoC IMU odometry — parameter-matched BASELINES (~69k params).

ZetaPhi model code is withheld pending IP filing; the harness, baselines,
protocol, and all measured numbers (including ZetaPhi's) are published.
"""
import torch
import torch.nn as nn

OUT_DIM = 6  # dp_body (3) + rotvec (3)


def sinusoidal_pe(max_len, d_model):
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float()
                    * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class SensorTransformer(nn.Module):
    """Sinusoidal PE (param-free) — a learned 4096-len pos_embed alone would be
    ~196k params, 3x the whole budget, and is the wrong prior for long streams."""

    def __init__(self, in_channels=6, d_model=56, n_heads=4, n_layers=2,
                 dim_ff=192, max_len=4096):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.register_buffer('pos_embed', sinusoidal_pe(max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, OUT_DIM)

    def forward(self, x):
        B, T, C = x.shape
        x = self.proj(x) + self.pos_embed[:, :T, :]
        x = self.transformer(x)
        return self.head(x.mean(dim=1))


class SensorGRU(nn.Module):
    def __init__(self, in_channels=6, hidden_dim=86, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden_dim, num_layers=num_layers,
                          batch_first=True)
        self.head = nn.Linear(hidden_dim, OUT_DIM)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class SensorTCN(nn.Module):
    """Causal dilated TCN (left-pad only — fair streaming baseline)."""

    def __init__(self, in_channels=6, hidden_dim=67, levels=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        chans = [in_channels] + [hidden_dim] * len(levels)
        self.convs = nn.ModuleList([
            nn.Conv1d(chans[i], chans[i + 1], kernel_size=3, dilation=d)
            for i, d in enumerate(levels)
        ])
        self.pads = [2 * d for d in levels]
        self.relu = nn.ReLU()
        self.head = nn.Linear(hidden_dim, OUT_DIM)

    def forward(self, x):
        x = x.transpose(1, 2)
        for conv, p in zip(self.convs, self.pads):
            x = self.relu(conv(nn.functional.pad(x, (p, 0))))
        return self.head(x.mean(dim=-1))



def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_model(name, in_channels=6, **kw):
    if name == 'gru':
        return SensorGRU(in_channels, **kw)
    if name == 'tcn':
        return SensorTCN(in_channels, **kw)
    if name == 'transformer':
        return SensorTransformer(in_channels, **kw)
    if name.startswith('zetaphi'):
        raise NotImplementedError(
            "ZetaPhi model code is withheld pending IP filing. "
            "All ZetaPhi numbers in results/ were produced by this exact "
            "harness with the proprietary model plugged into build_model.")
    raise ValueError(name)


if __name__ == '__main__':
    for n in ['gru', 'tcn', 'transformer']:
        print(f"{n:<12} {count_params(build_model(n)):>8,}")
