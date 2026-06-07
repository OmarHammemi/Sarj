from __future__ import annotations
import torch, torch.nn as nn

class ConvNormBlock(nn.Module):
    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x).transpose(1, 2)
        x = self.drop(self.act(self.norm(x)))
        return x.transpose(1, 2)

class DurationPredictor(nn.Module):
    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__()
        self.block1 = ConvNormBlock(d_model, kernel_size, dropout)
        self.block2 = ConvNormBlock(d_model, kernel_size, dropout)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.block2(self.block1(x.transpose(1, 2))).transpose(1, 2)
        return self.proj(h).squeeze(-1)

def length_regulator(encoded, durations):
    outs, lens = [], []
    for b in range(encoded.size(0)):
        reps = []
        for t in range(encoded.size(1)):
            d = int(torch.round(durations[b, t]).clamp(min=0).item())
            if d > 0:
                reps.append(encoded[b, t].unsqueeze(0).repeat(d, 1))
        o = torch.cat(reps, dim=0) if reps else encoded.new_zeros((1, encoded.size(-1)))
        outs.append(o)
        lens.append(o.size(0))
    ml = max(lens)
    pad = encoded.new_zeros(encoded.size(0), ml, encoded.size(-1))
    for b, o in enumerate(outs):
        pad[b, : o.size(0)] = o
    return pad, torch.tensor(lens, device=encoded.device, dtype=torch.long)