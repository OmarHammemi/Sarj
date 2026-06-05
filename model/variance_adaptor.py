"""Duration predictor and length regulator."""

from __future__ import annotations

import torch
import torch.nn as nn


class DurationPredictor(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=padding),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size, padding=padding),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.proj(h).squeeze(-1)


def length_regulator(encoded: torch.Tensor, durations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand encoder outputs according to predicted/target durations."""
    batch_outputs = []
    mel_lengths = []
    for b in range(encoded.size(0)):
        reps = []
        for t in range(encoded.size(1)):
            dur = int(torch.round(durations[b, t]).clamp(min=0).item())
            if dur > 0:
                reps.append(encoded[b, t].unsqueeze(0).repeat(dur, 1))
        if reps:
            out = torch.cat(reps, dim=0)
        else:
            out = encoded.new_zeros((1, encoded.size(-1)))
        batch_outputs.append(out)
        mel_lengths.append(out.size(0))

    max_len = max(mel_lengths)
    d_model = encoded.size(-1)
    padded = encoded.new_zeros((encoded.size(0), max_len, d_model))
    for b, out in enumerate(batch_outputs):
        padded[b, : out.size(0)] = out
    return padded, torch.tensor(mel_lengths, device=encoded.device, dtype=torch.long)
