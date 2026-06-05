"""Mel decoder and PostNet."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MelDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_mels: int = 80,
        n_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [ConvBlock(d_model, kernel_size, dropout) for _ in range(n_layers)]
        )
        self.proj = nn.Conv1d(d_model, n_mels, kernel_size=1)

    def forward(self, x: torch.Tensor, mel_lengths: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        h = x.transpose(1, 2)
        for layer in self.layers:
            h = layer(h)
        mel = self.proj(h)
        return mel


class PostNet(nn.Module):
    def __init__(self, n_mels: int = 80, channels: int = 256, n_layers: int = 3, kernel_size: int = 5):
        super().__init__()
        layers = []
        in_ch = n_mels
        for i in range(n_layers):
            out_ch = channels if i < n_layers - 1 else n_mels
            padding = kernel_size // 2
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
                    nn.BatchNorm1d(out_ch),
                    nn.Tanh() if i < n_layers - 1 else nn.Identity(),
                    nn.Dropout(0.1) if i < n_layers - 1 else nn.Identity(),
                ]
            )
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.net(mel)
