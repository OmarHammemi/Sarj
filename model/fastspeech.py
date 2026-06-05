"""FastSpeech2-small acoustic model (<20M parameters)."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.decoder import MelDecoder, PostNet
from model.encoder import TextEncoder
from model.variance_adaptor import DurationPredictor, length_regulator


class FastSpeech2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        d_model: int = 256,
        n_heads: int = 4,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        ffn_dim: int = 1024,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_postnet: bool = True,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.use_postnet = use_postnet
        self.encoder = TextEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=encoder_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_len=max_seq_len,
        )
        self.duration_predictor = DurationPredictor(d_model, kernel_size, dropout)
        self.decoder = MelDecoder(d_model, n_mels, decoder_layers, kernel_size, dropout)
        self.postnet = PostNet(n_mels) if use_postnet else None

    def forward(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        mel_targets: torch.Tensor | None = None,
        duration_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(tokens, token_lengths)
        duration_pred = self.duration_predictor(encoded).clamp(min=0.0)
        durations = duration_targets if duration_targets is not None else duration_pred
        expanded, mel_lengths = length_regulator(encoded, durations)

        mel_pred = self.decoder(expanded, mel_lengths)
        outputs: dict[str, torch.Tensor] = {
            "mel_pred": mel_pred,
            "duration_pred": duration_pred,
            "mel_lengths": mel_lengths,
        }
        if self.postnet is not None:
            mel_post = self.postnet(mel_pred) + mel_pred
            outputs["mel_post"] = mel_post
        return outputs

    @torch.no_grad()
    def infer(self, tokens: torch.Tensor, token_lengths: torch.Tensor) -> torch.Tensor:
        self.eval()
        out = self.forward(tokens, token_lengths)
        if "mel_post" in out:
            mel = out["mel_post"]
        else:
            mel = out["mel_pred"]
        return mel[0, :, : out["mel_lengths"][0].item()]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: dict, vocab_size: int) -> FastSpeech2:
    mcfg = cfg["model"]
    model = FastSpeech2(
        vocab_size=vocab_size,
        n_mels=cfg["audio"]["n_mels"],
        d_model=mcfg["d_model"],
        n_heads=mcfg["n_heads"],
        encoder_layers=mcfg["encoder_layers"],
        decoder_layers=mcfg["decoder_layers"],
        ffn_dim=mcfg["ffn_dim"],
        kernel_size=mcfg["kernel_size"],
        dropout=mcfg["dropout"],
        use_postnet=mcfg.get("use_postnet", True),
        max_seq_len=mcfg.get("max_seq_len", 512),
    )
    n_params = count_parameters(model)
    if n_params >= 20_000_000:
        raise ValueError(f"Model has {n_params:,} parameters (must be < 20M)")
    print(f"Model parameters: {n_params:,}")
    return model
