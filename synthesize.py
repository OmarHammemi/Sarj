"""Synthesize speech from Arabic text."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from data.text_normalize import load_vocab, text_to_ids
from model.fastspeech import FastSpeech2

ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def griffin_lim(mel: np.ndarray, cfg: dict, n_iters: int) -> np.ndarray:
    import librosa

    sr = cfg["audio"]["sample_rate"]
    n_fft = cfg["audio"]["n_fft"]
    hop = cfg["audio"]["hop_length"]
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=cfg["audio"]["n_mels"],
        fmin=cfg["audio"]["fmin"],
        fmax=cfg["audio"]["fmax"],
    )
    mel_exp = np.exp(mel)
    spec = np.maximum(1e-10, np.dot(np.linalg.pinv(mel_basis), mel_exp))
    wav = librosa.griffinlim(
        spec,
        n_iter=n_iters,
        hop_length=hop,
        win_length=cfg["audio"]["win_length"],
        n_fft=n_fft,
    )
    return wav


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[FastSpeech2, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = FastSpeech2(
        vocab_size=ckpt["vocab_size"],
        n_mels=cfg["audio"]["n_mels"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        encoder_layers=cfg["model"]["encoder_layers"],
        decoder_layers=cfg["model"]["decoder_layers"],
        ffn_dim=cfg["model"]["ffn_dim"],
        kernel_size=cfg["model"]["kernel_size"],
        dropout=cfg["model"]["dropout"],
        use_postnet=cfg["model"].get("use_postnet", True),
        max_seq_len=cfg["model"].get("max_seq_len", 512),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, cfg


def synthesize(
    text: str,
    checkpoint: Path,
    config_path: Path,
    output: Path,
    device_name: str = "cpu",
) -> Path:
    cfg = load_config(config_path)
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    vocab = load_vocab(ROOT / "data" / "vocab.json")
    model, ckpt_cfg = load_model(checkpoint, device)

    ids = text_to_ids(text, vocab, add_eos=True)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)

    with torch.no_grad():
        mel = model.infer(tokens, lengths).cpu().numpy()

    wav = griffin_lim(mel, ckpt_cfg, cfg["synthesis"]["griffin_lim_iters"])
    peak = np.max(np.abs(wav)) if wav.size else 1.0
    if peak > 0:
        wav = wav / peak * 0.95

    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, wav, ckpt_cfg["audio"]["sample_rate"])
    print(f"Saved {output} ({len(wav) / ckpt_cfg['audio']['sample_rate']:.2f}s)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize Arabic speech")
    parser.add_argument("--text", required=True, help="Arabic input text")
    parser.add_argument("--out", required=True, help="Output WAV path")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "best.pt"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    args = parser.parse_args()
    synthesize(args.text, Path(args.checkpoint), Path(args.config), Path(args.out), args.device)


if __name__ == "__main__":
    main()
