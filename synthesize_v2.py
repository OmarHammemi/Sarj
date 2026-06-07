"""Synthesize with mel clamping + stronger Griffin-Lim."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, soundfile as sf, torch, yaml
from data.audio_utils import mel_to_wav
from data.text_normalize import load_vocab, text_to_ids
from synthesize import load_model

ROOT = Path(__file__).resolve().parent

def clamp_mel(mel: np.ndarray, lo=-5.5, hi=1.5) -> np.ndarray:
    return np.clip(mel, lo, hi)

def synthesize(text, checkpoint, config_path, output, device_name="cuda"):
    cfg = yaml.safe_load(open(config_path))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model, ckpt_cfg = load_model(checkpoint, device)
    ids = text_to_ids(text, load_vocab(ROOT / "data/vocab.json"), add_eos=True)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)

    with torch.no_grad():
        mel = model.infer(tokens, lengths).cpu().numpy()

    mel = clamp_mel(mel)
    n_iters = cfg["synthesis"].get("griffin_lim_iters", 128)
    wav = mel_to_wav(mel, ckpt_cfg, n_iters)
    peak = np.max(np.abs(wav)) or 1.0
    wav = wav / peak * 0.95

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, wav, ckpt_cfg["audio"]["sample_rate"])
    print(f"Saved {output} ({len(wav)/ckpt_cfg['audio']['sample_rate']:.2f}s)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--config", default="configs/train_hesham.yaml")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    synthesize(a.text, Path(a.checkpoint), Path(a.config), Path(a.out), a.device)