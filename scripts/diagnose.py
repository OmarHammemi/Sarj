"""Compare GT mel vs model mel vs original wav."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, soundfile as sf, yaml, torch
from data.audio_utils import mel_to_wav
from data.text_normalize import load_vocab, text_to_ids
from synthesize import load_model

def diagnose(config_path, checkpoint, row_idx=0, out_dir="samples/diag"):
    cfg = yaml.safe_load(open(config_path))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sr = cfg["audio"]["sample_rate"]
    n_iters = cfg["synthesis"].get("griffin_lim_iters", 64)

    row = pd.read_csv(ROOT / "data/train.csv").iloc[row_idx]
    text = row["text"]
    gt_mel = np.load(ROOT / row["mel_path"])

    def save_mel(mel, name):
        wav = mel_to_wav(mel, cfg, n_iters)
        peak = np.max(np.abs(wav)) or 1.0
        wav = wav / peak * 0.95
        path = out / name
        sf.write(path, wav, sr)
        return path

    p_a = save_mel(gt_mel, "A_gt_mel.wav")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(Path(checkpoint), device)
    ids = text_to_ids(text, load_vocab(ROOT / "data/vocab.json"), add_eos=True)
    tokens = torch.tensor([ids], device=device)
    lengths = torch.tensor([len(ids)], device=device)
    with torch.no_grad():
        pred = model.infer(tokens, lengths).cpu().numpy()

    p_b = save_mel(pred, "B_model_same_text.wav")

    orig = ROOT / row["wav_path"]
    p_c = out / "C_original.wav"
    if orig.exists():
        sf.write(p_c, *sf.read(str(orig)))

    t = min(pred.shape[1], gt_mel.shape[1])
    l1 = float(np.mean(np.abs(pred[:, :t] - gt_mel[:, :t])))

    print("Text:", text[:100])
    print(f"Mel L1 (pred vs gt): {l1:.3f}  (lower = better, aim < 0.5)")
    print("A:", p_a, "— vocoder ceiling")
    print("B:", p_b, "— model on same text")
    print("C:", p_c, "— studio target")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_hesham.yaml")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--row", type=int, default=0)
    args = p.parse_args()
    diagnose(args.config, args.checkpoint, args.row)