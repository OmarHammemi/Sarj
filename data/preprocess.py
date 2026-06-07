from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import librosa, numpy as np, pandas as pd, soundfile as sf, yaml
from tqdm import tqdm
from data.audio_utils import mel_spectrogram
from data.text_normalize import build_vocab, normalize_arabic, save_vocab

def char_durations(n, mel_frames):
    if n <= 0:
        return np.array([], dtype=np.int64)
    base, rem = mel_frames // n, mel_frames - (mel_frames // n) * n
    d = np.full(n, base, dtype=np.int64)
    d[:rem] += 1
    d = np.maximum(d, 1)
    diff = int(d.sum()) - mel_frames
    i = 0
    while diff > 0 and i < n:
        if d[n - 1 - i] > 1:
            d[n - 1 - i] -= 1
            diff -= 1
        i += 1
    while diff < 0:
        d[diff % n] += 1
        diff += 1
    return d

def preprocess(config_path, metadata_path):
    cfg = yaml.safe_load(open(config_path))
    a, sr = cfg["audio"], cfg["audio"]["sample_rate"]
    pw = ROOT / "data" / "processed" / "wav"
    pm = ROOT / "data" / "processed" / "mel"
    dd = ROOT / "data" / "durations"
    for d in (pw, pm, dd):
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in tqdm(pd.read_csv(metadata_path).iterrows(), desc="Preprocess"):
        wav, _ = librosa.load(ROOT / row["wav_path"], sr=sr, mono=True)
        wav, _ = librosa.effects.trim(wav, top_db=a["trim_top_db"])
        if len(wav) == 0:
            continue
        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = wav / peak * 0.95
        dur = len(wav) / sr
        if dur < a["min_duration"] or dur > a["max_duration"]:
            continue
        text = normalize_arabic(str(row["text"]))
        if not text:
            continue
        cid = row["id"]
        sf.write(pw / f"{cid}.wav", wav, sr)
        mel = mel_spectrogram(wav, cfg)
        np.save(pm / f"{cid}.npy", mel)
        np.save(dd / f"{cid}.npy", char_durations(len(text), mel.shape[1]))
        rows.append({
            "id": cid,
            "wav_path": str((pw / f"{cid}.wav").relative_to(ROOT)),
            "mel_path": str((pm / f"{cid}.npy").relative_to(ROOT)),
            "duration_path": str((dd / f"{cid}.npy").relative_to(ROOT)),
            "text": text,
            "duration_sec": round(dur, 3),
            "mel_frames": mel.shape[1],
            "num_chars": len(text),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No clips survived preprocessing.")

    df = df.sample(frac=1, random_state=cfg["dataset"]["seed"]).reset_index(drop=True)
    s = int(len(df) * cfg["dataset"]["train_split"])
    df.iloc[:s].to_csv(ROOT / "data" / "train.csv", index=False)
    df.iloc[s:].to_csv(ROOT / "data" / "val.csv", index=False)
    save_vocab(build_vocab(df.iloc[:s]["text"].tolist()), ROOT / "data" / "vocab.json")
    print(json.dumps({
        "clips": len(df), "train": s, "val": len(df) - s,
        "hours": round(df["duration_sec"].sum() / 3600, 3),
        "vocab_size": len(json.load(open(ROOT / "data" / "vocab.json"))),
    }, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--metadata", default=str(ROOT / "data" / "metadata.csv"))
    a = p.parse_args()
    preprocess(Path(a.config), Path(a.metadata))