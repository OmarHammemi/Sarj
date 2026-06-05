"""Preprocess audio, extract log-mels, build train/val splits and durations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import yaml
from tqdm import tqdm

from data.text_normalize import build_vocab, normalize_arabic, save_vocab


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def mel_spectrogram(wav: np.ndarray, cfg: dict) -> np.ndarray:
    sr = cfg["audio"]["sample_rate"]
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=cfg["audio"]["n_fft"],
        win_length=cfg["audio"]["win_length"],
        hop_length=cfg["audio"]["hop_length"],
        n_mels=cfg["audio"]["n_mels"],
        f_min=cfg["audio"]["fmin"],
        f_max=cfg["audio"]["fmax"],
    )
    wav_t = torch.from_numpy(wav).float().unsqueeze(0)
    mel = mel_fn(wav_t)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel.squeeze(0).numpy()


def char_durations(num_chars: int, mel_frames: int) -> np.ndarray:
    """Assign mel frames to characters proportionally (bootstrap alignments)."""
    if num_chars <= 0:
        return np.array([], dtype=np.int64)
    base = mel_frames // num_chars
    rem = mel_frames - base * num_chars
    durations = np.full(num_chars, base, dtype=np.int64)
    durations[:rem] += 1
    durations = np.maximum(durations, 1)
    diff = int(durations.sum()) - mel_frames
    if diff > 0:
        for i in range(num_chars - 1, -1, -1):
            if diff == 0:
                break
            if durations[i] > 1:
                durations[i] -= 1
                diff -= 1
    elif diff < 0:
        for i in range(num_chars):
            if diff == 0:
                break
            durations[i] += 1
            diff += 1
    return durations


def preprocess(config_path: Path, metadata_path: Path) -> None:
    cfg = load_config(config_path)
    audio_cfg = cfg["audio"]
    sr = audio_cfg["sample_rate"]

    proc_wav = ROOT / "data" / "processed" / "wav"
    proc_mel = ROOT / "data" / "processed" / "mel"
    dur_dir = ROOT / "data" / "durations"
    for d in (proc_wav, proc_mel, dur_dir):
        d.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(metadata_path)
    rows = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Preprocessing"):
        wav_path = ROOT / row["wav_path"]
        wav, _ = librosa.load(wav_path, sr=sr, mono=True)
        wav, _ = librosa.effects.trim(wav, top_db=audio_cfg["trim_top_db"])
        if len(wav) == 0:
            continue

        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = wav / peak * 0.95

        duration = len(wav) / sr
        if duration < audio_cfg["min_duration"] or duration > audio_cfg["max_duration"]:
            continue

        text = normalize_arabic(str(row["text"]))
        if not text:
            continue

        clip_id = row["id"]
        out_wav = proc_wav / f"{clip_id}.wav"
        out_mel = proc_mel / f"{clip_id}.npy"
        out_dur = dur_dir / f"{clip_id}.npy"

        sf.write(out_wav, wav, sr)
        mel = mel_spectrogram(wav, cfg)
        np.save(out_mel, mel.astype(np.float32))
        np.save(out_dur, char_durations(len(text), mel.shape[1]))

        rows.append(
            {
                "id": clip_id,
                "wav_path": str(out_wav.relative_to(ROOT)),
                "mel_path": str(out_mel.relative_to(ROOT)),
                "duration_path": str(out_dur.relative_to(ROOT)),
                "text": text,
                "duration_sec": round(duration, 3),
                "mel_frames": mel.shape[1],
                "num_chars": len(text),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No clips survived preprocessing.")

    seed = cfg["dataset"]["seed"]
    train_frac = cfg["dataset"]["train_split"]
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * train_frac)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)

    train_df.to_csv(ROOT / "data" / "train.csv", index=False)
    val_df.to_csv(ROOT / "data" / "val.csv", index=False)

    vocab = build_vocab(train_df["text"].tolist())
    save_vocab(vocab, ROOT / "data" / "vocab.json")

    summary = {
        "clips": len(df),
        "train": len(train_df),
        "val": len(val_df),
        "hours": round(df["duration_sec"].sum() / 3600.0, 3),
        "vocab_size": len(vocab),
    }
    with (ROOT / "data" / "preprocess_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Arabic TTS dataset")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--metadata", default=str(ROOT / "data" / "metadata.csv"))
    args = parser.parse_args()
    preprocess(Path(args.config), Path(args.metadata))


if __name__ == "__main__":
    main()
