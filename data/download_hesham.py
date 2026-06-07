"""Download + filter HeshamHaroon/arabic-msa-25k-saudi-male-tashkeel (~1 hour)."""
from __future__ import annotations
import argparse, io, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from datasets import load_dataset
from tqdm import tqdm

RAW_DIR = ROOT / "data" / "raw" / "wav"
DATASET_ID = "HeshamHaroon/arabic-msa-25k-saudi-male-tashkeel"

def is_arabic(text: str, min_ratio: float = 0.6) -> bool:
    text = text.strip()
    if not text:
        return False
    ar = len(re.findall(r"[\u0600-\u06FF]", text))
    return ar / max(len(text.replace(" ", "")), 1) >= min_ratio

def decode_audio(audio_obj) -> tuple[np.ndarray, int] | None:
    """Handle HF formats: {array,sr}, {bytes,path}, or raw bytes."""
    if audio_obj is None:
        return None

    # Standard decoded Audio feature
    if isinstance(audio_obj, dict):
        if "array" in audio_obj and audio_obj["array"] is not None:
            wav = np.asarray(audio_obj["array"], dtype=np.float32)
            sr = int(audio_obj.get("sampling_rate") or 22050)
            return wav, sr

        # Parquet binary format (this dataset)
        if audio_obj.get("bytes"):
            try:
                wav, sr = sf.read(io.BytesIO(audio_obj["bytes"]))
                return np.asarray(wav, dtype=np.float32), int(sr)
            except Exception:
                pass

        if audio_obj.get("path"):
            try:
                wav, sr = sf.read(audio_obj["path"])
                return np.asarray(wav, dtype=np.float32), int(sr)
            except Exception:
                pass

    if isinstance(audio_obj, (bytes, bytearray)):
        wav, sr = sf.read(io.BytesIO(audio_obj))
        return np.asarray(wav, dtype=np.float32), int(sr)

    return None

def download_filtered(
    output_csv: Path,
    max_hours: float = 1.0,
    min_dur: float = 2.0,
    max_dur: float = 12.0,
    min_chars: int = 10,
    max_chars: int = 180,
    target_sr: int = 22050,
):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    max_seconds = max_hours * 3600.0

    print(f"Streaming {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    rows, total_sec = [], 0.0
    skipped = {
        "duration": 0, "text": 0, "arabic": 0,
        "empty": 0, "no_audio": 0, "decode_fail": 0,
    }

    pbar = tqdm(desc="Filter+download")
    for item in ds:
        pbar.update(1)

        text = str(item.get("text") or item.get("text_stripped") or "").strip()
        dur_meta = float(item.get("audio_duration_s") or 0)

        if dur_meta and (dur_meta < min_dur or dur_meta > max_dur):
            skipped["duration"] += 1
            continue
        if len(text) < min_chars or len(text) > max_chars:
            skipped["text"] += 1
            continue
        if not is_arabic(text):
            skipped["arabic"] += 1
            continue

        decoded = decode_audio(item.get("audio"))
        if decoded is None:
            skipped["decode_fail"] += 1
            continue

        wav, sr_in = decoded
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if wav.size == 0:
            skipped["empty"] += 1
            continue

        if sr_in != target_sr:
            wav = librosa.resample(wav, orig_sr=sr_in, target_sr=target_sr)

        dur = len(wav) / target_sr
        if dur < min_dur or dur > max_dur:
            skipped["duration"] += 1
            continue

        if total_sec + dur > max_seconds and rows:
            break

        clip_id = f"hesham_{len(rows):05d}"
        out = RAW_DIR / f"{clip_id}.wav"
        sf.write(out, wav, target_sr)

        rows.append({
            "id": clip_id,
            "wav_path": str(out.relative_to(ROOT)),
            "text": text,
            "duration_sec": round(dur, 3),
            "source_dataset": DATASET_ID,
        })
        total_sec += dur
        if total_sec >= max_seconds:
            break

    pbar.close()

    if not rows:
        raise RuntimeError(f"No clips passed filters. Skipped: {skipped}")

    meta = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(output_csv, index=False)

    summary = {
        "clips": len(meta),
        "hours": round(meta["duration_sec"].sum() / 3600, 3),
        "avg_dur": round(meta["duration_sec"].mean(), 2),
        "skipped": skipped,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved -> {output_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(ROOT / "data" / "metadata.csv"))
    p.add_argument("--max-hours", type=float, default=1.0)
    p.add_argument("--min-dur", type=float, default=2.0)
    p.add_argument("--max-dur", type=float, default=12.0)
    a = p.parse_args()
    download_filtered(
        Path(a.output), max_hours=a.max_hours,
        min_dur=a.min_dur, max_dur=a.max_dur,
    )