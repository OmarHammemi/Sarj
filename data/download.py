"""Download ~1 hour of single-speaker Saudi Arabic speech from Hugging Face."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "wav"

# Datasets visible on the task's HF search page:
# https://huggingface.co/datasets?modality=modality:audio&sort=trending&search=saudi
RECOMMENDED_DATASETS = {
    "saudi-podcast-1hr": "AhmedAshrafMarzouk/saudi-podcast-1hr",
    "hesham-saudi-male-tashkeel": "HeshamHaroon/arabic-msa-25k-saudi-male-tashkeel",
    "arabic-tts-saudi": "AhmedAshrafMarzouk/arabic-tts-saudi-audio-dataset",
}

DEFAULT_DATASET = RECOMMENDED_DATASETS["saudi-podcast-1hr"]


def find_text(item: dict) -> str | None:
    for key in (
        "text",
        "transcription",
        "transcript",
        "sentence",
        "content",
        "arabic",
        "arabic_text",
        "label",
    ):
        value = item.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return None


def clip_id_from_item(item: dict, fallback: str) -> str:
    for key in ("id", "file", "filename", "file_name", "audio_id"):
        value = item.get(key)
        if value:
            return str(value).replace("/", "_").replace(".wav", "")
    return fallback


def download_from_hub_files(
    dataset_name: str,
    output_csv: Path,
    max_hours: float | None = 1.0,
) -> pd.DataFrame:
    """Download repos that ship metadata.csv + loose WAV files."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    max_seconds = max_hours * 3600.0 if max_hours is not None else None

    meta_path = hf_hub_download(dataset_name, "metadata.csv", repo_type="dataset")
    meta = pd.read_csv(meta_path)
    text_col = next(c for c in meta.columns if c in ("transcription", "text", "transcript", "sentence"))
    file_col = next(c for c in meta.columns if c in ("file_name", "filename", "file", "audio"))

    rows = []
    total_seconds = 0.0
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Downloading clips"):
        wav_name = str(row[file_col])
        text = str(row[text_col]).strip()
        if not text:
            continue

        cached = hf_hub_download(dataset_name, wav_name, repo_type="dataset")
        wav, sr = sf.read(cached)
        duration = len(wav) / float(sr)

        if max_seconds is not None and total_seconds + duration > max_seconds and rows:
            break

        clip_id = Path(wav_name).stem
        out_path = RAW_DIR / wav_name
        if Path(cached).resolve() != out_path.resolve():
            shutil.copy2(cached, out_path)

        rows.append(
            {
                "id": clip_id,
                "wav_path": str(out_path.relative_to(ROOT)),
                "text": text,
                "duration_sec": round(duration, 3),
                "source_dataset": dataset_name,
            }
        )
        total_seconds += duration
        if max_seconds is not None and total_seconds >= max_seconds:
            break

    return _save_metadata(rows, output_csv)


def download_from_datasets_api(
    dataset_name: str,
    output_csv: Path,
    max_hours: float | None = 1.0,
    split: str = "train",
    seed: int = 42,
) -> pd.DataFrame:
    """Fallback for parquet/audio-feature datasets (uses streaming to avoid torchcodec)."""
    from datasets import load_dataset

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    max_seconds = max_hours * 3600.0 if max_hours is not None else None

    print(f"Loading dataset via streaming: {dataset_name}")
    ds = load_dataset(dataset_name, split=split, streaming=True)

    rows = []
    total_seconds = 0.0
    for i, item in enumerate(tqdm(ds, desc="Saving clips")):
        audio = item["audio"]
        text = find_text(item)
        if text is None:
            continue

        duration = len(audio["array"]) / float(audio["sampling_rate"])
        if max_seconds is not None and total_seconds + duration > max_seconds and rows:
            break

        clip_id = clip_id_from_item(item, fallback=f"{i:05d}")
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in clip_id)
        wav_path = RAW_DIR / f"{safe_id}.wav"
        sf.write(wav_path, audio["array"], int(audio["sampling_rate"]))

        rows.append(
            {
                "id": safe_id,
                "wav_path": str(wav_path.relative_to(ROOT)),
                "text": text,
                "duration_sec": round(duration, 3),
                "source_dataset": dataset_name,
            }
        )
        total_seconds += duration
        if max_seconds is not None and total_seconds >= max_seconds:
            break

    return _save_metadata(rows, output_csv)


def _save_metadata(rows: list[dict], output_csv: Path) -> pd.DataFrame:
    if not rows:
        raise RuntimeError("No clips downloaded. Check dataset format or credentials.")

    meta = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(output_csv, index=False)
    total_hours = meta["duration_sec"].sum() / 3600.0
    print(f"Saved {len(meta)} clips ({total_hours:.2f} hours) -> {output_csv}")
    return meta


def download(
    dataset_name: str,
    output_csv: Path,
    max_hours: float | None = 1.0,
    split: str = "train",
    seed: int = 42,
) -> pd.DataFrame:
    files = list_repo_files(dataset_name, repo_type="dataset")
    if "metadata.csv" in files and any(f.endswith(".wav") for f in files):
        return download_from_hub_files(dataset_name, output_csv, max_hours=max_hours)
    return download_from_datasets_api(
        dataset_name, output_csv, max_hours=max_hours, split=split, seed=seed
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Arabic TTS dataset from Hugging Face (Saudi audio search page)"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset id")
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "metadata.csv"),
        help="Output metadata CSV path",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=1.0,
        help="Maximum total audio hours to download (default: 1.0). Use 0 for all clips.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    max_hours = None if args.max_hours <= 0 else args.max_hours
    download(args.dataset, Path(args.output), max_hours=max_hours, split=args.split, seed=args.seed)


if __name__ == "__main__":
    main()
