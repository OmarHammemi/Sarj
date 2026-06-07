"""Measure Real-Time Factor (RTF) on CPU or GPU."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.audio_utils import mel_to_wav
from data.text_normalize import load_vocab, text_to_ids
from synthesize import load_model
DEFAULT_TEXT = "السَّلامُ عَلَيْكُمْ وَرَحْمَةُ اللهِ"


def benchmark(
    checkpoint: Path,
    config_path: Path,
    device_name: str,
    text: str,
    warmup: int = 3,
    runs: int = 5,
) -> dict:
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    vocab = load_vocab(ROOT / "data" / "vocab.json")
    model, ckpt_cfg = load_model(checkpoint, device)
    ids = text_to_ids(text, vocab, add_eos=True)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)
    n_iters = cfg["synthesis"]["griffin_lim_iters"]
    sr = ckpt_cfg["audio"]["sample_rate"]

    def run_once() -> tuple[float, float]:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            mel = model.infer(tokens, lengths).cpu().numpy()
        wav = mel_to_wav(mel, ckpt_cfg, n_iters)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        audio_dur = len(wav) / sr
        return elapsed, audio_dur

    for _ in range(warmup):
        run_once()

    elapsed_times = []
    audio_dur = 0.0
    for _ in range(runs):
        elapsed, audio_dur = run_once()
        elapsed_times.append(elapsed)

    avg_elapsed = statistics.mean(elapsed_times)
    rtf = avg_elapsed / audio_dur if audio_dur > 0 else float("inf")
    return {
        "device": str(device),
        "text": text,
        "audio_duration_sec": round(audio_dur, 4),
        "avg_inference_sec": round(avg_elapsed, 4),
        "rtf": round(rtf, 4),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RTF")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "latest.pt"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    result = benchmark(
        Path(args.checkpoint),
        Path(args.config),
        args.device,
        args.text,
        warmup=args.warmup,
        runs=args.runs,
    )
    print("RTF benchmark result:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
