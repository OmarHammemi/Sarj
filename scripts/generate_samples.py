"""Batch-generate sample wavs."""
import subprocess, sys
from pathlib import Path

PHRASES = [
    ("01_salam", "السَّلَامُ عَلَيْكُمْ"),
    ("02_how", "كَيْفَ حَالُكَ الْيَوْمَ"),
    ("03_arabic", "أُحِبُّ تَعَلُّمَ اللُّغَةِ الْعَرَبِيَّةِ"),
    ("04_thanks", "شُكْرًا جَزِيلًا لَكُمْ"),
    ("05_tech", "تُسَاعِدُ أَدَواتُ تَطْوِيرِ البرمَجَاتِ فِي تَسْهِيلِ العَمَلِ"),
]

def main():
    out_dir = Path("samples/batch")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = "checkpoints/best.pt"
    cfg = "configs/train_hesham.yaml"
    for name, text in PHRASES:
        out = out_dir / f"{name}.wav"
        subprocess.run([
            sys.executable, "synthesize_v2.py",
            "--text", text, "--out", str(out),
            "--checkpoint", ckpt, "--config", cfg, "--device", "cuda",
        ], check=True)
        print(f"OK {name}: {text[:40]}")

if __name__ == "__main__":
    main()