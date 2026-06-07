from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, soundfile as sf, torch, yaml
from data.audio_utils import mel_to_wav
from data.text_normalize import load_vocab, text_to_ids
from model.fastspeech import FastSpeech2

ROOT = Path(__file__).resolve().parent

def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    m = FastSpeech2(
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
    m.load_state_dict(ckpt["model"])
    m.to(device)
    m.eval()
    return m, cfg

def synthesize(text, checkpoint, config_path, output, device_name="cpu"):
    cfg = yaml.safe_load(open(config_path))
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    model, ckpt_cfg = load_model(checkpoint, device)
    ids = text_to_ids(text, load_vocab(ROOT / "data" / "vocab.json"), add_eos=True)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        mel = model.infer(tokens, lengths).cpu().numpy()
    wav = mel_to_wav(mel, ckpt_cfg, cfg["synthesis"].get("griffin_lim_iters", 64))
    peak = np.max(np.abs(wav)) if wav.size else 1.0
    if peak > 0:
        wav = wav / peak * 0.95
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, wav, ckpt_cfg["audio"]["sample_rate"])
    print(f"Saved {output} ({len(wav) / ckpt_cfg['audio']['sample_rate']:.2f}s)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "best.pt"))
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    a = p.parse_args()
    synthesize(a.text, Path(a.checkpoint), Path(a.config), Path(a.out), a.device)