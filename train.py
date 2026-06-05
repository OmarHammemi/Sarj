"""Train FastSpeech2-small on Arabic TTS data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import TTSDataset, collate_batch
from data.text_normalize import load_vocab
from model.fastspeech import build_model, count_parameters

ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def duration_loss(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    losses = []
    for b in range(pred.size(0)):
        n = int(lengths[b].item())
        p = torch.log1p(pred[b, :n].clamp(min=0.0))
        t = torch.log1p(target[b, :n].clamp(min=0.0))
        losses.append(F.mse_loss(p, t))
    return torch.stack(losses).mean()


def mel_loss(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    losses = []
    for b in range(pred.size(0)):
        n = int(lengths[b].item())
        losses.append(F.l1_loss(pred[b, :, :n], target[b, :, :n]))
    return torch.stack(losses).mean()


def validate(model, loader, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(batch["tokens"], batch["token_lengths"], batch["mels"], batch["durations"])
            loss = mel_loss(out["mel_pred"], batch["mels"], batch["mel_lengths"])
            if "mel_post" in out:
                loss = loss + mel_loss(out["mel_post"], batch["mels"], batch["mel_lengths"])
            loss = loss + duration_loss(out["duration_pred"], batch["durations"], batch["token_lengths"])
            total += loss.item()
    return total / max(len(loader), 1)


def train(config_path: Path, device_name: str, resume: Path | None = None) -> None:
    cfg = load_config(config_path)
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    print(f"Using device: {device}")

    vocab = load_vocab(ROOT / "data" / "vocab.json")
    train_ds = TTSDataset(ROOT / "data" / "train.csv", ROOT / "data" / "vocab.json")
    val_ds = TTSDataset(ROOT / "data" / "val.csv", ROOT / "data" / "vocab.json")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_batch,
    )

    model = build_model(cfg, len(vocab)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / "train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    step = 0
    best_val = float("inf")
    if resume and resume.exists():
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = state.get("step", 0)
        best_val = state.get("best_val", best_val)
        print(f"Resumed from {resume} at step {step}")

    pbar = tqdm(total=cfg["train"]["max_steps"], initial=step, desc="Training")
    train_iter = iter(train_loader)
    model.train()

    while step < cfg["train"]["max_steps"]:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch["tokens"], batch["token_lengths"], batch["mels"], batch["durations"])

        loss_mel = mel_loss(out["mel_pred"], batch["mels"], batch["mel_lengths"])
        if "mel_post" in out:
            loss_mel = loss_mel + mel_loss(out["mel_post"], batch["mels"], batch["mel_lengths"])
        loss_dur = duration_loss(out["duration_pred"], batch["durations"], batch["token_lengths"])
        loss = loss_mel + loss_dur

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        optimizer.step()
        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=float(loss.item()))

        if step % cfg["train"]["log_every"] == 0:
            record = {"step": step, "train_loss": float(loss.item())}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        if step % cfg["train"]["save_every"] == 0:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                    "vocab_size": len(vocab),
                    "param_count": count_parameters(model),
                    "best_val": best_val,
                },
                ckpt_dir / "latest.pt",
            )

        if step % cfg["train"]["val_every"] == 0:
            val_loss = validate(model, val_loader, device)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": cfg,
                        "vocab_size": len(vocab),
                        "param_count": count_parameters(model),
                        "best_val": best_val,
                    },
                    ckpt_dir / "best.pt",
                )
            model.train()

    pbar.close()
    print(f"Training complete. Best val loss: {best_val:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FastSpeech2-small")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume")
    args = parser.parse_args()
    train(Path(args.config), args.device, Path(args.resume) if args.resume else None)


if __name__ == "__main__":
    main()
