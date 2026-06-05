"""PyTorch dataset for FastSpeech2 training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.text_normalize import load_vocab, text_to_ids

ROOT = Path(__file__).resolve().parents[1]


class TTSDataset(Dataset):
    def __init__(self, csv_path: Path, vocab_path: Path, max_text_len: int = 256):
        self.df = pd.read_csv(csv_path)
        self.vocab = load_vocab(vocab_path)
        self.max_text_len = max_text_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        ids = text_to_ids(str(row["text"]), self.vocab, add_eos=True)
        ids = ids[: self.max_text_len]
        mel = np.load(ROOT / row["mel_path"])
        durations = np.load(ROOT / row["duration_path"])

        if len(durations) != len(ids):
            # Recompute if text was truncated.
            from data.preprocess import char_durations

            durations = char_durations(len(ids), mel.shape[1])

        return {
            "tokens": torch.tensor(ids, dtype=torch.long),
            "mel": torch.tensor(mel, dtype=torch.float),
            "durations": torch.tensor(durations, dtype=torch.float),
            "text": row["text"],
            "id": row["id"],
        }


def collate_batch(batch: list[dict]) -> dict:
    token_lengths = torch.tensor([b["tokens"].size(0) for b in batch], dtype=torch.long)
    mel_lengths = torch.tensor([b["mel"].size(1) for b in batch], dtype=torch.long)
    max_tokens = int(token_lengths.max())
    max_mel = int(mel_lengths.max())
    n_mels = batch[0]["mel"].size(0)

    tokens = torch.zeros(len(batch), max_tokens, dtype=torch.long)
    durations = torch.zeros(len(batch), max_tokens, dtype=torch.float)
    mels = torch.zeros(len(batch), n_mels, max_mel, dtype=torch.float)

    for i, item in enumerate(batch):
        t_len = item["tokens"].size(0)
        m_len = item["mel"].size(1)
        tokens[i, :t_len] = item["tokens"]
        durations[i, :t_len] = item["durations"]
        mels[i, :, :m_len] = item["mel"]

    return {
        "tokens": tokens,
        "token_lengths": token_lengths,
        "durations": durations,
        "mels": mels,
        "mel_lengths": mel_lengths,
        "texts": [b["text"] for b in batch],
        "ids": [b["id"] for b in batch],
    }
