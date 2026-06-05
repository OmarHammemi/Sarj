"""Arabic text normalization and character vocabulary."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

PAD = "<pad>"
EOS = "<eos>"
UNK = "<unk>"

SPECIAL_TOKENS = [PAD, EOS, UNK]


def normalize_arabic(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u0640", "")  # tatweel
    text = re.sub("[أإآ]", "ا", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vocab(texts: list[str], min_freq: int = 1) -> dict[str, int]:
    counts: dict[str, int] = {}
    for text in texts:
        for ch in normalize_arabic(text):
            counts[ch] = counts.get(ch, 0) + 1

    chars = sorted(ch for ch, n in counts.items() if n >= min_freq)
    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def save_vocab(vocab: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def text_to_ids(text: str, vocab: dict[str, int], add_eos: bool = True) -> list[int]:
    unk_id = vocab[UNK]
    ids = [vocab.get(ch, unk_id) for ch in normalize_arabic(text)]
    if add_eos:
        ids.append(vocab[EOS])
    return ids


def ids_to_text(ids: list[int], inv_vocab: dict[int, str]) -> str:
    chars = []
    for idx in ids:
        tok = inv_vocab.get(idx, UNK)
        if tok in (PAD, EOS):
            continue
        if tok != UNK:
            chars.append(tok)
    return "".join(chars)
