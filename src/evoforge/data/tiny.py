from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch.utils.data import Dataset

DEFAULT_TEXT = (
    "In the beginning AI models were tiny,"
    " and curiosity powered their updates. "
    "This miniature corpus keeps experiments quick,"
    " ideal for laptop-scale research.\n"
)


class TinyCharDataset(Dataset[torch.Tensor]):
    def __init__(self, text: str, seq_len: int) -> None:
        if seq_len < 8:
            raise ValueError("seq_len must be >= 8 for training samples")
        vocab = sorted(set(text))
        self.vocab: List[str] = vocab
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        encoded = [self.stoi[ch] for ch in text]
        self.data = torch.tensor(encoded, dtype=torch.long)
        self.seq_len = seq_len
        if self.data.numel() <= seq_len + 1:
            raise ValueError("Text too short for given seq_len")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return self.data.numel() - self.seq_len - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x.clone(), y.clone()


def load_default_text() -> str:
    return DEFAULT_TEXT * 8  # repeat to lengthen corpus


def create_tiny_dataset(seq_len: int) -> TinyCharDataset:
    text = load_default_text()
    return TinyCharDataset(text, seq_len)
