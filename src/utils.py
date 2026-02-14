"""Shared helpers used across the project."""

import math
import os
import random

import torch


def detect_device():
    """Auto-detect the best available compute device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def bpc(loss_value):
    """Convert cross-entropy loss (nats) to bits-per-character."""
    return loss_value / math.log(2.0)


def read_corpus(path):
    """Read a text file and return its contents as a string.

    Falls back to a tiny Shakespeare snippet if the file doesn't exist,
    so experiments can still run without data.
    """
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    return (
        "ROMEO:\nBut soft, what light through yonder window breaks?\n"
        "It is the east, and Juliet is the sun.\n"
    )

