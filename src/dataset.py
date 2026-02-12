"""Dataset utilities for character-level language modelling."""

import torch
from torch.utils.data import Dataset, DataLoader


class CharChunkDataset(Dataset):
    """Sliding-window dataset over a 1-D tensor of token IDs.

    Each sample is an (input, target) pair of length ``seq_len``,
    where the target is shifted one position to the right.
    """

    def __init__(self, ids, seq_len, step=None):
        self.ids = ids
        self.seq_len = seq_len
        self.step = step if step is not None else seq_len

        # Pre-compute valid start positions
        max_start = len(ids) - seq_len - 1
        self.starts = list(range(0, max(max_start, 0) + 1, self.step))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        x = self.ids[s : s + self.seq_len]
        y = self.ids[s + 1 : s + 1 + self.seq_len]
        return x, y


def prepare_loaders(train_ids, val_ids, seq_len, batch_size, overlap_step=None):
    """Build DataLoaders for training and validation splits.

    Args:
        train_ids: 1-D LongTensor of training token IDs.
        val_ids:   1-D LongTensor of validation token IDs.
        seq_len:   Sequence length per sample.
        batch_size: Batch size.
        overlap_step: Sliding-window stride (None = non-overlapping).

    Returns:
        (train_loader, val_loader)
    """
    train_ds = CharChunkDataset(train_ids, seq_len, step=overlap_step)
    val_ds = CharChunkDataset(val_ids, seq_len, step=overlap_step)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=True,
    )
    return train_loader, val_loader
