"""Dataset utilities for character-level language modelling."""

from torch.utils.data import Dataset, DataLoader


class CharChunkDataset(Dataset):
    """Non-overlapping chunks over a 1-D tensor of token IDs.

    Each sample is an (input, target) pair of length ``seq_len``,
    where the target is shifted one position to the right.
    """

    def __init__(self, ids, seq_len):
        self.ids = ids
        self.seq_len = seq_len

        # Pre-compute valid start positions
        max_start = len(ids) - seq_len - 1
        self.starts = list(range(0, max(max_start, 0) + 1, seq_len))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        x = self.ids[s : s + self.seq_len]
        y = self.ids[s + 1 : s + 1 + self.seq_len]
        return x, y


def prepare_loaders(train_ids, val_ids, seq_len, batch_size):
    """Build DataLoaders for training and validation splits."""
    train_ds = CharChunkDataset(train_ids, seq_len)
    val_ds = CharChunkDataset(val_ids, seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=True,
    )
    return train_loader, val_loader
