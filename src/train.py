import math
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.model import CharRNN, CharVocab, detect_device, save_checkpoint
from src.generate import sample_text

DEFAULT_CONFIG = {
    "data_path": "data/tinyshakespeare.txt",
    "seq_len": 256,
    "batch_size": 128,
    "embedding_dim": 256,
    "hidden_dim": 512,
    "num_layers": 2,
    "dropout": 0.2,
    "num_epochs": 20,
    "learning_rate": 2e-3,
    "grad_clip": 1.0,
    "log_every": 100,
    "sample_every": 200,
    "max_generate": 400,
    "temperature": 0.9,
    "top_k": 40,
    "top_p": 0.9,
    "val_fraction": 0.05,
    "save_path": "checkpoints/char_rnn_checkpoint.pt",
}

# return a fresh config dict with any overrides applied
def make_config(**overrides):
    return {**DEFAULT_CONFIG, **overrides}

# set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# convert cross-entropy loss to bits-per-character
def bpc(loss_value):
    return loss_value / math.log(2.0)

# read a text file and return its contents as a string
def read_corpus(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()

# non-overlapping chunks over a 1-D tensor of token IDs
class CharChunkDataset(Dataset):
    def __init__(self, ids, seq_len):
        self.ids = ids
        self.seq_len = seq_len

        max_start = len(ids) - seq_len - 1
        self.starts = list(range(0, max(max_start, 0) + 1, seq_len))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        x = self.ids[s : s + self.seq_len]
        y = self.ids[s + 1 : s + 1 + self.seq_len]
        return x, y

# compute average loss and BPC over a DataLoader
def evaluate(model, loader, criterion, vocab_size, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
            n_batches += 1

    if n_batches == 0:
        return float("nan"), float("nan")

    avg = total_loss / n_batches
    return avg, bpc(avg)

# train and return path to best checkpoint
def train(config=None, on_epoch=None):
    cfg = make_config(**(config or {}))

    set_seed(42)
    device = detect_device()
    print(f"Training on {device}\n")

    text = read_corpus(cfg["data_path"])
    vocab = CharVocab.from_text(text)
    vocab_size = len(vocab)

    data_ids = torch.tensor(vocab.encode(text), dtype=torch.long)
    n_val = max(1, int(len(data_ids) * cfg["val_fraction"]))
    train_ids, val_ids = data_ids[:-n_val], data_ids[-n_val:]

    train_loader = DataLoader(
        CharChunkDataset(train_ids, cfg["seq_len"]),
        batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        CharChunkDataset(val_ids, cfg["seq_len"]),
        batch_size=cfg["batch_size"], shuffle=False, drop_last=True,
    )

    model = CharRNN(
        vocab_size=vocab_size,
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    global_step = 0
    best_val = float("inf")

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        running = 0.0
        epoch_total = 0.0
        epoch_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["grad_clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

            running += loss.item()
            epoch_total += loss.item()
            epoch_batches += 1
            global_step += 1

            if global_step % cfg["log_every"] == 0:
                avg_bpc = bpc(running / cfg["log_every"])
                print(f"  epoch {epoch} | step {global_step} | train BPC {avg_bpc:.3f}")
                running = 0.0

            if global_step % cfg["sample_every"] == 0:
                snippet = sample_text(
                    model, vocab,
                    max_tokens=cfg["max_generate"],
                    temperature=cfg["temperature"],
                    top_k=cfg["top_k"],
                    top_p=cfg["top_p"],
                    device=device,
                )
                preview = snippet[:200] + "..." if len(snippet) > 200 else snippet
                print(f"\n--- sample ---\n{preview}\n---\n")

        val_loss, val_bpc = evaluate(model, val_loader, criterion, vocab_size, device)
        print(f"  epoch {epoch} done | val BPC {val_bpc:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(cfg["save_path"], model, vocab, cfg)
            print(f"  -> saved best checkpoint (BPC {val_bpc:.3f})")

        if on_epoch:
            on_epoch({
                "epoch": epoch,
                "num_epochs": cfg["num_epochs"],
                "train_loss": epoch_total / max(epoch_batches, 1),
                "val_loss": val_loss,
            })

        scheduler.step(val_loss)
        print()

    print(f"Training complete — best val BPC {bpc(best_val):.3f}")

    return cfg["save_path"]

# Run with python -m src.train
if __name__ == "__main__":
    train()
