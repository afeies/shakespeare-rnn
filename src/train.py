"""Training loop and evaluation for CharRNN."""

import math

import torch
import torch.nn as nn

from src.utils import bpc, read_corpus, set_seed, detect_device
from src.vocab import CharVocab
from src.model import CharRNN
from src.dataset import prepare_loaders
from src.sampler import sample_text
from src.checkpoint import save_checkpoint
from src.config import DEFAULT_CONFIG


def evaluate(model, loader, criterion, vocab_size, device):
    """Compute average loss and BPC over a DataLoader."""
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


def train(config=None, verbose=True):
    """Run a full training session and return the path to the best checkpoint.

    Args:
        config:  Dict of hyper-parameters (falls back to DEFAULT_CONFIG).
        verbose: Whether to print progress during training.

    Returns:
        Path to the saved checkpoint with the lowest validation loss.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    set_seed(42)
    device = detect_device()
    if verbose:
        print(f"Training on {device}\n")

    # ---- data ----
    text = read_corpus(cfg["data_path"])
    vocab = CharVocab.from_text(text)
    vocab_size = len(vocab)

    data_ids = torch.tensor(vocab.encode(text), dtype=torch.long)
    n_val = max(1, int(len(data_ids) * cfg["val_fraction"]))
    train_ids, val_ids = data_ids[:-n_val], data_ids[-n_val:]

    train_loader, val_loader = prepare_loaders(
        train_ids, val_ids,
        seq_len=cfg["seq_len"],
        batch_size=cfg["batch_size"],
        overlap_step=cfg["overlap_step"],
    )

    # ---- model ----
    model = CharRNN(
        vocab_size=vocab_size,
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        rnn_type=cfg["rnn_type"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    # ---- training loop ----
    global_step = 0
    best_val = float("inf")

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        running = 0.0

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
            global_step += 1

            if verbose and global_step % cfg["log_every"] == 0:
                avg_bpc = bpc(running / cfg["log_every"])
                print(f"  epoch {epoch} | step {global_step} | train BPC {avg_bpc:.3f}")
                running = 0.0

            if verbose and global_step % cfg["sample_every"] == 0:
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

        # end-of-epoch validation
        val_loss, val_bpc = evaluate(model, val_loader, criterion, vocab_size, device)
        if verbose:
            print(f"  epoch {epoch} done | val BPC {val_bpc:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(cfg["save_path"], model, vocab, cfg)
            if verbose:
                print(f"  -> saved best checkpoint (BPC {val_bpc:.3f})")

        scheduler.step(val_loss)
        if verbose:
            print()

    if verbose:
        print(f"Training complete — best val BPC {bpc(best_val):.3f}")

    return cfg["save_path"]
