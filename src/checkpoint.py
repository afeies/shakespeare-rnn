"""Save and load model checkpoints."""

import torch

from src.model import CharRNN
from src.vocab import CharVocab
from src.utils import detect_device


def save_checkpoint(path, model, vocab, config):
    """Persist model weights, vocab mappings, and config to disk."""
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "itos": vocab.itos,
        "stoi": vocab.stoi,
    }, path)


def load_checkpoint(path, device=None):
    """Load a checkpoint and return (model, vocab, config, device).

    Args:
        path:   Path to the ``.pt`` checkpoint file.
        device: Torch device to map tensors onto. Auto-detected if ``None``.

    Returns:
        Tuple of (model, vocab, config_dict, device).
    """
    if device is None:
        device = detect_device()

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    vocab = CharVocab.from_checkpoint(ckpt)

    model = CharRNN(
        vocab_size=len(vocab),
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        rnn_type=cfg["rnn_type"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, vocab, cfg, device
