"""Training and generation configuration defaults."""

DEFAULT_CONFIG = {
    "data_path": "data/tinyshakespeare.txt",
    "seq_len": 256,
    "batch_size": 128,
    "embedding_dim": 256,
    "hidden_dim": 512,
    "num_layers": 2,
    "dropout": 0.2,
    "rnn_type": "LSTM",
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
    "overlap_step": None,
    "save_path": "checkpoints/char_rnn_checkpoint.pt",
}


def make_config(**overrides):
    """Return a fresh config dict with any overrides applied."""
    cfg = dict(DEFAULT_CONFIG)
    for k, v in overrides.items():
        if k not in cfg:
            raise ValueError(f"Unknown config key: {k!r}")
        cfg[k] = v
    return cfg
