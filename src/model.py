import torch
import torch.nn as nn

# detect the best available device (MPS > CUDA > CPU)
def detect_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# maps character to integer IDs and back
class CharVocab:
    def __init__(self, itos, stoi):
        self.itos = list(itos)
        self.stoi = dict(stoi)

    # build the vocabulary from a corpus string
    @classmethod
    def from_text(cls, text):
        chars = sorted(set(text))
        return cls(itos=chars, stoi={c: i for i, c in enumerate(chars)})

    # convert string to token IDs
    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    # convert token IDs back to a string
    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def __len__(self):
        return len(self.itos)

    def __repr__(self):
        return f"CharVocab(size={len(self)})"

# Embedding -> LSTM -> Linear head for next-character prediction
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.emb(x)
        x, hidden = self.rnn(x, hidden)
        x = self.drop(x)
        logits = self.fc(x)
        return logits, hidden

# persist model weights, vocab mappings, and config to disk
def save_checkpoint(path, model, vocab, config):
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "itos": vocab.itos,
        "stoi": vocab.stoi,
    }, path)

# load a checkpoint and return (model, vocab, config, device)
def load_checkpoint(path, device=None):
    if device is None:
        device = detect_device()

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    vocab = CharVocab(itos=ckpt["itos"], stoi=ckpt["stoi"])

    model = CharRNN(
        vocab_size=len(vocab),
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, vocab, cfg, device
