import torch
import torch.nn as nn


class CharVocab:
    """Character-level vocabulary for encoding/decoding text."""
    def __init__(self, itos, stoi):
        self.itos = itos
        self.stoi = stoi

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


class CharRNN(nn.Module):
    """Character-level RNN with GRU or LSTM architecture."""
    def __init__(self, vocab_size, emb, hidden, layers, dropout, rnn_type="GRU"):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        rnn_cls = {"GRU": nn.GRU, "LSTM": nn.LSTM}[rnn_type.upper()]
        self.rnn = rnn_cls(emb, hidden, num_layers=layers, dropout=dropout if layers > 1 else 0.0, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, vocab_size)
        self.rnn_type = rnn_type.upper()
        self.layers = layers
        self.hidden = hidden

    def forward(self, x, h=None):
        x = self.emb(x)
        x, h = self.rnn(x, h)
        x = self.drop(x)
        return self.fc(x), h

    def init_hidden(self, batch_size, device):
        if self.rnn_type == "LSTM":
            return (torch.zeros(self.layers, batch_size, self.hidden, device=device),
                    torch.zeros(self.layers, batch_size, self.hidden, device=device))
        else:
            return torch.zeros(self.layers, batch_size, self.hidden, device=device)


def detect_device():
    """Auto-detect the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_model(checkpoint_path="char_rnn_checkpoint.pt"):
    """Load the trained CharRNN model from checkpoint.

    Args:
        checkpoint_path: Path to the saved checkpoint file

    Returns:
        tuple: (model, vocab, device)
    """
    device = detect_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Rebuild model with saved config
    model = CharRNN(
        vocab_size=len(checkpoint["itos"]),
        emb=checkpoint["config"]["embedding_dim"],
        hidden=checkpoint["config"]["hidden_dim"],
        layers=checkpoint["config"]["num_layers"],
        dropout=checkpoint["config"]["dropout"],
        rnn_type=checkpoint["config"]["rnn_type"]
    ).to(device)

    # Load trained weights
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Restore vocabulary
    vocab = CharVocab(checkpoint["itos"], checkpoint["stoi"])

    return model, vocab, device
