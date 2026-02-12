"""Character-level RNN supporting GRU and LSTM backends."""

import torch
import torch.nn as nn


class CharRNN(nn.Module):
    """Embedding -> RNN -> Linear head for next-character prediction."""

    SUPPORTED_TYPES = {"GRU": nn.GRU, "LSTM": nn.LSTM}

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 dropout=0.2, rnn_type="GRU"):
        super().__init__()
        rnn_type = rnn_type.upper()
        if rnn_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"rnn_type must be one of {list(self.SUPPORTED_TYPES)}, "
                f"got {rnn_type!r}"
            )

        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = self.SUPPORTED_TYPES[rnn_type](
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    # ------------------------------------------------------------------

    def forward(self, x, hidden=None):
        x = self.emb(x)
        x, hidden = self.rnn(x, hidden)
        x = self.drop(x)
        logits = self.fc(x)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        """Return zero-initialised hidden state for a given batch size."""
        shape = (self.num_layers, batch_size, self.hidden_dim)
        if self.rnn_type == "LSTM":
            return (torch.zeros(*shape, device=device),
                    torch.zeros(*shape, device=device))
        return torch.zeros(*shape, device=device)
