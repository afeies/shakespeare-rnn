import torch
import torch.nn as nn

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