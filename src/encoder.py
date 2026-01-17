class CharVocab:
    """Character-level vocabulary for encoding/decoding text."""
    def __init__(self, itos, stoi):
        self.itos = itos
        self.stoi = stoi

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)
