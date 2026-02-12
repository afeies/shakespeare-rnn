"""Character-level vocabulary for encoding and decoding text."""


class CharVocab:
    """Maps characters to integer IDs and back.

    Can be built from raw text or from pre-existing mappings
    (e.g. loaded from a checkpoint).
    """

    def __init__(self, itos, stoi):
        self.itos = list(itos)
        self.stoi = dict(stoi)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_text(cls, text):
        """Build vocabulary from a corpus string."""
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        return cls(itos=chars, stoi=stoi)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        """Restore vocabulary from a saved checkpoint dict."""
        return cls(itos=checkpoint["itos"], stoi=checkpoint["stoi"])

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, text):
        """Convert a string to a list of token IDs (unknown chars skipped)."""
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        """Convert a list of token IDs back to a string."""
        return "".join(self.itos[i] for i in ids)

    def __len__(self):
        return len(self.itos)

    def __repr__(self):
        return f"CharVocab(size={len(self)})"
