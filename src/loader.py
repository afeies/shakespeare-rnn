import torch

from src.utils import detect_device
from src.encoder import CharVocab
from src.rnnModel import CharRNN

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
