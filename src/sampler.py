"""Text generation / sampling from a trained CharRNN."""

import random
import torch


def _apply_top_k(probs, k):
    """Zero out everything outside the top-k most probable tokens."""
    k = min(k, probs.numel())
    _, topk_idx = torch.topk(probs, k)
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[topk_idx] = True
    return probs.masked_fill(~mask, 0.0)


def _apply_top_p(probs, p):
    """Nucleus sampling: keep the smallest set whose cumulative prob >= p."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    keep = cumsum <= p
    keep[0] = True  # always keep the most probable token
    filtered = torch.zeros_like(sorted_probs)
    filtered[keep] = sorted_probs[keep]
    # scatter back to original ordering
    out = torch.zeros_like(probs)
    out.scatter_(0, sorted_idx, filtered)
    return out


@torch.no_grad()
def sample_text(model, vocab, *, max_tokens=300, temperature=0.9,
                top_k=40, top_p=0.9, prompt="", device="cpu"):
    """Generate text autoregressively from *model*.

    Args:
        model:       A trained CharRNN in eval mode.
        vocab:       CharVocab instance.
        max_tokens:  Number of characters to generate.
        temperature: Softmax temperature (lower = more conservative).
        top_k:       Top-k filtering (None to disable).
        top_p:       Nucleus sampling threshold (None to disable).
        prompt:      Optional seed string; a random character is used if empty.
        device:      Torch device.

    Returns:
        The generated string (prompt + new characters).
    """
    model.eval()

    if not prompt:
        prompt = random.choice(vocab.itos)

    input_ids = torch.tensor(
        vocab.encode(prompt), dtype=torch.long, device=device
    ).unsqueeze(0)
    hidden = None
    output = list(prompt)

    for _ in range(max_tokens):
        logits, hidden = model(input_ids, hidden)
        last_logits = logits[0, -1, :] / max(temperature, 1e-8)
        probs = torch.softmax(last_logits, dim=-1)

        if top_k is not None:
            probs = _apply_top_k(probs, top_k)
        if top_p is not None:
            probs = _apply_top_p(probs, top_p)

        total = probs.sum()
        if total <= 0 or torch.isnan(total):
            next_id = torch.argmax(last_logits).item()
        else:
            probs = probs / total
            next_id = torch.multinomial(probs, 1).item()

        output.append(vocab.itos[next_id])
        input_ids = torch.tensor([[next_id]], device=device)

    return "".join(output)
