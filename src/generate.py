import random
import torch

# zero out everything outside the top-k most probable tokens
def _apply_top_k(probs, k):
    k = min(k, probs.numel())
    _, topk_idx = torch.topk(probs, k)
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[topk_idx] = True
    return probs.masked_fill(~mask, 0.0)

# nucleus sampling: keep the smallest set whole cumulative prob >= p
def _apply_top_p(probs, p):
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    keep = cumsum <= p
    
    # always keep the most probable token
    keep[0] = True
    filtered = torch.zeros_like(sorted_probs)
    filtered[keep] = sorted_probs[keep]
    
    # scatter back to original ordering
    out = torch.zeros_like(probs)
    out.scatter_(0, sorted_idx, filtered)
    return out

# generate text autoregressively, yielding partial output as it grows
@torch.no_grad()
def stream_text(model, vocab, *, max_tokens=300, temperature=0.9,
                top_k=40, top_p=0.9, device="cpu", chunk_size=4):
    model.eval()

    # generation is seeded with a single random character from the vocabulary.
    seed = random.choice(vocab.itos)
    input_ids = torch.tensor(
        vocab.encode(seed), dtype=torch.long, device=device
    ).unsqueeze(0)
    hidden = None
    output = [seed]

    for step in range(max_tokens):
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

        if (step + 1) % chunk_size == 0:
            yield "".join(output)

    yield "".join(output)

# like :func:`stream_text`, but blocks and returns the full string
def sample_text(model, vocab, **kwargs):
    result = ""
    for result in stream_text(model, vocab, **kwargs):
        pass
    return result
