import torch
import random


class GeneratorSession:
    """Manages stateful text generation with pause/resume capability."""

    def __init__(self, model, vocab, device, max_tokens=500):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.max_tokens = max_tokens
        self.reset()

    def reset(self):
        """Reset session to initial state with a random prompt."""
        self.hidden = None
        self.output_chars = []
        self.input_ids = None
        self.position = 0
        self.is_complete = False
        self.stop_flag = False

        # Initialize with random character
        prompt = random.choice(self.vocab.itos)
        self.output_chars = list(prompt)
        self.input_ids = torch.tensor(
            self.vocab.encode(prompt),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

    def generate_next_char(self, temperature=0.9, top_k=40, top_p=0.9):
        """Generate one character. Returns (char, is_complete)."""
        with torch.no_grad():
            # Forward pass
            logits, self.hidden = self.model(self.input_ids, self.hidden)
            last_logits = logits[0, -1, :] / max(1e-6, temperature)
            probs = torch.softmax(last_logits, dim=-1)

            # Apply top_k filtering
            if top_k is not None:
                k = min(top_k, probs.numel())
                topk_vals, topk_idx = torch.topk(probs, k)
                mask = torch.zeros_like(probs, dtype=torch.bool)
                mask[topk_idx] = True
                probs = probs.masked_fill(~mask, 0)

            # Apply top_p (nucleus) sampling
            if top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=0)
                keep = cumsum <= top_p
                keep[0] = True
                filtered = torch.zeros_like(sorted_probs).masked_scatter(
                    keep, sorted_probs[keep])
                probs = torch.zeros_like(probs).scatter(0, sorted_idx, filtered)

            # Sample next character
            s = probs.sum()
            if s <= 0 or torch.isnan(s):
                next_id = torch.argmax(last_logits)
            else:
                probs = probs / s
                next_id = torch.multinomial(probs, 1).item()

            # Update state
            char = self.vocab.itos[int(next_id)]
            self.output_chars.append(char)
            self.input_ids = torch.tensor([[next_id]], device=self.device)
            self.position += 1

            if self.position >= self.max_tokens:
                self.is_complete = True

            return char, self.is_complete
