"""Gradio web interface for Shakespeare RNN text generation."""

from functools import lru_cache
from pathlib import Path

import gradio as gr

DEFAULT_CHECKPOINT = "checkpoints/char_rnn_checkpoint.pt"


@lru_cache(maxsize=1)
def _load_model():
    """Load the model once, on the first generation request."""
    from src.model import load_checkpoint

    return load_checkpoint(DEFAULT_CHECKPOINT)


def generate_text(prompt, length, temperature, top_k, top_p):
    """Called by Gradio when the user clicks Generate.

    A generator: Gradio streams each yielded string to the output textbox,
    giving a typewriter effect.
    """
    if not Path(DEFAULT_CHECKPOINT).exists():
        raise gr.Error(
            f"No trained checkpoint found at '{DEFAULT_CHECKPOINT}'. "
            "Train one first: uv run python -m src.train"
        )

    from src.generate import stream_text

    model, vocab, _cfg, device = _load_model()
    yield from stream_text(
        model,
        vocab,
        max_tokens=int(length),
        temperature=temperature,
        top_k=int(top_k),
        top_p=top_p,
        prompt=prompt,
        device=device,
        chunk_size=4,
    )


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

CSS = """
.header { text-align: center; margin-bottom: 0.5em; }
.output-text textarea { font-family: 'Georgia', serif !important; font-size: 1.05em; line-height: 1.6; }
footer { display: none !important; }
"""


def build_ui():
    """Construct and return the Gradio Blocks app (without launching)."""
    with gr.Blocks(title="Shakespeare RNN") as demo:
        gr.Markdown(
            "# Shakespeare RNN\n"
            "Generate character-level text with a trained RNN.  "
            "Tweak the knobs below to control style and randomness.",
            elem_classes="header",
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="e.g. ROMEO:",
                    lines=2,
                )
                length = gr.Slider(
                    50, 2000, value=500, step=50, label="Length (characters)",
                )
                temperature = gr.Slider(
                    0.1, 2.0, value=0.9, step=0.05, label="Temperature",
                )
                top_k = gr.Slider(
                    1, 100, value=40, step=1, label="Top-k",
                )
                top_p = gr.Slider(
                    0.1, 1.0, value=0.9, step=0.05, label="Top-p (nucleus)",
                )
                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=2):
                output = gr.Textbox(
                    label="Generated text",
                    lines=20,
                    elem_classes="output-text",
                )

        generate_btn.click(
            fn=generate_text,
            inputs=[prompt, length, temperature, top_k, top_p],
            outputs=output,
        )

        gr.Markdown(
            "---\n"
            "*Powered by a character-level RNN trained on Shakespeare / custom corpora.*"
        )

    return demo


def launch():
    """Build and launch the UI (theme/css go to launch() in Gradio 6)."""
    build_ui().launch(theme=gr.themes.Soft(), css=CSS)


# Allow running directly: python app.py
if __name__ == "__main__":
    launch()
