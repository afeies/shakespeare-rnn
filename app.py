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


GENERATION_LENGTH = 500
TEMPERATURE = 0.9
TOP_K = 40
TOP_P = 0.9


def generate_text():
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
        max_tokens=GENERATION_LENGTH,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
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
            "Generate character-level text with a trained RNN.",
            elem_classes="header",
        )

        generate_btn = gr.Button("Generate", variant="primary")
        output = gr.Textbox(
            label="Generated text",
            lines=20,
            elem_classes="output-text",
        )

        generate_btn.click(
            fn=generate_text,
            inputs=None,
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
