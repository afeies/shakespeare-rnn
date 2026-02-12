"""Gradio web interface for Shakespeare RNN text generation."""

from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Globals populated at startup
# ---------------------------------------------------------------------------
_model = None
_vocab = None
_device = None
_loaded_path = None

DEFAULT_CHECKPOINT = "checkpoints/char_rnn_checkpoint.pt"


def _ensure_model(checkpoint_path=None):
    """Lazy-load the model so the import itself stays cheap."""
    global _model, _vocab, _device, _loaded_path

    path = checkpoint_path or DEFAULT_CHECKPOINT
    if _model is not None and _loaded_path == path:
        return

    from src.checkpoint import load_checkpoint

    _model, _vocab, _cfg, _device = load_checkpoint(path)
    _loaded_path = path


def generate_text(prompt, length, temperature, top_k, top_p):
    """Called by Gradio when the user clicks Generate."""
    _ensure_model()

    from src.sampler import sample_text

    length = int(length)
    top_k = int(top_k)

    text = sample_text(
        _model,
        _vocab,
        max_tokens=length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        prompt=prompt,
        device=_device,
    )
    return text


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
    with gr.Blocks(css=CSS, title="Shakespeare RNN", theme=gr.themes.Soft()) as demo:
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
                    show_copy_button=True,
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


# Allow running directly: python app.py
if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
