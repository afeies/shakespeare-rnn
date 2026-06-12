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


def _format_elapsed(seconds):
    """Format a duration as e.g. '3m 42s'."""
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs}s"


def run_training():
    """Called by Gradio when the user clicks Train.

    Runs the training loop in a background thread and streams
    (status, logs, summary) updates as each epoch completes.
    """
    import queue
    import threading
    import time

    from src.train import bpc, train

    events = queue.Queue()

    def worker():
        try:
            path = train(verbose=False, on_epoch=lambda s: events.put(("epoch", s)))
            events.put(("done", path))
        except Exception as exc:  # surfaced to the user via gr.Error below
            events.put(("error", exc))

    start = time.monotonic()
    threading.Thread(target=worker, daemon=True).start()

    log_lines = []
    best_val = float("inf")
    yield "Status: Training...", "", ""

    while True:
        kind, payload = events.get()

        if kind == "epoch":
            best_val = min(best_val, payload["val_loss"])
            log_lines.append(
                f"Epoch {payload['epoch']}/{payload['num_epochs']}\n"
                f"Train Loss: {payload['train_loss']:.2f}\n"
                f"Val Loss: {payload['val_loss']:.2f}\n"
            )
            yield "Status: Training...", "\n".join(log_lines), ""

        elif kind == "done":
            # Model is cached per-process; drop it so the next generation
            # picks up the freshly trained checkpoint.
            _load_model.cache_clear()
            summary = (
                "---\n"
                "**Training Complete**\n\n"
                f"Elapsed Time: {_format_elapsed(time.monotonic() - start)}  \n"
                f"Final BPC: {bpc(best_val):.2f}  \n"
                f"Model Saved: {payload}"
            )
            yield "Status: Complete", "\n".join(log_lines), summary
            return

        else:  # error
            yield "Status: Idle", "\n".join(log_lines), ""
            raise gr.Error(f"Training failed: {payload}")


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

CSS = """
.header { text-align: center; margin-bottom: 0.5em; }
.output-text textarea { font-family: 'Georgia', serif !important; font-size: 1.05em; line-height: 1.6; }
footer { display: none !important; }
.nav-bar { gap: 0.5em; margin-bottom: 1em; }
.nav-btn { border-radius: 0 !important; }
.train-logs textarea { font-family: monospace !important; }
.placeholder { text-align: center; color: var(--body-text-color-subdued); margin-top: 4em; }
"""

NAV_TABS = ["Generate", "Train", "Models", "Learn"]


def build_ui():
    """Construct and return the Gradio Blocks app (without launching)."""
    with gr.Blocks(title="Shakespeare RNN") as demo:
        with gr.Row(elem_classes="nav-bar"):
            nav_buttons = [
                gr.Button(name, elem_classes="nav-btn") for name in NAV_TABS
            ]

        with gr.Column(visible=True) as generate_page:
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

        with gr.Column(visible=False) as train_page:
            gr.Markdown("# Train", elem_classes="header")

            train_btn = gr.Button("Train", variant="primary")
            train_status = gr.Markdown("Status: Idle")
            train_logs = gr.Textbox(
                label="Training logs",
                lines=16,
                interactive=False,
                elem_classes="train-logs",
            )
            train_summary = gr.Markdown("")

            train_btn.click(
                fn=run_training,
                inputs=None,
                outputs=[train_status, train_logs, train_summary],
            )

        placeholder_pages = []
        for name in NAV_TABS[2:]:
            with gr.Column(visible=False) as page:
                gr.Markdown(f"*{name} — coming soon.*", elem_classes="placeholder")
            placeholder_pages.append(page)

        pages = [generate_page, train_page, *placeholder_pages]

        def make_select(index):
            def select():
                return [gr.update(visible=i == index) for i in range(len(pages))]

            return select

        for i, btn in enumerate(nav_buttons):
            btn.click(fn=make_select(i), inputs=None, outputs=pages)

    return demo


def launch():
    """Build and launch the UI (theme/css go to launch() in Gradio 6)."""
    build_ui().launch(theme=gr.themes.Soft(), css=CSS)


# Allow running directly: python app.py
if __name__ == "__main__":
    launch()
