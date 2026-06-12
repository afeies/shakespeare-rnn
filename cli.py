"""CLI for training and generating text with a character-level RNN."""

from pathlib import Path

import typer

app = typer.Typer(help="Shakespeare RNN — character-level text generation")


@app.command()
def train(
    data: Path = typer.Option(
        "data/tinyshakespeare.txt", "--data", "-d",
        help="Path to the training corpus.",
    ),
    epochs: int = typer.Option(20, "--epochs", "-e", help="Number of training epochs."),
    lr: float = typer.Option(2e-3, "--lr", help="Learning rate."),
    batch_size: int = typer.Option(128, "--batch-size", "-b", help="Batch size."),
    seq_len: int = typer.Option(256, "--seq-len", help="Sequence length per sample."),
    hidden: int = typer.Option(512, "--hidden", help="Hidden dimension."),
    save_path: Path = typer.Option(
        "checkpoints/char_rnn_checkpoint.pt", "--save", "-s",
        help="Where to save the best checkpoint.",
    ),
):
    """Train a character-level RNN on a text corpus."""
    from src.train import make_config, train as run_training

    cfg = make_config(
        data_path=str(data),
        num_epochs=epochs,
        learning_rate=lr,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden,
        save_path=str(save_path),
    )
    run_training(config=cfg)


@app.command()
def generate(
    checkpoint: Path = typer.Option(
        "checkpoints/char_rnn_checkpoint.pt", "--checkpoint", "-c",
        help="Path to a trained checkpoint.",
    ),
    prompt: str = typer.Option("", "--prompt", "-p", help="Seed text for generation."),
    length: int = typer.Option(500, "--length", "-n", help="Characters to generate."),
    temperature: float = typer.Option(0.9, "--temp", "-t", help="Sampling temperature."),
    top_k: int = typer.Option(40, "--top-k", help="Top-k filtering."),
    top_p: float = typer.Option(0.9, "--top-p", help="Nucleus sampling threshold."),
):
    """Generate text from a trained checkpoint."""
    from src.model import load_checkpoint
    from src.generate import sample_text

    if not checkpoint.exists():
        typer.echo(f"Checkpoint not found: {checkpoint}", err=True)
        raise typer.Exit(code=1)

    model, vocab, _cfg, device = load_checkpoint(str(checkpoint))
    text = sample_text(
        model, vocab,
        max_tokens=length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        prompt=prompt,
        device=device,
    )
    typer.echo(text)


@app.command()
def ui():
    """Launch the Gradio web UI for interactive text generation."""
    from app import build_ui

    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    app()
