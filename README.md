# Shakespeare RNN

A character-level LSTM that learns to generate text in the style of Shakespeare (or any other corpus you throw at it).

## Quick start

```bash
uv sync
```

### Train a model

```bash
uv run python -m src.train
```

All hyperparameters live in `DEFAULT_CONFIG` at the top of `src/train.py` —
edit them there. The best checkpoint (lowest validation loss) is saved to
`checkpoints/char_rnn_checkpoint.pt`.

### Run the app

```bash
uv run python app.py
```

Opens a Gradio interface in your browser with a prompt box and sliders for
temperature, top-k, top-p, and output length.

## Project structure

```
├── app.py              # Gradio web frontend
├── pyproject.toml
├── checkpoints/        # saved .pt checkpoints (gitignored)
├── data/               # training corpora
└── src/
    ├── model.py        # vocab, CharRNN (LSTM), checkpoint save/load
    ├── train.py        # config + dataset + training loop (python -m src.train)
    └── generate.py     # top-k / top-p sampling
```


## Notes

Study notes (RNN/LSTM concepts, tensor shapes, glossary) and the
hyperparameter experiment log live in [NOTES.md](NOTES.md).
