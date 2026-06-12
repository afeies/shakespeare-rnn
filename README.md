# Shakespeare RNN

A character-level LSTM that learns to generate text in the style of Shakespeare (or any other corpus you throw at it).

## Quick start

```bash
uv sync
```

### Generate text (CLI)

```bash
# generate from the default checkpoint
uv run python cli.py generate --prompt "ROMEO:" --length 500

# tweak sampling parameters
uv run python cli.py generate -p "To be" -n 800 --temp 0.7 --top-k 30
```

### Train a model (CLI)

```bash
uv run python cli.py train --data data/tinyshakespeare.txt --epochs 20
```

### Web UI

```bash
uv run python cli.py ui
# or directly:
uv run python app.py
```

Opens a Gradio interface in your browser with sliders for temperature, top-k, top-p, and output length.

## Project structure

```
├── cli.py              # typer CLI (train / generate / ui)
├── app.py              # Gradio web frontend
├── pyproject.toml
├── checkpoints/        # saved .pt checkpoints (gitignored)
├── data/               # training corpora
└── src/
    ├── model.py        # vocab, CharRNN (LSTM), checkpoint save/load
    ├── train.py        # config defaults, dataset, training loop
    └── generate.py     # top-k / top-p sampling
```

---

## Performance log

Step 1 — hidden dim 256
1. epochs 5, num_layers 1: 2.173
2. epochs 20, num layers 1: 1.906
    - cpu (Apple M4): 2m 2.3s
    - mps: 2m 12.6s
3. epochs 20, num_layers 2: 1.821
    - mps: 3m 50.8s
4. epochs 20, num_layers 3: 1.793
    - mps: 5m 47.8s
5. epochs 20, num_layers 2, hidden_dim 512: 1.772
    - mps: 5m 52.1s

Step 2 — seq len 128, grad clip 1.0
1. seq len 256: 1.737
    - mps: 5m 44.2s

seq len 256, learning rate 2e-3, grad clip 1.0, batch size 128, dropout 0.1

Nan fix: seq len 384, learning rate 1e-4, grad clip 0.1, batch size 32, dropout 0.3: 2.059 still learning
    - mps: 16m 55.4s

2. Add scheduler, 2e-3: 1.735

Base learning rate optimization:
- 5e-3: 1.771
- 3e-3: 1.760
- 1e-3: 1.753

Add weight decay to optimizer, 2e-3: 1.733

3. overlap step original: None
- 64: 1.741
    - 25m 44.9s
- 32: 1.753

4. dropout original: 0.1
- 0.2: 1.731
- 0.3: 1.731

5. model architecture original: GRU
- LSTM: 1.835

## Summary of Algorithm
1. Initialize Parameters
- define a neural network class inheriting from `nn.Module`
    - `nn.Embedding` to learn a vector representation for each character
    - `nn.GRU` to model sequential dependencies between characters
    - `nn.Linear` to map hidden states to vocab scores (logits)
- layers used:
    - `Embedding(vocab_size, embedding_dim)`
    - `GRU(embedding_dim, hidden_dim, num_layers)`
    - `Linear(hidden_dim, vocab_size)`
2. Forward Pass
- call model(x) which internally runs forward(x, h=None)
- sequence of operations:
    1. input: x is a batch of token IDs of shape [B, T]
    2. embedding layer: maps IDs -> vectors [B, T, E]
    3. GRU layer: processes vectors sequentially -> [B, T, H]
    4. linear layer: converts GRU output to logits -> [B, T, V]
- the output logits represent unnormalized scores for each character in the vocabulary at every time step
3. Loss Calculation
- use `nn.CrossEntropyLoss()` to compute next-character prediction loss
- compares predicted logits ([B*T, V]) against targest token IDs ([B*T])
4. Backprogagation
- call loss.backward() to compute gradients of all parameters
- PyTorch builds and tracks the computation graph automatically
- gradients are stored in .grad attributes of each parameter
5. Parameter Update
- call `optimizer.step()` (using `torch.optim.Adam`) to apply parameter updates
- clear previous gradients with `optimizer.zero_grad()` or `zero_grad(set_to_none=True)`
6. Repeat
- iterate over many epochs:
    - for each batch, preform forward -> loss -> backward -> update
    - occasionally sample generate text to check training quality
    - evaluate validation loss and save checkpoint if improved


### New Terms and Concepts
- Recurrent Neural Network (RNN)
    - a type of neural network designed to handle sequential data by maintaining a hidden state that evolves over time steps
    - unlike feedforward networks, RNNs can "remember" previous inputs to make better predictions for the current input
- Gated Recurrent Unit (GRU)
    - a variant of RNN that uses gates to control how much of the past information is kept or forgotten
    - helps solve the vanishing gradient problem and trains more efficiently
    - each step updates a hidden state based on the previous hidden state and current input
- Long Short-Term Memory (LSTM)
    - a variant of RNN that uses three gates (input, forget, output) and a cell state to manage long-term dependencies
    - similar to GRUs but slightly more complex
- Embedding Layer
    - learns a dense vector representation (embedding) for each token in the vocab
    - replaces one-hot vectors for characters
        - has vector length of V
        - all values are 0 except for one 1 at the index of the character
- Sequence Length
    - the number of characters in each training sample
- Batch Size
    - the number of sequences processed in parallel during training
- Vocabulary
    - the total number of unique characters in the dataset
- Bits Per Character (BPC)
    - a measurement of how well the model predicts each character, interpreted in bits
    - lower BPC means better predictions
    - `BPC = loss (in nats) / ln(2)`
- Gradient Clipping
    - a technique used to prevent exploding gradients in RNNs
    - limits the overall norm of the 
- Temperature
    - a sampling parameter that controls the randomness of predictions during text generation
        - high temperature (> 1.0): more random
        - low temperature (< 1.0): more conservative
- Top-k and Top-p Sampling
    - techniques used during generation to restrict sampling:
        - top-k: only keep k most like characters
        - 

- GRU
- LSTM
- embedding layer
- logits

- Unnormalized vs Normalized Outputs
    - unnormalized values = logits
        - the output of the final linear layer
        - raw scores - they can be any real number 
        - not yet probabilites
    - normalized values = probabilities
        - after applying softmax to logits
        - all values between 0 and 1
        - sum to 1 for each prediction
    

- token - the smallest unit of text that a model processes as a single element
    - can be whatever segmentation you choose
    - in this RNN, tokens are individual characters from our training text

    1. text -> tokens
    - "R" -> ID 21, "O" -> ID 14, etc.
    2. tokens -> embeddings
    - each ID is mapped to an embedding vector (size = embedding_dim)
    3. embeddings -> RNN
    - the RNN processes them one by one, updating the hidden state
    4. RNN output -> probabilites over tokens
    - softmax gives probabilities for each token in the vocabulary as the next character

https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
- `nn.GRU`
    - a GRU keeps a hidden state h_t and updates it with two gates:
    
    GRU operations

    1. reset gate: $r_t = \sigma(W_r * [h_{t - 1}, x_t])$
    - determines how much of the previous hidden state h_(t - 1) should be forgotten
        - increases when the current char should be interpreted through what just came (e.g. prefixes)
        - lowers when a new topic appears (stop consulting old context)

    2. update gate: $z_t = \sigma(W_z * [h_{t - 1}, x_t])$
    - determines how much of the new information x_t should be used to update the hidden state
        - increases when the broader state should persist across characters
        - lowers when the new input should overwrie the old memory (e.g. negation)


- `seq_len` = 128
    - input: 128 characters from the text
    - output: the next 128 characters (each shifted by 1 position)

- `batch_size` = 128
    - batch size: the number of training examples process together on one forward and backward pass
    - 128 sequences

        - Input tensor to the model for one batch has shape:
            - [B, T] = [128, 128]

- `nn.Embedding(vocab_size, embedding_dim)`
    - character embedding vector - numeric representation of a character
        - `vocab_size`: number of unique characters
            - depends on the dataset
        - `embedding_dim`: size of the vector for each character
            - the larger, the more detailed representation of each character
    
    V (vocab size) = 3: ids {0, 1, 2}
    E (embedding_dim) = 4


- hidden state - the model's memory
    - `hidden_dim`: how many features (neurons) the hdden state has at each time step
        - how much information it can store in its memory

- `grad_clip`
    - gradient clipping - a technique to limit the size of gradients during backpropagation
        - if they get too large, they cause exploding gradients

- `temperature` - controls how random or confident the model's predictions are when generating text
    - lower t < 1: sharper distribution (model more confident)
    - higher t > 1: flatter distribution (model less confident)

- `top_k`
    - softmax gives probabilities for every character in the vocabulary and many of these have very low probability

- `top_p`
    - selectd from the smallest set of tokens whose cumulative probability is at least p
    - adaptive compared to top k
        - model is confident: few tokens are considered
        - model is uncertain: moke tokens are considered

### Letter Notations
- B - batch size, 128
- T - sequence length, 128
- E - embedding dimension, 256
- H - hidden size, 256
    - number of features in the RNN hidden state
- L - number of layers, 2
    - stacked recurrent layers in the RNN
- V - vocab size
    - around 65 for Shakespeare

### Common Tensor Shapes
- inputs (x): [B, T] -> token IDs
- embeddings (self.emb(x)): [B, T, E]
- RNN outputs: [B, T, H]
- logits (self.fc): [B, T, V]

### Shape FLow Diagram
