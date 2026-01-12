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
