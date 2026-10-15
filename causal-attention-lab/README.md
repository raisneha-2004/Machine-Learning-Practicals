# Causal Attention & Masking Lab 🔍

A beginner-friendly Python project that demonstrates the difference between normal self-attention and causal self-attention using NumPy.

## Project Overview

Causal masking is an important concept in autoregressive Transformer decoders. It prevents a token from attending to future tokens during attention calculation.

I built this project to understand how attention scores are calculated, how masking is applied before softmax, and how the resulting attention probabilities change.

## Features

- Sentence tokenization
- Random numerical embeddings
- Scaled dot-product attention
- Normal self-attention calculation
- Upper-triangular causal mask
- Softmax probability calculation
- Validation of attention row sums
- Token-by-token visibility explanation

## Technologies Used

- Python
- NumPy
- Matrix multiplication
- Softmax
- Causal masking

## How It Works

1. **Tokenization:** Splits the input sentence into words.
2. **Embeddings:** Creates random numerical vectors for tokens.
3. **Attention Scores:** Calculates scaled dot-product scores.
4. **Normal Attention:** Applies softmax without masking.
5. **Causal Masking:** Blocks future-token positions by replacing their scores with negative infinity.
6. **Masked Softmax:** Converts the remaining scores into probabilities.
7. **Validation:** Checks that future positions have zero attention and every row sums to approximately 1.

## Project Structure

```text
causal-attention-lab/
├── main.py
└── README.md
```

## How to Run

Install NumPy:

```bash
python -m pip install numpy
```

Run the program:

```bash
python main.py
```

Enter a sentence when prompted.

## Learning Outcomes

Through this project, I learned:

- The difference between normal and causal self-attention.
- Why future tokens must be masked in autoregressive decoding.
- How to apply a causal mask before softmax.
- How attention probabilities are normalized.
- How to validate attention calculations using NumPy.

## Limitations

This is a simplified learning demonstration, not a trained Transformer model. It uses random embeddings and the same matrix for queries, keys, and values. It does not include learned projections, positional encodings, or a complete decoder.

## Future Improvements

- Visualize attention matrices as heatmaps.
- Add an interactive mask toggle.
- Compare attention outputs for different sentences.
- Extend the demo to multi-head attention.

## Author

Created as a hands-on learning project to explore causal attention and masking in Transformer architectures.
