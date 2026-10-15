# Transformer Pipeline Visualizer 🧠

A beginner-friendly Python project that demonstrates the basic steps involved in processing text using concepts inspired by Transformer architectures.

The project shows how a sentence can be converted into tokens and numerical embeddings, processed through self-attention, and used to demonstrate next-token probability calculations.

## 📌 Project Overview

Transformers are deep learning architectures widely used in modern Natural Language Processing (NLP) applications, including Large Language Models (LLMs).

I created this project to understand the basic mathematical operations behind the Transformer pipeline through a simple, hands-on Python implementation.

Instead of using a pretrained model, this project uses NumPy to demonstrate the core calculations step by step.

## ✨ Features

- **Tokenization:** Splits the input sentence into individual words.
- **Embeddings:** Represents each token using a numerical vector.
- **Self-Attention:** Calculates relationships between tokens using scaled dot-product scores.
- **Softmax:** Converts attention scores into probabilities.
- **Attention Output:** Combines the token vectors using their attention weights.
- **Next-Token Probability Demo:** Displays illustrative probabilities for a small vocabulary.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical calculations and matrix operations |
| Softmax | Converts scores into probability distributions |
| Self-Attention | Demonstrates how tokens can attend to other tokens |

## ⚙️ How It Works

### 1. Tokenization

The input sentence is split into individual words.

Example:

Input: `I love learning AI`

Output: `['I', 'love', 'learning', 'AI']`

### 2. Embeddings

Each token is represented by a numerical vector. In this project, the vectors are generated randomly to demonstrate how text can be represented numerically.

### 3. Self-Attention

The program calculates scaled dot-product attention scores using the embedding matrix.

Formula:

`Attention Scores = (Q × Kᵀ) / √dₖ`

For simplicity, the same embedding matrix is used to represent the queries (Q), keys (K), and values (V).

### 4. Softmax

Softmax converts the attention scores into probabilities. Each row of the attention matrix sums to approximately 1.

### 5. Attention Output

The attention weights are multiplied by the embedding matrix to calculate a weighted output representation for each token.

### 6. Next-Token Probability Demo

The program generates illustrative probabilities for a small vocabulary using random scores and softmax.

These probabilities are included to demonstrate the mathematical idea of probability normalization. They do not predict the actual next word in the sentence.

## 📁 Project Structure

```text
transformer-pipeline/
├── main.py
└── README.md
```

## 🚀 How to Run

**Step 1: Install NumPy**

```bash
pip install numpy
```

**Step 2: Run the program**

```bash
python main.py
```

**Step 3: Enter a sentence**

For example:

```text
I love learning AI
```

The program displays the tokens, embeddings, attention weights, attention output, and illustrative next-token probabilities.

## 🎯 Learning Outcomes

Through this project, I learned:

- How text is split into tokens.
- How words can be represented as numerical vectors.
- How matrix multiplication is used in self-attention.
- How softmax creates probability distributions.
- How attention weights combine information from different tokens.
- How to implement basic Transformer-related calculations using NumPy.

## ⚠️ Limitations

This is a learning demonstration, not a complete Transformer model.

- Embeddings are randomly generated, not learned from data.
- Queries, keys, and values use the same embedding matrix.
- There is no trained encoder, decoder, or feed-forward network.
- Next-token probabilities use random scores and are not meaningful language predictions.

## 🔮 Future Improvements

- Implement separate Query, Key, and Value projection matrices.
- Add positional encoding.
- Visualize attention weights using a heatmap.
- Extend the project to multiple attention heads.
- Compare the implementation with a pretrained Transformer model.

## 👩‍💻 Author

Created as a hands-on learning project to explore the mathematical foundations of self-attention and Transformer-based language processing.
