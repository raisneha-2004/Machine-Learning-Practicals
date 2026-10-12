
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hashlib

st.set_page_config(
    page_title="Self-Attention Visualizer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Self-Attention Visualizer")
st.write(
    "Enter a sentence to see how Self-Attention "
    "calculates attention weights between words."
)

sentence = st.text_input(
    "Enter your sentence:",
    value="I love machine learning",
    placeholder="Type your own sentence here..."
)

words = sentence.lower().strip().split()

if not sentence.strip():
    st.info("Please enter a sentence.")

elif len(words) < 2:
    st.warning("Please enter at least two words.")

elif len(words) > 15:
    st.warning("Please enter at most 15 words.")

else:
    # Create a repeatable numerical vector for each word.
    # A word gets the same vector whenever it appears.
    def word_vector(word, dimension=8):
        seed_bytes = hashlib.sha256(
            word.encode("utf-8")
        ).digest()

        seed = int.from_bytes(seed_bytes[:4], "little")
        rng = np.random.default_rng(seed)

        return rng.random(dimension)

    embeddings = np.array([
        word_vector(word) for word in words
    ])

    # Fixed weight matrices for this educational demo
    rng = np.random.default_rng(10)

    Wq = rng.normal(0, 0.5, (8, 8))
    Wk = rng.normal(0, 0.5, (8, 8))
    Wv = rng.normal(0, 0.5, (8, 8))

    # Query, Key and Value
    Q = embeddings @ Wq
    K = embeddings @ Wk
    V = embeddings @ Wv

    # Scaled dot-product attention
    scores = Q @ K.T / np.sqrt(K.shape[1])

    # Softmax converts scores into attention weights
    scores = scores - np.max(
        scores, axis=1, keepdims=True
    )

    exp_scores = np.exp(scores)

    attention_weights = exp_scores / np.sum(
        exp_scores, axis=1, keepdims=True
    )

    # Final attention output
    output = attention_weights @ V

    st.subheader("1. Words in Your Sentence")
    st.write(" | ".join(words))

    st.subheader("2. Self-Attention Heatmap")
    st.caption(
        "Rows represent words paying attention. "
        "Columns represent words receiving attention."
    )

    # Compact heatmap
    fig, ax = plt.subplots(figsize=(5, 3))

    heatmap = ax.imshow(
        attention_weights,
        cmap="Blues",
        vmin=0,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(
        words, rotation=35, ha="right", fontsize=8
    )

    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=8)

    ax.set_xlabel("Words Being Attended To", fontsize=9)
    ax.set_ylabel("Word Paying Attention", fontsize=9)
    ax.set_title("Attention Weights", fontsize=11)

    for i in range(len(words)):
        for j in range(len(words)):
            ax.text(
                j, i,
                f"{attention_weights[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black"
            )

    fig.colorbar(
        heatmap,
        ax=ax,
        label="Attention Weight",
        fraction=0.046,
        pad=0.04
    )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.subheader("3. Attention Weights Table")

    table = pd.DataFrame(
        attention_weights,
        index=words,
        columns=words
    )

    st.dataframe(table.round(3))

    st.subheader("4. Output")
    st.write("Self-Attention output shape:", output.shape)

    st.success(
        "Self-Attention calculation completed successfully!"
    )