"""
Label Smoothing Experiment (NumPy)

Compare:
1. Normal labels (eps = 0)
2. Label smoothing (eps = 0.1)

Metrics: Accuracy, Confidence, ECE, NLL
Dataset: Digits
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

K = 10


def smooth_labels(y, K, eps):
    one_hot = np.eye(K)[y]
    return (1 - eps) * one_hot + eps / K


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def train(Xtr, ytr, eps, seed, hidden=64, epochs=500, lr=0.3):
    rng = np.random.default_rng(seed)

    W1 = rng.normal(0, 0.1, (Xtr.shape[1], hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.1, (hidden, K))
    b2 = np.zeros(K)

    targets = smooth_labels(ytr, K, eps)
    n = len(Xtr)

    for epoch in range(epochs):
        # Forward pass
        z1 = Xtr @ W1 + b1
        h = np.maximum(0, z1)
        probabilities = softmax(h @ W2 + b2)

        # Backward pass
        dz2 = (probabilities - targets) / n

        dW2 = h.T @ dz2
        db2 = dz2.sum(axis=0)

        dh = dz2 @ W2.T
        dh[z1 <= 0] = 0

        dW1 = Xtr.T @ dh
        db1 = dh.sum(axis=0)

        # Update weights
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    def predict(X):
        h = np.maximum(0, X @ W1 + b1)
        return softmax(h @ W2 + b2)

    return predict


def ece(probs, y, bins=10):
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = predictions == y

    edges = np.linspace(0, 1, bins + 1)
    error = 0.0

    for i in range(bins):
        mask = (confidence > edges[i]) & (confidence <= edges[i + 1])

        if mask.any():
            accuracy_in_bin = correct[mask].mean()
            confidence_in_bin = confidence[mask].mean()

            error += mask.mean() * abs(
                accuracy_in_bin - confidence_in_bin
            )

    return error


def evaluate(probs, y):
    return {
        "accuracy": (probs.argmax(axis=1) == y).mean(),
        "confidence": probs.max(axis=1).mean(),
        "ece": ece(probs, y),
        "nll": -np.log(
            probs[np.arange(len(y)), y] + 1e-12
        ).mean(),
    }


if __name__ == "__main__":

    # Check label smoothing formula
    sample = smooth_labels(np.array([0]), 5, 0.1)[0]

    assert np.allclose(
        sample, [0.92, 0.02, 0.02, 0.02, 0.02]
    )
    assert np.isclose(sample.sum(), 1.0)

    print("Formula check passed!")
    print("Correct class = 0.92; other classes = 0.02 each\n")

    # Load and normalize dataset
    X, y = load_digits(return_X_y=True)
    X = X / 16.0

    # Reduced settings for faster execution
    EPS_LIST = [0.0, 0.1]
    SEEDS = range(5)
    EPOCHS = 500

    results = {eps: [] for eps in EPS_LIST}
    saved_predictions = {}

    for seed in SEEDS:
        print(f"Running seed {seed + 1}/{len(SEEDS)}...")

        Xtr, Xte, ytr, yte = train_test_split(
            X,
            y,
            train_size=300,
            random_state=seed,
            stratify=y,
        )

        for eps in EPS_LIST:
            model = train(
                Xtr, ytr, eps, seed, epochs=EPOCHS
            )

            predictions = model(Xte)
            results[eps].append(evaluate(predictions, yte))

            if seed == 0:
                saved_predictions[eps] = (predictions, yte)

    # Print average results
    print("\nAVERAGE RESULTS")
    print("-" * 62)
    print(
        f"{'Model':<22}{'Accuracy':>10}"
        f"{'Confidence':>12}{'ECE':>8}{'NLL':>8}"
    )

    for eps in EPS_LIST:
        average = {
            key: np.mean([result[key] for result in results[eps]])
            for key in results[eps][0]
        }

        name = "Normal labels" if eps == 0 else "Label smoothing"

        print(
            f"{name:<22}"
            f"{average['accuracy']:>10.3f}"
            f"{average['confidence']:>12.3f}"
            f"{average['ece']:>8.3f}"
            f"{average['nll']:>8.3f}"
        )

    # Create graphs
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    colors = ["#64748b", "#38bdf8"]

    for eps, color in zip(EPS_LIST, colors):
        probs, y_test = saved_predictions[eps]

        label = "Normal labels" if eps == 0 else "Label smoothing"

        ax[0].hist(
            probs.max(axis=1),
            bins=20,
            range=(0, 1),
            alpha=0.6,
            color=color,
            label=label,
        )

    ax[0].set_title("Model Confidence")
    ax[0].set_xlabel("Maximum predicted probability")
    ax[0].set_ylabel("Number of test images")
    ax[0].legend()

    for eps, color in zip(EPS_LIST, colors):
        probs, y_test = saved_predictions[eps]

        confidence = probs.max(axis=1)
        correct = probs.argmax(axis=1) == y_test

        edges = np.linspace(0, 1, 11)
        x_values, y_values = [], []

        for low, high in zip(edges[:-1], edges[1:]):
            mask = (confidence > low) & (confidence <= high)

            if mask.sum() >= 3:
                x_values.append(confidence[mask].mean())
                y_values.append(correct[mask].mean())

        label = "Normal labels" if eps == 0 else "Label smoothing"

        ax[1].plot(
            x_values,
            y_values,
            "o-",
            color=color,
            label=label,
        )

    ax[1].plot(
        [0, 1], [0, 1], "--",
        color="gray",
        label="Perfect calibration",
    )

    ax[1].set_title("Confidence vs Actual Accuracy")
    ax[1].set_xlabel("Average confidence")
    ax[1].set_ylabel("Actual accuracy")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig("label_smoothing_results.png", dpi=130)
    plt.close()

    print("\nExperiment completed successfully!")
    print("Graph saved as label_smoothing_results.png")