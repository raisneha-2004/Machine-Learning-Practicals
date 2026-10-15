
import numpy as np

print("=" * 45)
print("       CAUSAL ATTENTION & MASKING LAB")
print("=" * 45)

# Step 1: Take sentence input
sentence = input("\nEnter a sentence: ").strip()

if not sentence:
    print("Please enter a sentence.")
    raise SystemExit

tokens = sentence.split()

if len(tokens) > 12:
    print("Please enter 12 tokens or fewer.")
    raise SystemExit

print("\n1. Tokens:")
print(tokens)

# Step 2: Create simple random embeddings
np.random.seed(7)
X = np.random.randn(len(tokens), 4)

print("\n2. Token Embeddings:")
print(np.round(X, 2))

# Step 3: Calculate scaled dot-product attention scores
scores = X @ X.T / np.sqrt(X.shape[1])

print("\n3. Raw Attention Scores:")
print(np.round(scores, 2))


# Step 4: Define softmax function
def softmax(matrix):
    matrix = matrix - np.max(matrix, axis=1, keepdims=True)
    exp_matrix = np.exp(matrix)
    return exp_matrix / exp_matrix.sum(axis=1, keepdims=True)


# Step 5: Normal self-attention
normal_attention = softmax(scores)

print("\n4. Normal Self-Attention:")
print(np.round(normal_attention, 2))


# Step 6: Create causal mask
# True means that future position is blocked.
mask = np.triu(
    np.ones(scores.shape, dtype=bool),
    k=1
)

print("\n5. Causal Mask (1 = blocked):")
print(mask.astype(int))


# Step 7: Apply mask before softmax
masked_scores = np.where(mask, -np.inf, scores)

causal_attention = softmax(masked_scores)

print("\n6. Causal Attention:")
print(np.round(causal_attention, 2))


# Step 8: Check whether future tokens are blocked
future_attention = causal_attention[mask]

print("\n7. Validation:")
print("Future-token attention is zero:",
      np.allclose(future_attention, 0))

row_sums = causal_attention.sum(axis=1)

print("Causal attention row sums:",
      np.round(row_sums, 2))

print("All row sums equal 1:",
      np.allclose(row_sums, 1.0))

# Step 9: Show token-by-token explanation
print("\n8. What can each token attend to?")

for i, token in enumerate(tokens):
    visible_tokens = tokens[:i + 1]
    print(f"{token} -> {visible_tokens}")

print("\nExperiment completed successfully!")
