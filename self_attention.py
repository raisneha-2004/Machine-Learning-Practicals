
import numpy as np

# Step 1: Input sentence
words = ["I", "love", "Python"]

# Step 2: Convert words into simple vectors
X = np.array([
    [1, 0],
    [0, 1],
    [1, 1]
], dtype=float)

# Step 3: Create Query, Key and Value
Q = X
K = X
V = X

# Step 4: Calculate attention scores
scores = Q @ K.T

# Step 5: Apply softmax
exp_scores = np.exp(scores)
attention_weights = exp_scores / exp_scores.sum(
    axis=1, keepdims=True
)

# Step 6: Calculate final output
output = attention_weights @ V

# Step 7: Display results
print("Input Sentence:", " ".join(words))

print("\nAttention Scores:")
print(scores)

print("\nAttention Weights:")
print(np.round(attention_weights, 2))

print("\nSelf-Attention Output:")
print(np.round(output, 2))