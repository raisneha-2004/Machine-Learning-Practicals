
import numpy as np

# Step 1: Encoder outputs
words = ["I", "love", "Python"]

encoder_outputs = np.array([
    [1, 0],
    [0, 2],
    [1, 1]
], dtype=float)

# Step 2: Create a query
query = np.array([0, 1], dtype=float)

# Step 3: Calculate attention scores
scores = encoder_outputs @ query

# Step 4: Calculate attention weights using softmax
exp_scores = np.exp(scores)
attention_weights = exp_scores / exp_scores.sum()

# Step 5: Calculate context vector
context_vector = attention_weights @ encoder_outputs

# Step 6: Display results
print("Input Words:", words)

print("\nAttention Scores:")
print(scores)

print("\nAttention Weights:")
print(np.round(attention_weights, 2))

print("\nContext Vector:")
print(np.round(context_vector, 2))