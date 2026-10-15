
import numpy as np

sentence = input("Enter a sentence: ").strip()

if not sentence:
    print("Please enter a sentence.")
    raise SystemExit

# Step 1: Tokenization
tokens = sentence.split()
print("\n1. Tokens:", tokens)

# Step 2: Simple numerical embeddings
np.random.seed(10)
embeddings = np.random.rand(len(tokens), 4)
print("\n2. Embeddings:\n", np.round(embeddings, 2))

# Step 3: Self-attention
scores = embeddings @ embeddings.T / np.sqrt(4)

# Step 4: Softmax
scores = scores - scores.max(axis=1, keepdims=True)
exp_scores = np.exp(scores)
attention = exp_scores / exp_scores.sum(axis=1, keepdims=True)

print("\n3. Attention weights:\n", np.round(attention, 2))
print("Row sums:", np.round(attention.sum(axis=1), 2))

# Step 5: Weighted value vectors
output = attention @ embeddings
print("\n4. Attention output:\n", np.round(output, 2))

# Step 6: Illustrative next-token probabilities
# These are random demo scores, not a trained decoder.
vocabulary = ["learn", "AI", "today"]
logits = np.random.rand(len(vocabulary))
probabilities = np.exp(logits) / np.exp(logits).sum()

print("\n5. Illustrative next-token probabilities:")
for word, probability in zip(vocabulary, probabilities):
    print(f"{word}: {probability:.2%}")

print("\nDemo completed successfully!")
