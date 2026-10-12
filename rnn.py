# RNN (Recurrent Neural Network) - basic example
# Idea: read words ONE BY ONE and keep a "memory" (hidden state) that updates at every step.
import numpy as np

np.random.seed(1)
words = ["I", "love", "eating", "mangoes"]
size = 3                                              # small size so output is easy to read

embed = {w: np.random.randn(size) for w in words}     # each word -> numbers
Wx = np.random.randn(size, size) * 0.5                # weights for the current word
Wh = np.random.randn(size, size) * 0.5                # weights for the old memory
b = np.zeros(size)

h = np.zeros(size)                                    # memory starts empty
print("RNN reading the sentence word by word\n")
for word in words:
    x = embed[word]
    h = np.tanh(Wx @ x + Wh @ h + b)                  # new memory = f(current word, old memory)
    print(f"Word: {word:8s} -> memory (hidden state): {h.round(3)}")

print("\nThe last memory summarizes the whole sentence.")
print("Problem: for long sentences, old information fades (vanishing gradient). LSTM fixes this.")