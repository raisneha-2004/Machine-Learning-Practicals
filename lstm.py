# LSTM (Long Short-Term Memory) - basic example
# Idea: same as RNN, but with GATES that decide what to forget, what to add and what to show.
import numpy as np

np.random.seed(2)
words = ["I", "love", "eating", "mangoes"]
size = 3

def sigmoid(z):
    return 1 / (1 + np.exp(-z))                       # gives a value between 0 and 1

embed = {w: np.random.randn(size) for w in words}
# one weight matrix per gate; input = [current word, old hidden state]
Wf, Wi, Wo, Wc = (np.random.randn(size, size * 2) * 0.5 for _ in range(4))

h = np.zeros(size)                                    # short-term memory (hidden state)
c = np.zeros(size)                                    # long-term memory (cell state)
print("LSTM reading the sentence word by word\n")
for word in words:
    z = np.concatenate([embed[word], h])
    f = sigmoid(Wf @ z)                               # forget gate : how much old memory to keep
    i = sigmoid(Wi @ z)                               # input gate  : how much new info to add
    o = sigmoid(Wo @ z)                               # output gate : how much memory to show
    c_new = np.tanh(Wc @ z)                           # new candidate information
    c = f * c + i * c_new                             # update long-term memory
    h = o * np.tanh(c)                                # update short-term memory
    print(f"Word: {word:8s} forget={f.round(2)} input={i.round(2)} output={o.round(2)}")
    print(f"{'':14s}cell state (long-term) = {c.round(3)}\n")

print("Gates are values between 0 and 1: 0 = block completely, 1 = let everything pass.")