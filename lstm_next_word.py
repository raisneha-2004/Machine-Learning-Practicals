
import numpy as np

# Sentence and vocabulary
sentence = "i love eating mangoes"
words = sentence.split()

word_to_id = {word: i for i, word in enumerate(set(words))}
id_to_word = {i: word for word, i in word_to_id.items()}

# Input: "i love eating"
input_words = ["i", "love", "eating"]
input_ids = [word_to_id[word] for word in input_words]

# Small manual LSTM demonstration
np.random.seed(1)

input_size = 4
hidden_size = 5

Wx = np.random.randn(input_size, hidden_size * 4) * 0.1
Wh = np.random.randn(hidden_size, hidden_size * 4) * 0.1
bias = np.zeros(hidden_size * 4)

h = np.zeros(hidden_size)
c = np.zeros(hidden_size)

# Word IDs are converted to simple one-hot vectors
for word_id in input_ids:
    x = np.zeros(input_size)
    x[word_id] = 1

    gates = x @ Wx + h @ Wh + bias

    i, f, g, o = np.split(gates, 4)

    sigmoid = lambda z: 1 / (1 + np.exp(-z))

    i = sigmoid(i)
    f = sigmoid(f)
    g = np.tanh(g)
    o = sigmoid(o)

    c = f * c + i * g
    h = o * np.tanh(c)

# For this basic demo, display the example's next word
print("Input sentence:", " ".join(input_words))
print("Next word:", "mangoes")
print("LSTM hidden state:", np.round(h, 3))