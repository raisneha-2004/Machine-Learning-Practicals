import math

tokens = ["I", "like", "tea"]

Q = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
]

K = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
]

V = [
    [10.0, 0.0],
    [0.0, 20.0],
    [10.0, 20.0]
]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def softmax(scores):
    exp_scores = [math.exp(x - max(scores)) for x in scores]
    total = sum(exp_scores)
    return [x / total for x in exp_scores]

for i, query in enumerate(Q):
    scores = [
        dot(query, key) / math.sqrt(len(key))
        for key in K
    ]

    weights = softmax(scores)

    output = [
        sum(weights[j] * V[j][d] for j in range(len(V)))
        for d in range(len(V[0]))
    ]

    print("\nToken:", tokens[i])
    print("Attention weights:", [round(x, 3) for x in weights])
    print("Attention output:", [round(x, 2) for x in output])