"""
Multi-Head Attention from scratch (NumPy only).

Steps (same as in the lecture notes):
  1. Project the input X into Q, K, V with learned matrices.
  2. Split Q, K, V into `num_heads` smaller parts (one part per head).
  3. Every head does scaled dot-product attention in parallel.
  4. Join (concatenate) the outputs of all heads.
  5. Pass the joined result through a final linear layer.
"""
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)  # for numerical stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q, K, V: (..., seq, d_k). Returns (output, attention_weights)."""
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)  # masked positions get ~0 weight
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def causal_mask(seq_len):
    """True where attention is allowed (a token can see itself and the past)."""
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


class MultiHeadAttention:
    def __init__(self, d_model, num_heads, seed=0):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model, self.h = d_model, num_heads
        self.d_k = d_model // num_heads
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.Wq = rng.normal(0, scale, (d_model, d_model))
        self.Wk = rng.normal(0, scale, (d_model, d_model))
        self.Wv = rng.normal(0, scale, (d_model, d_model))
        self.Wo = rng.normal(0, scale, (d_model, d_model))

    def split_heads(self, x):
        # (batch, seq, d_model) -> (batch, heads, seq, d_k)
        b, s, _ = x.shape
        return x.reshape(b, s, self.h, self.d_k).transpose(0, 2, 1, 3)

    def merge_heads(self, x):
        # (batch, heads, seq, d_k) -> (batch, seq, d_model)  (= concatenation)
        b, _, s, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, s, self.d_model)

    def __call__(self, x, mask=None):
        Q = self.split_heads(x @ self.Wq)
        K = self.split_heads(x @ self.Wk)
        V = self.split_heads(x @ self.Wv)
        out, weights = scaled_dot_product_attention(Q, K, V, mask)
        out = self.merge_heads(out) @ self.Wo
        return out, weights  # weights: (batch, heads, seq, seq)

    def forward_slow(self, x, mask=None):
        """Same maths with an explicit Python loop over heads (used to verify)."""
        outs = []
        for i in range(self.h):
            sl = slice(i * self.d_k, (i + 1) * self.d_k)
            q, k, v = (x @ self.Wq)[..., sl], (x @ self.Wk)[..., sl], (x @ self.Wv)[..., sl]
            o, _ = scaled_dot_product_attention(q, k, v, mask)
            outs.append(o)
        return np.concatenate(outs, axis=-1) @ self.Wo


if __name__ == "__main__":
    batch, seq, d_model, heads = 2, 6, 64, 8
    x = np.random.default_rng(1).normal(size=(batch, seq, d_model))
    mha = MultiHeadAttention(d_model, heads)

    out, w = mha(x)
    print("input  :", x.shape)
    print("output :", out.shape, "(same shape as input)")
    print("weights:", w.shape, "= (batch, heads, seq, seq)")
    print("each head size d_k =", mha.d_k)

    # Checks
    assert out.shape == x.shape
    assert np.allclose(w.sum(-1), 1.0), "attention rows must sum to 1"
    assert np.allclose(out, mha.forward_slow(x)), "vectorised == loop over heads"
    out_m, w_m = mha(x, mask=causal_mask(seq))
    assert np.allclose(np.triu(w_m[0, 0], k=1), 0, atol=1e-6), "causal mask hides the future"
    print("All checks passed: shapes, rows sum to 1, loop == vectorised, causal mask works.")