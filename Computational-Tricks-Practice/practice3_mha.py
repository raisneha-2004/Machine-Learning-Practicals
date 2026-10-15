from sympy import python


python
import torch
import torch.nn as nn

torch.manual_seed(7)

sequence = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0]
])

sequence = sequence.unsqueeze(0)

attention = nn.MultiheadAttention(
    embed_dim=4,
    num_heads=2,
    batch_first=True
)

output, weights = attention(
    query=sequence,
    key=sequence,
    value=sequence,
    need_weights=True
)

print("Multi-Head Attention Demo")
print("Input shape:", tuple(sequence.shape))
print("Output shape:", tuple(output.shape))
print("Attention weights shape:", tuple(weights.shape))

print("\nOutput:")
print(output.detach().round(decimals=3))

print("\nAttention weights:")
print(weights.detach().round(decimals=3))