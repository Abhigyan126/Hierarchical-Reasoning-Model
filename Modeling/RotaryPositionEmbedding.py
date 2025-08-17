import torch
import torch.nn as nn

class RotaryPositionEmbedding(nn.Module):

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, dim: int, max_length: int, base: float = 10000.0, dtype=torch.float32):
        """
        Rotary Position Embedding (RoPE) in PyTorch, matching the Swift MLX version.

        Args:
            dim: Embedding dimension (must be even)
            max_length: Maximum sequence length
            base: Frequency base (usually 10000.0)
            dtype: torch dtype
        """
        super().__init__()

        # 1 / (base^(i/dim))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=dtype) / dim))

        # t = [0, 1, 2, ..., max_length-1]
        t = torch.arange(max_length, dtype=dtype)

        # Outer product: [seq_len, dim/2]
        freqs = torch.outer(t, inv_freq)

        # Duplicate for cos/sin interleaving: [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1).unsqueeze(-2)  # shape: [seq_len, 1, dim]

        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension in half, swap halves, and negate the second.
        Matches Swift's rotateHalf().
        """
        half_dim = x.size(-1) // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor.
        x shape: [batch, seq_len, heads, head_dim] or compatible with broadcasting.
        """
        # Assumes seq_len matches precomputed cos/sin
        return (x * self.cos) + (self.rotate_half(x) * self.sin)


"""# TESTING
rope = RotaryPositionEmbedding(dim=8, max_length=16)
x = torch.randn(2, 16, 4, 8)
y = rope(x)
print(y.shape)
"""
