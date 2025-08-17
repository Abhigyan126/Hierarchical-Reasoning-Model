import torch
import torch.nn as nn
import torch.nn.functional as F
from .Linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, dim: int, expansion: float, dtype=torch.float32, device=None):
        """
        SwiGLU feed-forward block

        Args:
            dim: Input/output dimension
            expansion: Expansion factor for intermediate dimension
            dtype: torch dtype
            device: device to put the module on
        """
        super().__init__()

        # Match: findMultiple(Int(expansion * Float(dim) * 2.0 / 3.0), 256)
        inter_dim_raw = int(expansion * dim * 2.0 / 3.0)
        inter_dim = ((inter_dim_raw + 255) // 256) * 256  # round up to nearest multiple of 256

        # gateUpProj outputs double inter_dim (half for gate, half for up projection)
        self.gate_up_proj = Linear(dim, inter_dim * 2, bias=False, dtype=dtype)
        self.down_proj = Linear(inter_dim, dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split along last dimension into gate and up parts
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

"""# TESTING
if __name__ == "__main__":
    torch.manual_seed(42)

    dim = 16
    expansion = 4.0
    batch_size = 2
    seq_len = 5

    model = SwiGLU(dim=dim, expansion=expansion)
    x = torch.randn(batch_size, seq_len, dim)

    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Sample output:\n", y)"""
