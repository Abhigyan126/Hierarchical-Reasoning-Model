import torch
import torch.nn as nn
import math
from .TruckNormalInit import trunc_normal_init

class Linear(nn.Module):
    def __init__(self, inDim, outDim, bias=True, dtype=torch.float32, key=None):
        """
        inDim  : int - input dimension
        outDim : int - output dimension
        bias   : bool - whether to include bias term
        dtype  : torch.dtype
        key    : optional int for reproducibility (like MLX key)
        """
        super().__init__()

        if key is not None:
            torch.manual_seed(key)

        # Create parameters
        self.weight = nn.Parameter(torch.empty(inDim, outDim, dtype=dtype))
        std = 1.0 / math.sqrt(inDim)  # same as Swift's init
        trunc_normal_init(self.weight, std=std)

        if bias:
            self.bias = nn.Parameter(torch.zeros(outDim, dtype=dtype))
        else:
            self.register_parameter('bias', None)  # PyTorch official style

    def forward(self, x):
        # Matmul + bias (matmul(x, weight) + bias)
        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

"""if __name__ == "__main__":
    torch.manual_seed(0)
    fc = Linear(8, 4, bias=True)
    x = torch.randn(2, 8)  # batch=2, inDim=8
    out = fc(x)
    print("Output shape:", out.shape)
    print(out)"""
