import torch

def rms_norm(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Root Mean Square Normalization (RMSNorm) in PyTorch.
    Matches the Swift MLX version from HierarchicalReasoningModel.

    Args:
        x: Input tensor
        epsilon: Small constant to avoid division by zero

    Returns:
        Tensor of same shape and dtype as input
    """
    original_dtype = x.dtype
    x_float = x.to(torch.float32)

    # Mean of squares along last dimension, keep shape
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)

    # Normalize and cast back
    return (x_float * torch.rsqrt(variance + epsilon)).to(original_dtype)

"""if __name__ == "__main__":
    t = torch.randn(2, 5, dtype=torch.float16)  # Test with float16
    out = rms_norm(t)
    print("Input:", t)
    print("Output:", out)
"""
