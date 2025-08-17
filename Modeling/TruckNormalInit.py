import torch
import math

def trunc_normal_init(tensor_or_shape, std=1.0, lower=-2.0, upper=2.0, dtype=None, seed=None):
    """
    Truncated normal initialization, matching Swift's truncNormalInit implementation.

    tensor_or_shape : tuple/list of ints or a torch.Tensor
    std   : standard deviation before truncation
    lower : lower bound (in std units)
    upper : upper bound (in std units)
    dtype : torch.dtype (optional if tensor provided)
    seed  : optional int for reproducibility
    """
    # Determine shape and dtype
    if isinstance(tensor_or_shape, torch.Tensor):
        shape = tensor_or_shape.shape
        dtype = tensor_or_shape.dtype if dtype is None else dtype
    else:
        shape = tuple(tensor_or_shape)
        dtype = torch.float32 if dtype is None else dtype

    if std == 0.0:
        result = torch.zeros(shape, dtype=dtype)
        if isinstance(tensor_or_shape, torch.Tensor):
            with torch.no_grad():
                tensor_or_shape.copy_(result)
            return tensor_or_shape
        return result

    if seed is not None:
        torch.manual_seed(seed)

    sqrt2 = math.sqrt(2.0)
    a = math.erf(lower / sqrt2)
    b = math.erf(upper / sqrt2)
    z = (b - a) / 2

    c = 1.0 / math.sqrt(2 * math.pi)
    pdfU = c * math.exp(-0.5 * lower**2)
    pdfL = c * math.exp(-0.5 * upper**2)
    compStd = std / math.sqrt(
        1.0 - (upper * pdfU - lower * pdfL) / z - ((pdfU - pdfL) / z) ** 2
    )

    # Uniform random values in [a, b]
    u = torch.empty(shape, dtype=dtype).uniform_(a, b)

    # Inverse erf to map uniform to normal quantiles
    result = torch.erfinv(u)

    # Scale and clip
    result = result * (sqrt2 * compStd)
    result = torch.clamp(result, min=lower * compStd, max=upper * compStd)

    # If a tensor was passed, fill it in-place
    if isinstance(tensor_or_shape, torch.Tensor):
        with torch.no_grad():
            tensor_or_shape.copy_(result)
        return tensor_or_shape

    return result
