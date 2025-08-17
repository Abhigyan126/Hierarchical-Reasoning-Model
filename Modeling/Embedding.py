import torch
import torch.nn as nn
from .TruckNormalInit import trunc_normal_init

class Embedding(nn.Module):
    def __init__(self, vocabSize, dim, initStd, dtype=torch.float32, key=None):
        """
        Equivalent to the Swift Embedding module.

        vocabSize : int - size of the vocabulary
        dim       : int - embedding dimension
        initStd   : float - std deviation for truncated normal init
        dtype     : torch.dtype - data type (default: torch.float32)
        key       : optional, for seeding (like MLX key)
        """
        super().__init__()

        # Optional seed for reproducibility
        if key is not None:
            torch.manual_seed(key)

        # Create embedding matrix
        self.embeddings = nn.Embedding(vocabSize, dim, dtype=dtype)

        # Apply truncated normal init
        trunc_normal_init(self.embeddings.weight, std=initStd)

    def forward(self, x):
        return self.embeddings(x)
"""
# TESTING
# Create embedding layer
vocab_size = 20
embedding_dim = 5
init_std = 0.02
embed = Embedding(vocabSize=vocab_size, dim=embedding_dim, initStd=init_std, key=42)

# Create sample input indices
sample_indices = torch.tensor([0, 1, 5, 10, 15])

# Get embeddings
output = embed(sample_indices)

# Print results
print("Sample indices:", sample_indices.tolist())
print("Embedding matrix shape:", embed.embeddings.weight.shape)
print("\nSample embeddings:\n", output)
"""
