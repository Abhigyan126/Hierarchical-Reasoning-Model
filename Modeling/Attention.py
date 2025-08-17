import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, dim, headDim, numHeads, keyValueHeadsPerHead, dtype=torch.float32, key=None):
        super().__init__()

        self.dim = dim
        self.headDim = headDim
        self.numHeads = numHeads
        self.outputSize = headDim * numHeads
        self.keyValueHeadsPerHead = keyValueHeadsPerHead
        self.numKeyValueHeads = numHeads * keyValueHeadsPerHead

        if key is not None:
            torch.manual_seed(key)

        # Might be replaced with Linear from Linear.py
        # QKV projection: out_dim = (numHeads + 2 * numKeyValueHeads) * headDim
        self.qkvProj = nn.Linear(
            dim,
            (numHeads + 2 * self.numKeyValueHeads) * headDim,
            bias=False,
            dtype=dtype
        )

        # Output projection
        self.outProj = nn.Linear(
            self.outputSize,
            dim,
            bias=False,
            dtype=dtype
        )

    def forward(self, x, rotary_position_embedding=None):
        batchSize, seqLen, _ = x.shape

        # Project input to QKV
        qkv = self.qkvProj(x)  # [B, L, (numHeads + 2*numKVHeads) * headDim]
        qkv = qkv.view(batchSize, seqLen, self.numHeads + 2 * self.numKeyValueHeads, self.headDim)

        # Split Q, K, V
        query = qkv[:, :, :self.numHeads]  # [B, L, numHeads, headDim]
        key = qkv[:, :, self.numHeads:self.numHeads + self.numKeyValueHeads]
        value = qkv[:, :, self.numHeads + self.numKeyValueHeads:]

        # Apply rotary position embedding if given
        if rotary_position_embedding is not None:
            query = rotary_position_embedding(query)
            key = rotary_position_embedding(key)

        # Reshape query: [B, L, numKVHeads, KVHeadsPerHead, headDim] → [B, numKVHeads, KVHeadsPerHead, L, headDim]
        query = query.view(batchSize, seqLen, self.numKeyValueHeads, self.keyValueHeadsPerHead, self.headDim)
        query = query.permute(0, 2, 3, 1, 4)  # [B, numKVHeads, KVHeadsPerHead, L, headDim]

        # Reshape key: [B, L, numKVHeads, headDim] → [B, numKVHeads, 1, L, headDim]
        key = key.permute(0, 2, 1, 3).unsqueeze(2)

        # Reshape value: [B, L, numKVHeads, headDim] → [B, numKVHeads, 1, L, headDim]
        value = value.permute(0, 2, 1, 3).unsqueeze(2)

        # Attention scores: [B, numKVHeads, KVHeadsPerHead, L, headDim] x [B, numKVHeads, 1, headDim, L]
        attn_logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.headDim)

        # Softmax over last dimension
        attn_weights = F.softmax(attn_logits.float(), dim=-1).type_as(attn_logits)

        # Weighted sum of values
        combined = torch.matmul(attn_weights, value)  # [B, numKVHeads, KVHeadsPerHead, L, headDim]

        # Rearrange back to [B, L, dim]
        combined = combined.permute(0, 3, 1, 2, 4).reshape(batchSize, seqLen, self.dim)

        return self.outProj(combined)

"""if __name__ == "__main__":
    torch.manual_seed(0)

    attn = Attention(dim=64, headDim=16, numHeads=4, keyValueHeadsPerHead=1)
    x = torch.randn(2, 10, 64)  # batch=2, seq_len=10, dim=64
    out = attn(x)
    print(out.shape)  # torch.Size([2, 10, 64])
"""
