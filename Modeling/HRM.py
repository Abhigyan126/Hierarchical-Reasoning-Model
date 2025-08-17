import math
import torch
import torch.nn as nn

from .Linear import Linear
from .Attention import Attention
from .RMSNorm import rms_norm
from .TruckNormalInit import trunc_normal_init
from .Embedding import Embedding
from .RotaryPositionEmbedding import RotaryPositionEmbedding
from .SwiGLU import SwiGLU


class HRMACTModelConfig:
    class TransformerConfig:
        def __init__(self, num_layers, hidden_size, num_heads, expansion,
                     norm_epsilon=1e-5, rope_theta=10000.0):
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.expansion = expansion
            self.norm_epsilon = norm_epsilon
            self.rope_theta = rope_theta

    class ACTConfig:
        def __init__(self, halt_max_steps, halt_exploration_probability):
            self.halt_max_steps = halt_max_steps
            self.halt_exploration_probability = halt_exploration_probability

    def __init__(self, seq_len, vocab_size,
                 high_level_cycles, low_level_cycles,
                 transformers: TransformerConfig,
                 act: ACTConfig,
                 dtype=torch.bfloat16):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.high_level_cycles = high_level_cycles
        self.low_level_cycles = low_level_cycles
        self.transformers = transformers
        self.act = act
        self.dtype = dtype


class HRMACTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, expansion, norm_epsilon, dtype=torch.float32):
        super().__init__()
        self.self_attn = Attention(
            dim=hidden_size,
            headDim=hidden_size // num_heads,
            numHeads=num_heads,
            keyValueHeadsPerHead=1,
            dtype=dtype
        )
        self.mlp = SwiGLU(dim=hidden_size, expansion=expansion, dtype=dtype)
        self.norm_epsilon = norm_epsilon

    def forward(self, x, rotary_position_embedding=None):
        x = rms_norm(x + self.self_attn(x, rotary_position_embedding=rotary_position_embedding),
                     epsilon=self.norm_epsilon)
        x = rms_norm(x + self.mlp(x), epsilon=self.norm_epsilon)
        return x


class HRMACTReasoner(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, expansion, norm_epsilon, dtype=torch.float32):
        super().__init__()
        self.blocks = nn.ModuleList([
            HRMACTBlock(hidden_size, num_heads, expansion, norm_epsilon, dtype=dtype)
            for _ in range(num_layers)
        ])

    def forward(self, hidden_state, input_injection, rotary_position_embedding=None):
        hidden_state = hidden_state + input_injection
        for block in self.blocks:
            hidden_state = block(hidden_state, rotary_position_embedding=rotary_position_embedding)
        return hidden_state


class HRMACTInner(nn.Module):
    class HiddenStates:
        def __init__(self, high_level, low_level):
            self.high_level = high_level
            self.low_level = low_level
        def inner_state(self):
            return [self.high_level, self.low_level]

    class Output:
        def __init__(self, hidden_states, output, qact_halt, qact_continue):
            self.hidden_states = hidden_states
            self.output = output
            self.qact_halt = qact_halt
            self.qact_continue = qact_continue

    def __init__(self, config: HRMACTModelConfig):
        super().__init__()
        self.config = config

        self.cls_token = trunc_normal_init(
            (config.transformers.hidden_size,),
            std=1.0 / math.sqrt(config.transformers.hidden_size),
            dtype=config.dtype
        )

        self.input_embedding = Embedding(
            vocabSize=config.vocab_size,
            dim=config.transformers.hidden_size,
            initStd=1.0 / math.sqrt(config.transformers.hidden_size),
            dtype=config.dtype
        )

        self.output_head = Linear(
            inDim=config.transformers.hidden_size,
            outDim=config.vocab_size,
            bias=False,
            dtype=config.dtype
        )

        self.qact_head = Linear(
            inDim=config.transformers.hidden_size,
            outDim=2,
            bias=True,
            dtype=config.dtype
        )
        nn.init.constant_(self.qact_head.bias, -5)

        self.rotary_emb = RotaryPositionEmbedding(
            dim=config.transformers.hidden_size // config.transformers.num_heads,
            max_length=config.seq_len + 1,
            base=config.transformers.rope_theta,
            dtype=config.dtype
        )

        self.high_level_reasoner = HRMACTReasoner(
            num_layers=config.transformers.num_layers,
            hidden_size=config.transformers.hidden_size,
            num_heads=config.transformers.num_heads,
            expansion=config.transformers.expansion,
            norm_epsilon=config.transformers.norm_epsilon,
            dtype=config.dtype
        )

        self.low_level_reasoner = HRMACTReasoner(
            num_layers=config.transformers.num_layers,
            hidden_size=config.transformers.hidden_size,
            num_heads=config.transformers.num_heads,
            expansion=config.transformers.expansion,
            norm_epsilon=config.transformers.norm_epsilon,
            dtype=config.dtype
        )

        self.initial_high_level = trunc_normal_init(
            (config.transformers.hidden_size,), std=1.0, dtype=config.dtype
        )
        self.initial_low_level = trunc_normal_init(
            (config.transformers.hidden_size,), std=1.0, dtype=config.dtype
        )

    def initial_hidden_states(self):
        return HRMACTInner.HiddenStates(
            high_level=self.initial_high_level.clone(),
            low_level=self.initial_low_level.clone()
        )

    def forward(self, hidden_states: HiddenStates, inputs):
        cls_tokens = self.cls_token.unsqueeze(0).unsqueeze(0).expand(inputs.size(0), -1, -1)
        input_embeddings = torch.cat([cls_tokens, self.input_embedding(inputs)], dim=1) * math.sqrt(
            self.config.transformers.hidden_size
        )

        low_level_z = hidden_states.low_level
        high_level_z = hidden_states.high_level

        for cycle in range(1, self.config.high_level_cycles * self.config.low_level_cycles):
            low_level_z = self.low_level_reasoner(
                hidden_state=low_level_z,
                input_injection=high_level_z + input_embeddings,
                rotary_position_embedding=self.rotary_emb
            )
            if cycle % self.config.low_level_cycles == 0:
                high_level_z = self.high_level_reasoner(
                    hidden_state=high_level_z,
                    input_injection=low_level_z,
                    rotary_position_embedding=self.rotary_emb
                )

        low_level_z = self.low_level_reasoner(
            hidden_state=low_level_z,
            input_injection=high_level_z + input_embeddings,
            rotary_position_embedding=self.rotary_emb
        )
        high_level_z = self.high_level_reasoner(
            hidden_state=high_level_z,
            input_injection=low_level_z,
            rotary_position_embedding=self.rotary_emb
        )

        output_logits = self.output_head(high_level_z[:, 1:])
        qact_logits = self.qact_head(high_level_z[:, 0])

        return HRMACTInner.Output(
            hidden_states=self.HiddenStates(high_level=high_level_z, low_level=low_level_z),
            output=output_logits,
            qact_halt=qact_logits[:, 0],
            qact_continue=qact_logits[:, 1]
        )
