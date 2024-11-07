import torch
import torch.nn as nn
from math import sqrt

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        # scaled dot-product attention mechanism
        B, L, H, E = target_embedding.shape

        # Calculate the attention scores using Einstein summation notation
        # blhe: target_embedding with shape [B, L, H, E]
        # she: source_embedding with shape [S, H, E]
        # bhls: scores with shape [B, H, L, S]
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        scale = 1. / sqrt(E)
        attention_matrix = self.dropout(torch.softmax(scores * scale, dim=-1))

        reprogramming_embedding = torch.einsum("bhls,she->blhe", attention_matrix, value_embedding)

        return reprogramming_embedding