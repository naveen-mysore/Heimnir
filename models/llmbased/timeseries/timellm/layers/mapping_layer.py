import torch
import torch.nn as nn


class MappingLayer(nn.Module):
    def __init__(self, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.mapping_layer = self.mapping_layer.to(torch.float32)

    def forward(self, word_embeddings):
        word_embeddings = word_embeddings.to(torch.float32)
        source_embeddings = self.mapping_layer(word_embeddings)
        return source_embeddings
