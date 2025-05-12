import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, in_dim: int, max_len: int = 5000):

        super().__init__()
        

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, in_dim, 2).float() * (-math.log(10000.0) / in_dim))
        

        pe = torch.zeros(max_len, in_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        

        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, in_dim: int, max_len:int=5000):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, in_dim)
        

        self.positional_encoding = PositionalEncoding(in_dim, max_len=max_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        token_embeddings = self.token_embed(x)

        return self.positional_encoding(token_embeddings)