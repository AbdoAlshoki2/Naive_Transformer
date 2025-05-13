import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implements the standard positional encoding as described in 'Attention Is All You Need'.

    Args:
        in_dim (int): Embedding dimension.
        max_len (int, optional): Maximum sequence length.
    """
    def __init__(self, in_dim: int, max_len: int = 5000):
        super().__init__()

        if in_dim % 2:
            raise ValueError(f"expected the in_dim to be even number, but got {in_dim}.")

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
    """
    Token embedding layer with positional encoding.

    Args:
        vocab_size (int): Vocabulary size.
        in_dim (int): Embedding dimension.
        max_len (int, optional): Maximum sequence length.
    """
    def __init__(self, vocab_size: int, in_dim: int, max_len:int=5000):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, in_dim)
        

        self.positional_encoding = PositionalEncoding(in_dim, max_len=max_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        token_embeddings = self.token_embed(x)

        return self.positional_encoding(token_embeddings)
