# Naive_Transformer

This repository contains an educational implementation of the Transformer architecture, as introduced in ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). The code is designed for learning and experimentation, with clear modular components and detailed comments to help you understand how Transformers work.

## Directory Structure

- **transformer/**  
  This directory contains the core modules for the Transformer model:
  - `transformer.py`: Implements the main Transformer model, including encoder and decoder layers, multi-head attention, feed-forward networks, and supporting classes.
  - `encodingLayer.py`: Provides token embedding and positional encoding layers, essential for injecting sequence order information into the model.
  - `__init__.py`: Allows easy import of the main classes.

## Features

- Modular and readable PyTorch code.
- Implements all key components of the original Transformer architecture.
- Designed for educational purposes, with docstrings and comments for clarity.

## Getting Started

To use the modules in your own project:

```python
from transformer import EmbeddingLayer
from transformer import Transformer

in_dim = 512
vocab_size = 10000
num_encoder_layers = 6
encoder_head_size = 8
num_decoder_layers = 6
decoder_head_size = 8

embedding = EmbeddingLayer(vocab_size, in_dim)

model = Transformer(
    in_dim=in_dim,
    num_encoder_layers=num_encoder_layers,
    encoder_head_size=encoder_head_size,
    num_decoder_layers=num_decoder_layers,
    decoder_head_size=decoder_head_size,
    vocab_size=vocab_size
)
```