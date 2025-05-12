import torch
import torch.nn as nn
from typing import Optional




def scaled_dot_product_attention(queries: torch.Tensor,
        keys: torch.Tensor,
        values:torch.Tensor,
        scale: Optional[float]=None,
        mask:Optional[torch.Tensor]=None,
        masked_atten:Optional[bool]=True,
        padding_mask:Optional[torch.Tensor]=None
    ):
    """
    Computes scaled dot-product attention.

    Args:
        queries (torch.Tensor): Query tensor of shape (..., seq_length, head_dim).
        keys (torch.Tensor): Key tensor of shape (..., seq_length, head_dim).
        values (torch.Tensor): Value tensor of shape (..., seq_length, head_dim).
        scale (float, optional): Scaling factor for attention scores.
        mask (torch.Tensor, optional): Attention mask.
        masked_atten (bool, optional): Whether to apply causal masking.
        padding_mask (torch.Tensor, optional): Padding mask for keys.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
    """
    # work with input of shape: (Batch_size, seq_length, head_out) or (Batch_size, nheads, seq_length, head_out)
    L , S = queries.size(-2) , keys.size(-2)   # seq_length
    if mask is None:
        mask = torch.ones((L,S) , dtype=queries.dtype, device=queries.device)
        mask = torch.tril(mask) if masked_atten else mask

    if scale is None:
        scale = (queries.size(-1) **-0.5)

    wei = queries @ keys.transpose(-2,-1) * torch.tensor(scale) # (Batch_size , seq_length , seq_length)
    wei = wei.masked_fill(mask == 0, float('-inf'))

    if padding_mask is not None:

        if padding_mask.dim() != 2:
            raise ValueError(f"Expected padding_mask to be 2 dimensions, but got {padding_mask.dim()}")

        key_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_length)
        

        wei = wei.masked_fill(key_padding_mask == 0, float('-inf'))        

    wei = wei.softmax(-1)
    return wei @ values, wei  # (Batch_size , seq_length , head_size)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.

    Allows the model to jointly attend to information from different representation subspaces.

    Args:
        n_heads (int): Number of attention heads.
        in_dim (int): Input dimension.
        mask (torch.Tensor, optional): Attention mask.
        scale (float, optional): Scaling factor for attention scores.
        masked_atten (bool, optional): Whether to apply causal masking.
        drop_out (float, optional): Dropout rate.
    """
    def __init__(self,
                n_heads:int,
                in_dim:int,
                mask:Optional[torch.Tensor]=None,
                scale:Optional[float]=None, 
                masked_atten:bool = True,
                drop_out=0.5
        ):
        super().__init__()
        self.n_heads = n_heads
        assert in_dim % n_heads == 0
        self.head_size = in_dim // n_heads

        self.mask = mask
        self.masked_atten = masked_atten
        self.scale = scale
        self.dropout = nn.Dropout(drop_out)


    def forward(self, 
                queries:torch.Tensor, 
                keys: torch.Tensor, 
                values:torch.Tensor,
                padding_mask: Optional[torch.Tensor]=None
        ):
        """
        Forward pass for multi-head attention.

        Args:
            queries (torch.Tensor): Query tensor.
            keys (torch.Tensor): Key tensor.
            values (torch.Tensor): Value tensor.
            padding_mask (torch.Tensor, optional): Padding mask.

        Returns:
            torch.Tensor: Output tensor after attention.
        """

        expected_dim = self.n_heads * self.head_size
        if keys.size(-1) != expected_dim or queries.size(-1) != expected_dim or values.size(-1) != expected_dim:
            raise ValueError(f"Expected the shape to be {expected_dim} but got: {keys.size(-1)} , {queries.size(-1)} , {values.size(-1)}")

        keys = keys.unflatten(-1 , (self.n_heads , self.head_size)).transpose(1,2) # (Batch_size , n_heads , seq_length , head_size)
        queries = queries.unflatten(-1 , (self.n_heads , self.head_size)).transpose(1,2) # (Batch_size , n_heads , seq_length , head_size)
        values = values.unflatten(-1 , (self.n_heads , self.head_size)).transpose(1,2) # (Batch_size , n_heads , seq_length , head_size)


        attn_out , self.attn_scores = scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=self.mask,
            masked_atten = self.masked_atten,
            padding_mask= padding_mask
        )
        attn_out = self.dropout(attn_out)
        attn_out = attn_out.transpose(1,2).flatten(-2)
        return attn_out

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Args:
        in_dim (int): Input and output dimension.
        drop_out (float, optional): Dropout rate.
    """
    def __init__(self, in_dim:int , drop_out=0.5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim , in_dim * 4),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(in_dim * 4 , in_dim),
            nn.Dropout(drop_out)
        )


    def forward(self , x: torch.Tensor):
        """
        Forward pass for feed-forward network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.proj(x)


class PreAttention(nn.Module):
    """
    Linear projections for queries, keys, and values.

    Args:
        in_dim (int): Input and output dimension.
    """
    def __init__(self, in_dim:int):
        super().__init__()
        self.k = nn.Linear(in_dim , in_dim)
        self.q = nn.Linear(in_dim , in_dim)
        self.v = nn.Linear(in_dim , in_dim)
    
    def forward(self, 
                input_1:torch.Tensor, 
                input_2: Optional[torch.Tensor] = None
        ):
        """
        Projects input(s) to queries, keys, and values.

        Args:
            input_1 (torch.Tensor): Main input tensor.
            input_2 (torch.Tensor, optional): Optional second input for queries.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: queries, keys, values.
        """
        if input_2 is not None:
            queries = self.q(input_2)  # (Batch_size , seq_length , in_dim)
        else:
            queries = self.q(input_1)  # (Batch_size , seq_length , in_dim)

        keys = self.k(input_1)  # (Batch_size , seq_length , in_dim)
        values = self.v(input_1)  # (Batch_size , seq_length , in_dim)

        return queries , keys , values 




class EncoderLayer(nn.Module):
    """
    Transformer encoder layer.

    Consists of multi-head self-attention and feed-forward network with residual connections and layer normalization.

    Args:
        n_heads (int): Number of attention heads.
        in_dim (int): Input and output dimension.
    """
    def __init__(self, n_heads:int, in_dim:int):
        super().__init__()
        masked_attn = False   # for encoder layers

        self.pre_attn = PreAttention(in_dim)
        self.ma = MultiHeadAttention(n_heads, in_dim , masked_atten=masked_attn)
        self.ff = FeedForward(in_dim)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        

    def forward(self, 
                x: torch.Tensor,  
                padding_mask: Optional[torch.Tensor]=None
        ):
        """
        Forward pass for encoder layer.

        Args:
            x (torch.Tensor): Input tensor.
            padding_mask (torch.Tensor, optional): Padding mask.

        Returns:
            torch.Tensor: Output tensor.
        """
        q,  k , v = self.pre_attn(x) # prepare the queries , keys, and values
        out = self.norm1(x + self.ma(q,  k , v, padding_mask)) # passing the queries, keys, and values then normalize them
        out = self.norm2(out + self.ff(out)) # passing out to the feedforward layer, then normalize
        return out



class DecoderLayer(nn.Module):
    """
    Transformer decoder layer.

    Consists of masked multi-head self-attention, cross-attention, and feed-forward network with residual connections and layer normalization.

    Args:
        n_heads (int): Number of attention heads.
        in_dim (int): Input and output dimension.
    """
    def __init__(self, n_heads:int, in_dim:int):
        super().__init__()
        self.masked_pre = PreAttention(in_dim)
        self.masked_ma = MultiHeadAttention(n_heads, in_dim)


        self.cross_pre = PreAttention(in_dim)
        self.cross_ma = MultiHeadAttention(n_heads, in_dim, masked_atten=False)
        
        self.ff = FeedForward(in_dim)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)

    def forward(self, 
                input_1:torch.Tensor, 
                input_2:Optional[torch.Tensor]=None,
                tgt_padding_mask: Optional[torch.Tensor]=None,
                src_padding_mask: Optional[torch.Tensor]=None
        ):
        """
        Forward pass for decoder layer.

        Args:
            input_1 (torch.Tensor): Decoder input tensor.
            input_2 (torch.Tensor, optional): Encoder output tensor.
            tgt_padding_mask (torch.Tensor, optional): Target padding mask for the decoder masked attention.
            src_padding_mask (torch.Tensor, optional): Source padding mask for the encoder cross attention.

        Returns:
            torch.Tensor: Output tensor.
        """

        out = input_1
        q , k , v = self.masked_pre(input_1) # prepare the queries , keys, and values
        out = self.norm1(out + self.masked_ma(q, k, v, tgt_padding_mask)) # passing the queries, keys, and values then normalize them

        if input_2 is not None: # if there is an encoder layer (encoder output)
            q , k , v = self.cross_pre(input_2, out) # the decoder input becomes second input with encoder output
            out = self.norm2(out + self.cross_ma(q, k, v, src_padding_mask)) # passing the queries, keys, and values then normalize them
        
        out = self.norm3(out + self.ff(out)) # passing out to the feedforward layer, then normalize
        return out

class Transformer(nn.Module):
    """
    Transformer model consisting of stacked encoder and decoder layers.

    Args:
        in_dim (int): Input and output dimension.
        num_encoder_layers (int): Number of encoder layers.
        encoder_head_size (int): Number of heads in encoder.
        num_decoder_layers (int): Number of decoder layers.
        decoder_head_size (int): Number of heads in decoder.
        vocab_size (int, optional): Vocabulary size for output projection.
    """
    def __init__(self,
                in_dim:int,
                num_encoder_layers:int,
                encoder_head_size:int, 
                num_decoder_layers:int, 
                decoder_head_size:int,
                vocab_size: Optional[int] = None
        ):
        super().__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(encoder_head_size, in_dim) for _ in range(num_encoder_layers)]) # stacking multiple encoder layers
        self.decoder_layers = nn.ModuleList([DecoderLayer(decoder_head_size , in_dim) for _ in range(num_decoder_layers)]) # stacking multiple decoder layers

        self.vocab_size = vocab_size
        if vocab_size is not None:
            self.projection = nn.Linear(in_dim , vocab_size)
        


    def forward(self, encoder_input:torch.Tensor, 
                decoder_input: torch.Tensor,
                src_padding_mask:Optional[torch.Tensor]=None,
                tgt_padding_mask:Optional[torch.Tensor]=None
        ):
        """
        Forward pass for the Transformer model.

        Args:
            encoder_input (torch.Tensor): Input tensor to the encoder.
            decoder_input (torch.Tensor): Input tensor to the decoder.
            src_padding_mask (torch.Tensor, optional): Source padding mask for the encoder input.
            tgt_padding_mask (torch.Tensor, optional): Target padding mask for the decoder input.

        Returns:
            torch.Tensor: Output tensor.
        """

        enc_out = encoder_input # take a copy
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, padding_mask=src_padding_mask)

        decoder_out = decoder_input
        for layer in self.decoder_layers:
            decoder_out = layer(
                decoder_out, 
                enc_out, 
                tgt_padding_mask=tgt_padding_mask,
                src_padding_mask=src_padding_mask
            )
        
        if self.vocab_size is not None:
            decoder_out = self.projection(decoder_out)
        
        return decoder_out

