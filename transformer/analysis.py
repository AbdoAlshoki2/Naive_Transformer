import matplotlib.pyplot as plt
import torch

def plot_attention_scores(
    transformer, tokenizer,
    encoder_input_ids, decoder_input_ids=None,
    layer_type="encoder", layer_index=0, head_index=None,
    truncate_pad=True, cmap="viridis"
):
    """
    Plot attention heatmaps for a given Transformer layer and head.

    Args:
        transformer: Transformer model.
        tokenizer: tokenizer for converting ids to tokens.
        encoder_input_ids: torch.Tensor of shape (1, src_len)
        decoder_input_ids: torch.Tensor of shape (1, tgt_len) — required for decoder or cross
        layer_type: "encoder" | "decoder" | "cross"
        layer_index: int, index of layer to visualize.
        head_index: optional int — which head to show (or None to show all).
        truncate_pad: bool — remove <pad> tokens from axes.
        cmap: str — colormap name (e.g. "viridis", "plasma").
    """
    def get_tokens(input_ids, truncate):
        pad_id = tokenizer.pad_token_id
        mask = input_ids[0] != pad_id
        seq_len = mask.sum().item() if truncate else input_ids.size(1)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        return tokens[:seq_len], seq_len

    if layer_type == "encoder":
        tokens_q, len_q = get_tokens(encoder_input_ids, truncate_pad)
        tokens_k, len_k = tokens_q, len_q
        mha = transformer.encoder_layers[layer_index].ma

    elif layer_type == "decoder":
        if decoder_input_ids is None:
            raise ValueError("decoder_input_ids is required for decoder self-attention")
        tokens_q, len_q = get_tokens(decoder_input_ids, truncate_pad)
        tokens_k, len_k = tokens_q, len_q
        mha = transformer.decoder_layers[layer_index].masked_ma

    elif layer_type == "cross":
        if decoder_input_ids is None:
            raise ValueError("decoder_input_ids is required for cross-attention")
        tokens_q, len_q = get_tokens(decoder_input_ids, truncate_pad)
        tokens_k, len_k = get_tokens(encoder_input_ids, truncate_pad)
        mha = transformer.decoder_layers[layer_index].cross_ma

    else:
        raise ValueError("layer_type must be one of: encoder, decoder, cross")

    scores = mha.attn_scores  # (B, n_heads, tgt_len, src_len)
    if scores is None:
        raise ValueError("No attention scores available. Run a forward pass first.")

    scores = scores[0].cpu().detach()  # shape: (n_heads, Q, K)

    if truncate_pad:
        scores = scores[:, :len_q, :len_k]

    heads = [head_index] if head_index is not None else range(scores.size(0))

    for h in heads:
        plt.figure(figsize=(6, 6))
        plt.title(f"{layer_type.title()} Layer {layer_index} - Head {h}")
        plt.imshow(scores[h], cmap=cmap)
        plt.xticks(range(len_k), tokens_k, rotation=90)
        plt.yticks(range(len_q), tokens_q)
        plt.xlabel("Key Tokens")
        plt.ylabel("Query Tokens")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

