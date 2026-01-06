"""
./src/transformer/cross_attention.py

Cross-attention for looking at external context (for example, image features).
Similar to MHA but queries come from decoder, keys/values from encoder.
"""

import jax
import jax.numpy as jnp

class CrossAttention:
    """
    Multi-head cross-attention for paying attention to encoder outputs.
    """
    def __init__(self,
                 embedding_dim=512,
                 num_heads=8,
                 num_blocks=8,
                 dropout=0.0
                 ):
        """
        Initializing the CrossAttention class

        Args:
            embedding_dim (int, optional): Embedding dimension. Defaults to 512.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            num_blocks (int, optional): Number of transformer blocks. Defaults to 8.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        """
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.dropout = dropout

        key = jax.random.PRNGKey(
            68157628006304057045295846951897664502295431894160942124012093587298959185368
        )
        k1, k2, k3, k4 = jax.random.split(key, 4)
        scale = 0.02

        self.W_Q = jax.random.normal(
            k1,
            (embedding_dim, embedding_dim)
        ) * scale
        self.W_K = jax.random.normal(
            k2,
            (embedding_dim, embedding_dim)
        ) * scale
        self.W_V = jax.random.normal(
            k3,
            (embedding_dim, embedding_dim)
        ) * scale

        residual_scale = scale / jnp.sqrt(2.0 * num_blocks)
        self.W_O = jax.random.normal(
            k4,
            (embedding_dim, embedding_dim)
        ) * residual_scale

    @staticmethod
    @jax.jit
    def fwd(params:dict,
            decoder_x:jnp.ndarray,
            encoder_outputs:jnp.ndarray,
            num_heads=8,
            embedding_dim=512,
            head_dim=64
            ):
        """
        Forward pass for cross-attention.

        Args:
            params (dict): Contains W_Q, W_K, W_V, W_O
            decoder_x (jnp.ndarray): Decoder embeddings (the text).
                                     Shape: (batch, seq_len_decoder, embedding_dim)
            encoder_outputs (jnp.ndarray): Encoder outputs (the images).
                                           Shape: (batch, seq_len_encoder, embedding_dim)
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            embedding_dim (int, optional): Total embedding dimension. Defaults to 512.
            head_dim (int, optional): Dimension per head. Defaults to 64.

        Returns:
            jnp.ndarray: Cross-attention output
                         Shape: (batch, seq_len_decoder, embedding_dim)
        """
        batch_size, seq_len_dec, _ = decoder_x.shape
        seq_len_enc = encoder_outputs.shape[1]

        # Query with decoder weights
        Q = decoder_x @ params['W_Q']

        # Keys and values with encoder weights
        K = encoder_outputs @ params['W_K']
        V = encoder_outputs @ params['W_V']

        # Reshape for MHA
        Q = Q.reshape(
            batch_size,
            seq_len_dec,
            num_heads,
            head_dim
        )

        K = K.reshape(
            batch_size,
            seq_len_enc,
            num_heads,
            head_dim
        )
        V = V.reshape(
            batch_size,
            seq_len_enc,
            num_heads,
            head_dim
        )

        # Transpose to (batch, num_heads, seq_len, head_dim)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        # (batch, num_heads, seq_len_decoder, head_dim) @ (batch, num_heads, head_dim, seq_len_encoder)
        # → (batch, num_heads, seq_len_decoder, seq_len_encoder)
        scores = Q @ K.transpose(0, 1, 3, 2) / jnp.sqrt(head_dim)
            # (batch, num_heads, seq_len_dec, seq_len_enc)

        attn_weights = jax.nn.softmax(scores, axis=-1) # Using softmax to get attention weights

        # Applying attention to values
        attn_output = attn_weights @ V

        # Combine the heads at the end
        attn_output = attn_output.transpose(0, 2, 1, 3) # (batch, seq_len_dec, num_heads, head_dim)
        attn_output = attn_output.reshape(
            batch_size,
            seq_len_dec,
            embedding_dim
        )

        output = attn_output @ params['W_O']

        return output, attn_weights # Type "tuple[Unknown, Array]" is not assignable to return type "ndarray" "tuple[Unknown, Array]" is not assignable to "Array"

    def get_params(self):
        """
        Get parameters for JAX functions

        Returns:
            dict: Dictionary containig query, key, value, and output weights
        """
        return {
            'W_Q': self.W_Q,
            'W_K': self.W_K,
            'W_V': self.W_V,
            'W_O': self.W_O
        }
