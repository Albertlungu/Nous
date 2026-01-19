"""
./src/embeddings/positional_encoding.py

Positional encoding for transformer models.

This module provides sinusoidal positional encodings to give the model
information about token positions in sequences.
"""
import jax # pylint: disable=no-member
import jax.numpy as jnp # pylint: disable=no-member
from jax import Array

class PositionalEncoding:
    """
    Sinusoidal positional encoding for transformer architectures.

    Generates position-dependent patterns using sine and cosine functions
    to encode sequence position information.
    """
    def __init__(
            self,
            embedding_dim:int,
            max_seq_length=256,
            dtype=None
            ):
        """
        Initialize the positional encoding generator.

        Args:
            embedding_dim (int): Embedding dimension.
            max_seq_length (int, optional): Maximum sequence length. Defaults to 256.
            dtype (jnp.dtype, optional): Data type for encoding. Defaults to jnp.bfloat16.
        """
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.dtype = dtype if dtype is not None else jnp.bfloat16

    def _create_positional_encoding(
            self,
            n=10000
            ):
        """
        Create sinusoidal positional encodings.

        Args:
            n (int, optional): Frequency parameter for encoding. Defaults to 10000.

        Returns:
            jnp.ndarray: Positional encoding matrix of shape (max_seq_length, embedding_dim).
        """

        L, d = self.max_seq_length, self.embedding_dim
        pos = jnp.arange(L)[:, None]
        i = jnp.arange(d)[None, :]
        angle_rates = 1 / jnp.power(n, (2 * (i//2)) / d)
        P = pos * angle_rates
        P = P.at[:, 0::2].set(jnp.sin(P[:, 0::2]))
        P = P.at[:, 1::2].set(jnp.cos(P[:, 1::2]))
        return P.astype(self.dtype)
