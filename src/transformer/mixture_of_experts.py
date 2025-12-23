"""
src/transformer/mixture_of_experts.py

Mixture of Experts implementation in JAX for Nous
"""

import os
import sys

import jax # pylint: disable=no-member
import jax.numpy as jnp # pylint: disable=no-member

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.embeddings.embeddings import EmbeddingLayer

class MOE:
    """
    Mixture of Experts implementation
    """

    def __init__(self, embedding_layer: EmbeddingLayer, ff_dim=None, num_experts=16, experts_per_token=2,
                num_blocks=2, scale=0.02, dropout=0.0, activation="gelu")-> None:
        """
        Initialization of MoE layer

        Args:
            embedding_layer (EmbeddingLayer class): Embedding layer class
            ff_dim (_type_, optional): Feedforward network dimension. Defaults to None.
            num_experts (int, optional): Total number of experts available. Defaults to 16.
            experts_per_token (int, optional): Number of experts active per token. Defaults to 2.
            num_blocks (int, optional): Number of transformer blocks.. Defaults to 2.
            scale (float, optional): Uhhhhh. Defaults to 0.02.
            dropout (float, optional): Probability of dropout. Defaults to 0.0.
            activation (str, optional): Type of dropout. Defaults to "gelu".
        """

        self.embedding_dim = embedding_layer.embedding_dim
        self.ff_dim = ff_dim if ff_dim is not None else 4 * self.embedding_dim
        self.num_experts = num_experts
        self.ept = experts_per_token
        self.num_blocks = num_blocks
        self.scale = scale
        self.dropout = dropout
        self.activation = activation

        # Router: decides which experts to use
        self.router_W = jax.random.normal(
            jax.random.PRNGKey(67),
            (self.embedding_dim, self.num_experts)
        ) * self.scale
        self.router_B = jnp.zeros(self.num_experts)

        # Create experts
        key = jax.random.PRNGKey(0)
        self.experts = []
        for i in range(self.num_experts):
            key, subkey = jax.random.split(key)
            expert = self._init_expert(subkey)
            self.experts.append(expert)


def _init_expert(self, key:int) -> dict:
    """
    Create a single expert. Each expert is an entire FFN.

    Args:
        key (int): PRNG key used

    Returns:
        dict: Contains W1, B1, W2, and B2 (weights and biases for both base and residual layers)
    """
    k1, k2 = jax.random.split(key)

    residual_scale = self.scale / jnp.sqrt(2.0 * self.num_blocks)

    return {
        'W1': jax.random.normal(k1, (self.embedding_dim, self.ff_dim)) * self.scale,
        'B1': jnp.zeros(self.ff_dim),
        'W2': jax.random.normal(k2, (self.ff_dim, self.embedding_dim)) * self.residual_scale,
        'B2': jnp.zeros(self.embedding_dim)
    }
