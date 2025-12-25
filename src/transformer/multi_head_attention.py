"""
src/training/multi_head_attention.py

Multi-head self-attention mechanism implemented with JAX for efficient autodiff and JIT
compilation.

This class implements the multi-head attention mechanism from "Attention is All You Need"
(Vaswani et al., 2017). It splits the embedding dimension into multiple attention heads,
allowing the model to jointly attend to information from different representation subspaces
at different positions.
"""

import os
import sys
from functools import partial

import jax # pylint: disable=no-member
import jax.numpy as jnp # pylint: disable=no-member

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.embeddings.embeddings import EmbeddingLayer
# from src.training.loss_function import CrossEntropyLoss

class MultiHeadAttention:
    """
      Multi-head self-attention mechanism implemented with JAX for efficient autodiff and JIT
      compilation.

      This class implements the multi-head attention mechanism from "Attention is All You Need"
      (Vaswani et al., 2017). It splits the embedding dimension into multiple attention heads,
      allowing the model to jointly attend to information from different representation subspaces
      at different positions.

      Attributes:
          embedding_dim (int): Dimensionality of input embeddings (e.g., 256)
          num_heads (int): Number of parallel attention heads (default: 8)
          head_dim (int): Dimension per head (embedding_dim // num_heads)
          W_Q (jnp.ndarray): Query projection matrix, shape (embedding_dim, embedding_dim)
          W_K (jnp.ndarray): Key projection matrix, shape (embedding_dim, embedding_dim)
          W_V (jnp.ndarray): Value projection matrix, shape (embedding_dim, embedding_dim)
          W_O (jnp.ndarray): Output projection matrix, shape (embedding_dim, embedding_dim)

      """
    def __init__(self,
                embedding_layer: EmbeddingLayer,
                num_heads=8,
                num_blocks=1,
                dropout=0.0) -> None:
        """
        Initializing MutliHeadAttention

        Args:
            embedding_layer (EmbeddingLayer): Instance of EmbeddingLayer class
            num_heads (int, optional): Number of heads for multi-head attention. Defaults to 8.
            num_blocks(int, optional): Number of transformer blocks. Defaults to 1.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        """
        self.embedding_dim = embedding_layer.embedding_dim

        self.num_heads = num_heads
        self.head_dim = self.embedding_dim // self.num_heads
        self.dropout = dropout

        key = jax.random.PRNGKey(0)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        scale = 0.02
        self.W_Q = jax.random.normal(k1, (self.embedding_dim, self.embedding_dim)) * scale
        self.W_K = jax.random.normal(k2, (self.embedding_dim, self.embedding_dim)) * scale
        self.W_V = jax.random.normal(k3, (self.embedding_dim, self.embedding_dim)) * scale

        residual_scale = scale / jnp.sqrt(2.0 * num_blocks)
        self.W_O = jax.random.normal(k4, (self.embedding_dim, self.embedding_dim)) * residual_scale

    @staticmethod
    def fwd(params:dict,
            x:jnp.ndarray,
            num_heads:int,
            head_dim:int,
            embedding_dim:int,
            dropout=0.0,
            training=True,
            rng_key=None) -> jnp.ndarray:
        """
        Pure function for jit computation

        Args:
          params (dict): dict with W_Q, W_K, W_V, W_O
          x (jnp.ndarray): Input embeddings (shape: batch, seq_len, embedding_dim)
          num_heads (int): Number of attention heads
          head_dim (int): Dimension per head
          embedding_dim (int): Embedding dimension
          dropout (float): Dropout probability
          training (bool): Whether in training mode
          rng_key (jax.random.PRNGKey): Random key for dropout

        Return:
            jnp.ndarray: Final output in embeddings
        """

        # Project to query, key, value
        Q = x @ params['W_Q']
        K = x @ params['W_K']
        V = x @ params["W_V"]

        # Reshape for multi-head (batch, seq_len, num_heads, head_dim)
        batch_size, seq_len, _ = x.shape
        Q = Q.reshape(batch_size, seq_len, num_heads, head_dim) # See readme for what .reshape does
        K = K.reshape(batch_size, seq_len, num_heads, head_dim)
        V = V.reshape(batch_size, seq_len, num_heads, head_dim)

        # Transpose to (batch, num_heads, seq_len, head_dim)
        Q = Q.transpose(0,2,1,3)
        K = K.transpose(0,2,1,3)
        V = V.transpose(0,2,1,3)

        # Scaled dot product attention
        scores = Q @ K.transpose(0, 1, 3, 2) / jnp.sqrt(head_dim)

        # Casual mask
        mask = jnp.tril(jnp.ones((seq_len, seq_len))) # See readme for what jnp.tril and jnp.ones do
        scores = jnp.where(mask == 0, -1e9, scores) # See readme for what jnp.where does

        # Softmax to find weights
        attn_weights = jax.nn.softmax(scores, axis=-1) # See readme for what axis=-1 is

        if training and dropout > 0.0 and rng_key is not None:
            keep_prob = 1.0 - dropout
            dropout_mask = jax.random.bernoulli(rng_key, keep_prob, attn_weights.shape)
            attn_weights = jnp.where(dropout_mask, attn_weights / keep_prob, 0.0)


        # Apply attention to values
        attn_output = attn_weights @ V

        # Combining the heads together to make final output
        attn_output = attn_output.transpose(0,2,1,3)
        attn_output = attn_output.reshape(batch_size, seq_len, embedding_dim)

        output = attn_output @ params["W_O"]

        return output

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4))
    def fwd_with_cache(params:dict,
                       x:jnp.ndarray,
                       num_heads:int,
                       head_dim:int,
                       embedding_dim:int,
                       past_kv=None
                       ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Fwd pass with KV cache for faster generation

        Args:
            params (dict): Dictionary with W_Q, W_K, W_V, and W_O
            x (jnp.ndarray): the new token embeddings only.
                Shape [batch_size, new_seq_len, embedding_dim]
                during gen, new_seq_len will be = 1
            num_heads (int): number of attention heads
            head_dim (int): Dimensions/head
            embedding_dim (int): Total embedding dimension
            past_kv (tuple, optional): Tuple of (past_K, past_V) from previous step.
                Defaults to None.

        Returns:
            tuple:
                - out (jnp.ndarray[int, int, int]): Final attention output after
                    combining heads and W_O
                - tuple:
                    - K (jnp.ndarray[int, int, int]): Key tensor in KV cache
                    - V (jnp.ndarray[int, int, int]): Value tensor in KV cache
        """
        batch_size, new_seq_len, _ = x.shape

        Q = x @ params["W_Q"]
        K_new = x @ params["W_K"]
        V_new = x @ params["W_V"]

        Q = Q.reshape(batch_size, new_seq_len, num_heads, head_dim)
        K_new = K_new.reshape(batch_size, new_seq_len, num_heads, head_dim)
        V_new = V_new.reshape(batch_size, new_seq_len, num_heads, head_dim)

        Q = Q.transpose(0, 2, 1, 3)
        K_new = K_new.transpose(0, 2, 1, 3)
        V_new = V_new.transpose(0, 2, 1, 3)

        if past_kv is not None:
            past_K, past_V = past_kv
            K = jnp.concatenate([past_K, K_new], axis=2)
            V = jnp.concatenate([past_V, V_new], axis=2)
        else:
            K = K_new
            V = V_new

        total_seq_len = K.shape[2] # Past seq len + new seq len
        scores = Q @ K.transpose(0, 1, 3, 2) / jnp.sqrt(head_dim)

        mask = jnp.ones((new_seq_len, total_seq_len))
        if new_seq_len > 1:
            past_len = total_seq_len - new_seq_len
            for i in range(new_seq_len):
                mask = mask.at[i, past_len + i + 1:].set(0)

        mask = mask[None, None, :, :]
        scores = jnp.where(mask == 0, -1e9, scores)

        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_out = attn_weights @ V

        attn_out = attn_out.transpose(0, 2, 1, 3)
        attn_out = attn_out.reshape(batch_size, new_seq_len, embedding_dim)

        out = attn_out @ params["W_O"]

        return out, (K, V)


    def compute_grads(self, x:jnp.ndarray, d_output:jnp.ndarray) -> tuple[dict, jnp.ndarray]:
        """
        Most efficient version of backprop using JAX's vjp

        Args:
            x (jnp.ndarray): input (batch, seq_len, embedding_dim)
            d_output (jnp.ndarray): gradient from next layer (batch, seq_len, embedding_dim)

        Returns:
            tuple:
                - grads: dict of parameter gradients
                - d_input (jnp.ndarray): gradient w.r.t input
        """
        params = self.get_params()
        output, vjp_fn = jax.vjp(
            lambda p, x_: self.fwd(p, x_, self.num_heads, self.head_dim, self.embedding_dim), params, x)
        grads_params, d_input = vjp_fn(d_output)
        return grads_params, d_input

    def get_params_and_grads(self, grads=None) -> list[dict]:
        """
        Return params and grads in the format Trainer expects.

        Args:
            grads (dict, optional): Dictionary of gradients. Defaults to None.

        Returns:
            list:
                - dict:
                    - 'value' and 'grad' keys
        """
        if grads is None:
            grads = {
                "W_Q": jnp.zeros_like(self.W_Q),
                "W_K": jnp.zeros_like(self.W_K),
                "W_V": jnp.zeros_like(self.W_V),
                "W_O": jnp.zeros_like(self.W_O)
            }
        return [
            {'value': self.W_Q, 'grad': grads['W_Q']},
            {'value': self.W_K, 'grad': grads['W_K']},
            {'value': self.W_V, 'grad': grads['W_V']},
            {'value': self.W_O, 'grad': grads['W_O']},
        ]

    def get_params(self) -> dict:
        """
        Gets attention layer parameters

        Returns:
            dict:
                - W_Q (jnp.array): Updated query weights
                - W_K (jnp.array): Updated key weights
                - W_V (jnp.array): Updated value weights
                - W_O (jnp.array): Updated output weights
        """
        return {
            "W_Q": self.W_Q,
            "W_K": self.W_K,
            "W_V": self.W_V,
            "W_O": self.W_O
        }
