"""
./src/transformer/transformer_block.py

The TransformerBlock class representing a single transformer block, computing both the forward and
    backward pass.

Each transformer block consists of a multi head attention block followed by a mixture of experts layer,
with residual connections and layer normalization applied at each sublayer (after the use of MHA,
then MoE)

Classes:
    TransformerBlock:
        Implements a transformer block with:
            - MHA Self-attention
            - Mixture of Experts (MoE)
            - Layer norm and residual connection
            - Fwd pass with optional dropout and KV cache
            - Gradient computation for backprop
            - Parameter access
"""

import os
import sys
from functools import partial

import jax # pylint: disable=no-member
import jax.numpy as jnp # pylint: disable=no-member

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.embeddings.embeddings import EmbeddingLayer
from src.transformer.multi_head_attention import MultiHeadAttention
from src.transformer.feed_forward import FeedForward
from src.transformer.mixture_of_experts import MOE
from src.transformer.cross_attention import CrossAttention


class TransformerBlock:
    """
    Represents a single transformer block, including attention and MoE layers.
    """

    def __init__(
            self,
            embedding_layer:EmbeddingLayer,
            num_heads=8,
            num_blocks=8,
            dropout=0.0,
            use_moe=True,
            num_experts=8,
            experts_per_token=2,
            dtype=None
            ):
        """
        Initializing instance variables for the TransformerBlock class

        Args:
            embedding_layer (EmbeddingLayer): EmbeddingLayer class, which takes no arguments.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            num_blocks (int, optional): Number of transformer blocks (depth). Defaults to 8.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            use_moe (bool, optional): Whether to use MoE instead of standard FFN. Defaults to True.
            num_experts (int, optional): Total number of experts. Defaults to 8.
            experts_per_token (int, optional): How many experts to use per token. Defaults to 2.
            dtype (jnp.dtype, optional): Data type for weights. Defaults to jnp.bfloat16.
        """
        self.dtype = dtype if dtype is not None else jnp.bfloat16
        self.embedding_dim = embedding_layer.embedding_dim
        self.num_heads = num_heads

        self.attention_layer = MultiHeadAttention(embedding_layer, num_heads, num_blocks, dropout, dtype=self.dtype)
        self.ffn = FeedForward(embedding_layer, num_blocks=num_blocks, dropout=dropout, dtype=self.dtype)

        self.gamma_1 = jnp.ones((self.embedding_dim,), dtype=self.dtype)
        self.beta_1 = jnp.zeros((self.embedding_dim,), dtype=self.dtype)
        self.gamma_2 = jnp.ones((self.embedding_dim,), dtype=self.dtype)
        self.beta_2 = jnp.zeros((self.embedding_dim,), dtype=self.dtype)

        self.gamma_cross = jnp.ones((self.embedding_dim,), dtype=self.dtype)
        self.beta_cross = jnp.zeros((self.embedding_dim,), dtype=self.dtype)

        self.use_moe = use_moe

        if self.use_moe:
            # Create MoE layer instead of FFN
            self.moe = MOE(
                embedding_dim=self.embedding_dim,
                ff_dim=4 * self.embedding_dim,
                num_experts=num_experts,
                experts_per_token=experts_per_token,
                num_blocks=num_blocks,
                dropout=dropout,
                dtype=self.dtype
            )
        else:
            self.ffn = FeedForward(
                embeddings=embedding_layer,
                num_blocks=num_blocks,
                dropout=dropout,
                dtype=self.dtype
            )

    @staticmethod
    @jax.jit
    def layer_norm(
        x:jnp.ndarray,
        gamma:jnp.ndarray,
        beta:jnp.ndarray,
        epsilon=1e-5
        ):
        """
        Layer normalization - normalizes across the feature dimension.

        Computes: output = gamma * (x - mean) / sqrt(variance + epsilon) + beta

        Args:
            x (jnp.ndarray): Input tensor (batch, seq_len, embedding_dim)
            gamma (jnp.ndarray): Scale parameter (embedding_dim,)
            beta (jnp.ndarray): Shift parameter (embedding_dim,)
            epsilon (float): Small constant for numerical stability

        Returns:
            jnp.ndarray: Normalized output, same shape as input
        """

        mean = jnp.mean(x, axis=-1 , keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / jnp.sqrt(var + epsilon)
        output = gamma * normalized + beta

        return output

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
    def fwd(
        params:dict,
        x:jnp.ndarray,
        num_heads:int,
        head_dim:int,
        embedding_dim:int,
        num_experts=8,
        experts_per_token=2,
        dropout=0.0,
        training=True,
        rng_key=None
        ):
        """
        Forward pass through transformer block (pure function for JIT).

        Flow:
            1. LayerNorm → MultiHeadAttention → Add residual
            2. LayerNorm → FeedForward → Add residual

        Args:
            params (dict): Contains all parameters.
            x (jnp.ndarray): Input embeddings (batch, seq_len, embedding_dim)
            num_heads (int): Number of attention heads
            head_dim (int): Dimension per head
            embedding_dim (int): Total embedding dimension
            num_experts (int, optional): Total number of experts. Defaults to 8.
            experts_per_token (int, optional): How many experts are used for each token.
                                                Defaults to 2.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            training (bool, optional): If the model is in training. Defaults to True.
            rng_key (jax.random.PRNGkey, optional): JAX PRNG key. Defaults to None.

        Returns:
            tuple:
                - jnp.ndarray: Output (batch, seq_len, embedding_dim)
                - jnp.float16: auxiliary loss.
          """

        if rng_key is not None:
            rng_attn, rng_ffn = jax.random.split(rng_key)
        else:
            rng_attn, rng_ffn = None, None

        # ==== Sublayer 1 ====
        residual_1 = x
        ln1_out = TransformerBlock.layer_norm(x, params['gamma_1'], params['beta_1'])

        attn_output = MultiHeadAttention.fwd(
            params['attn'],
            ln1_out,
            num_heads,
            head_dim,
            embedding_dim,
            dropout=dropout,
            training=training,
            rng_key=rng_attn
        )

        # Residual dropout on attention output
        if training and dropout > 0.0 and rng_attn is not None:
            keep_prob = 1.0 - dropout
            dropout_mask = jax.random.bernoulli(rng_attn, keep_prob, attn_output.shape)
            attn_output = jnp.where(dropout_mask, attn_output / keep_prob, 0.0)

        after_attention = residual_1 + attn_output

        # ==== Sublayer 2 ====
        residual_2 = after_attention
        ln2_out = TransformerBlock.layer_norm(
            after_attention,
            params['gamma_2'],
            params['beta_2']
        )

        if 'moe' in params:
            ff_output, aux_loss = MOE.fwd(
                params['moe'],
                ln2_out,
                num_experts=num_experts,
                experts_per_token=experts_per_token,
                activation='gelu',
                training=training,
                dropout=dropout,
                key=rng_ffn
            )
        else:
            ff_output = FeedForward.fwd(
                params['ffn'],
                ln2_out,
                dropout=dropout,
                training=training,
                rng_key=rng_ffn
            )
            aux_loss = 0.0

        final_output = residual_2 + ff_output

        return final_output, aux_loss

    @staticmethod
    @jax.jit
    def fwd_with_x_attn(
        params:dict,
        x:jnp.ndarray,
        encoder_outputs:jnp.ndarray,
        num_heads:int,
        head_dim:int,
        embedding_dim:int,
        num_experts=8,
        experts_per_token=2,
        dropout=0.0,
        training=True,
        rng_key=None
        ):
        """
        Forward pass with cross-attention for a multimodal transformer

        Flow (3 sublayers instead of 2):
            1. LayerNorm -> Self-Attention (MHA: text->text) -> Residual
            2. LayerNorm -> CrossAttention (text->image) -> Residual
            3. LayerNorm -> FFN -> Residual

        Args:
            params (dict): Parameters, including 'cross_attn' key with cross-attention params.
            x (jnp.ndarray): Decoder embeddings (text).
                             Shape: (batch, seq_len_dec, embedding_dim)
            encoder_outputs (jnp.ndarray): Encoder outputs (image).
                                           Shape: (batch, num_patches, embedding_dim)
            num_heads (int): Number of attention heads
            head_dim (int): Dimension per head
            embedding_dim (int): Total embedding dimension
            num_experts (int, optional): Total number of experts. Defaults to 8.
            experts_per_token (int, optional): How many experts are used for each token.
                                                Defaults to 2.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            training (bool, optional): If the model is in training. Defaults to True.
            rng_key (jax.random.PRNGkey, optional): JAX PRNG key. Defaults to None.

        Returns:
            tuple[jnp.ndarray, float, jnp.ndarray]: (output, aux_loss, cross_attn_weights)
        """

        if rng_key is not None:
            rng_attn, rng_cross, rng_ffn = jax.random.split(rng_key, 3)
        else:
            rng_attn, rng_cross, rng_ffn = None, None, None

        # ==== Sublayer 1 - Self-Attention ====
        residual_1 = x
        ln1_out = TransformerBlock.layer_norm(x, params['gamma_1'], params['beta_1'])

        attn_output = MultiHeadAttention.fwd(
            params['attn'],
            ln1_out,
            num_heads,
            head_dim,
            embedding_dim,
            dropout=dropout,
            training=training,
            rng_key=rng_attn
        )

        # Residual dropout on attention output
        if training and dropout > 0.0 and rng_attn is not None:
            keep_prob = 1.0 - dropout
            dropout_mask = jax.random.bernoulli(rng_attn, keep_prob, attn_output.shape)
            attn_output = jnp.where(dropout_mask, attn_output / keep_prob, 0.0)

        after_self_attn = residual_1 + attn_output

        # ===== Sublayer 2 - Cross-Attention (x-attn) =======

        residual_2 = after_self_attn

        ln2_out = TransformerBlock.layer_norm(
            after_self_attn,
            params['gamma_cross'],
            params['beta_cross']
        )

        cross_attn_output, cross_attn_weights = CrossAttention.fwd(
            params['cross_attn'],
            ln2_out,
            encoder_outputs,
            num_heads,
            head_dim,
            embedding_dim
        )

        if training and dropout > 0.0 and rng_cross is not None:
            keep_prob = 1.0 - dropout
            dropout_mask = jax.random.bernoulli(rng_cross, keep_prob, cross_attn_output.shape)
            cross_attn_output = jnp.where(dropout_mask, cross_attn_output / keep_prob, 0.0)

        after_cross_attn = residual_2 + cross_attn_output

        # ===== Sublayer 3 - FFN =====

        residual_3 = after_cross_attn

        ln3_out = TransformerBlock.layer_norm(
            after_cross_attn,
            params['gamma_2'],
            params['beta_2']
        )

        if 'moe' in params:
            ff_output, aux_loss = MOE.fwd(
                params['moe'],
                ln3_out,
                num_experts=num_experts,
                experts_per_token=experts_per_token,
                activation='gelu',
                training=training,
                dropout=dropout,
                key=rng_ffn
            )
        else:
            ff_output = FeedForward.fwd(
                params['ffn'],
                ln3_out,
                dropout=dropout,
                training=training,
                rng_key=rng_ffn
            )
            aux_loss = 0.0

        final_output = residual_3 + ff_output

        return final_output, aux_loss, cross_attn_weights


    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4))
    def fwd_with_cache(
        params:dict,
        x:jnp.ndarray,
        num_heads:int,
        head_dim:int,
        embedding_dim:int,
        past_kv=None
        ):
        """
        Forward pass with KV-cache

        Args:
            params (dict): Block params
            x (jnp.ndarray): new token embeddings, shape: [batch, 1, embedding_dim]
            num_heads (int): Number of attention heads
            head_dim (int): Dimension of each head
            embedding_dim (int): Embedding dimension
            past_kv (jnp.ndarray, optional): Cached K and V from the layer's other calls.
                Defaults to None.

        Returns:
            tuple:
                output (jnp.ndarray): (batch, 1, embedding_dim)
                new_kv (jnp.ndarray): Updated (K, V) cache
        """
        res1 = x
        ln1_out = TransformerBlock.layer_norm(x, params['gamma_1'], params['beta_1'])

        attn_out, new_kv = MultiHeadAttention.fwd_with_cache(
            params['attn'],
            ln1_out,
            num_heads,
            head_dim,
            embedding_dim,
            past_kv=past_kv
        )
        after_attention = res1 + attn_out

        res2 = after_attention
        ln2_out = TransformerBlock.layer_norm(after_attention, params['gamma_2'], params['beta_2'])

        ff_out = FeedForward.fwd(params['ffn'], ln2_out)
        final_out = res2 + ff_out

        return final_out, new_kv

    def get_params(self):
        """
        Get all parameters as a dictionary for JAX functions

        Returns:
            dict: All trainable params
        """
        params = {
            'attn': self.attention_layer.get_params(),
            'gamma_1': self.gamma_1,
            'beta_1': self.beta_1,
            'gamma_2': self.gamma_2,
            'beta_2': self.beta_2
        }

        if self.use_moe:
            params["moe"] = self.moe.get_params()
        else:
            params['ffn'] = {
                'W1': self.ffn.W1,
                'B1': self.ffn.B1,
                'W2': self.ffn.W2,
                'B2': self.ffn.B2
            }

        return params

    def compute_grads(
            self,
            x:jnp.ndarray,
            d_output:jnp.ndarray
            ):
        """
        Compute gradients using JAX autodiff

        Args:
            x (jnp.ndarray): Input to this block (batch, seq_len, embedding_dim)
            d_output (jnp.ndarray): Gradient from next layer/transformer block

        Returns:
            tuple:
                - grads_dict (dict): Gradients for all parameters
                - d_input (jnp.ndarray): Gradient w.r.t (for previous block)
        """
        params = self.get_params()

        output, vjp_fn = jax.vjp(
            lambda p, x_: self.fwd(
                p, x_, self.num_heads,
                self.embedding_dim // self.num_heads,
                self.embedding_dim
            ),
            params, x
        )

        grads_params, d_input = vjp_fn(d_output)
        return grads_params, d_input

    def get_params_and_grads(
            self,
            grads=None
            ):
        """
        Return params and grads in Trainer format

        Args:
            grads (dict, optional): Gradient dictionary. Defaults to None.

        Returns:
            list:
                dict:
                    - Grads
        """

        if grads is None:
            grads = {
                'attn': {k: jnp.zeros_like(v) for k,v in self.attention_layer.get_params().items()},
                'ffn':{
                    'W1': jnp.zeros_like(self.ffn.W1),
                    'B1': jnp.zeros_like(self.ffn.B1),
                    'W2': jnp.zeros_like(self.ffn.W2),
                    'B2': jnp.zeros_like(self.ffn.B2)
                },
                'gamma_1': jnp.zeros_like(self.gamma_1),
                'beta_1': jnp.zeros_like(self.beta_1),
                'gamma_2': jnp.zeros_like(self.gamma_2),
                'beta_2': jnp.zeros_like(self.beta_2)
            }
        result = []

        for key in ['W_Q', 'W_K', "W_V", "W_O"]:
            result.append({
                'value': self.attention_layer.get_params()[key],
                'grad': grads['attn'][key]
            })

        result.extend([
            {'value': self.ffn.W1, 'grad': grads['ffn']['W1']},
            {'value': self.ffn.B1, 'grad': grads['ffn']['B1']},
            {'value': self.ffn.W2, 'grad': grads['ffn']['W2']},
            {'value': self.ffn.B2, 'grad': grads['ffn']['B2']}
        ])

        result.extend([
            {'value': self.gamma_1, 'grad': grads['gamma_1']},
            {'value': self.beta_1, 'grad': grads['beta_1']},
            {'value': self.gamma_2, 'grad': grads['gamma_2']},
            {'value': self.beta_2, 'grad': grads['beta_2']}
        ])

        return result
