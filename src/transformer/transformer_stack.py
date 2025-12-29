"""
src/transformer/transformer_stack.py

This file contains the TransformerStack class, which is used to stack multiple transformer blocks
    on top of one another.
"""


import jax
import jax.numpy as jnp

from src.transformer.transformer_block import TransformerBlock
from src.embeddings.embeddings import EmbeddingLayer
from src.transformer.mixture_of_experts import MOE

class TransformerStack:
    """
    Stack of multiple transformer blocks for deep transformer architecture.

    Implements a sequence of transformer blocks where the output of one block
    feeds into the input of the next. More blocks = more model capacity and
    ability to learn complex patterns.

    Attributes:
        num_blocks (int): Number of stacked transformer blocks
        blocks (list): List of TransformerBlock instances
        embedding_dim (int): Dimension of embeddings
        num_heads (int): Number of attention heads per block
            use_moe (bool, optional): Whether to use MoE instead of standard FFN. Defaults to True.
            num_experts (int, optional): Total number of experts. Defaults to 8.
            experts_per_token (int, optional): How many experts to use per token. Defaults to 2.
      """
    def __init__(self,
                embedding_layer:EmbeddingLayer,
                num_blocks=8,
                num_heads=8,
                dropout=0.0,
                use_moe=True,
                num_experts=8,
                experts_per_token=2
                ) -> None:
        """
        Initialize stack of transformer blocks.

        Args:
            embedding_layer (EmbeddingLayer): Embedding layer instance
            num_blocks (int, optional): Number of blocks to stack. Defaults to 8.
            num_heads (int, optional): Number of attention heads per block. Defaults to 8.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            use_moe (bool, optional): Whether to use MoE instead of standard FFN. Defaults to True.
            num_experts (int, optional): Total number of experts. Defaults to 8.
            experts_per_token (int, optional): How many experts to use per token. Defaults to 2.
        """
        self.num_blocks = num_blocks
        self.embedding_dim = embedding_layer.embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.use_moe = use_moe
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token

        # Pass num_blocks and dropout to each TransformerBlock
        self.blocks = [
            TransformerBlock(
                embedding_layer, num_heads, num_blocks, dropout,
                use_moe=use_moe,
                num_experts=num_experts,
                experts_per_token=experts_per_token)
            for _ in range(num_blocks)
        ]

    def fwd(self, x:jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through all stacked blocks

        Args:
            x (jnp.ndarray): Input embeddings (batch, seq_len, embedding_dim)

        Returns:
            jnp.ndarray: Output after passing through all blocks
        """

        output = x
        total_aux_loss = 0.0
        for block in self.blocks:
            params = block.get_params()
            head_dim = self.embedding_dim // self.num_heads
            output, aux_loss = TransformerBlock.fwd(
                params,
                output,
                self.num_heads,
                head_dim,
                self.embedding_dim,
                self.num_experts,
                self.experts_per_token
            )
            total_aux_loss += aux_loss
        return output, total_aux_loss

    def compute_grads(self,
                      x:jnp.ndarray,
                      d_output:jnp.ndarray
                      ) -> list[dict]:
        """
        Backpropagate through all blocks to compute gradients.

        Uses reverse-mode autodiff: start from output, go backwards.

        Args:
            x (jnp.ndarray): Original input to the stack
            d_output (jnp.ndarray): Gradient from loss

        Returns:
            list:
                dict:
                    Contains gradients
        """
        activations = [x]
        current = x

        for block in self.blocks:
            params = block.get_params()
            head_dim = self.embedding_dim // self.num_heads
            current = TransformerBlock.fwd(
                params, current, self.num_heads, head_dim, self.embedding_dim
            )
            activations.append(current)

        all_grads = []
        d_current = d_output

        for i in reversed(range(self.num_blocks)):
            block = self.blocks[i]
            block_input = activations[i]

            grads, d_current = block.compute_grads(block_input, d_current)
            all_grads.insert(0, grads)

        return all_grads

    def get_params_and_grads(self, all_grads=None) -> list:
        """
        Collect parameters and gradients from all blocks.

        Args:
            all_grads (list, optional): List of gradient dicts for each block

        Returns:
            list: Flattened list of all params and grads
        """
        if all_grads is None:
            all_grads = [None] * self.num_blocks

        result = []
        for block, grads in zip(self.blocks, all_grads):
            result.extend(block.get_params_and_grads(grads))

        return result
