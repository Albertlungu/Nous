"""
src/vision/vit/vit_encoder.py

Vision Transformer (ViT) encoder for processing image patches.
Reuses existing TransformerBlock.
"""

# TODO: Add encode method

import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

import jax
import jax.numpy as jnp
from functools import partial

from src.transformer.transformer_block import TransformerBlock
from src.transformer.transformer_stack import TransformerStack
from src.embeddings.embeddings import EmbeddingLayer
from src.vision.vit.patch_embeddings import PatchEmbedding

class ViTEncoder:
    """
    Vision Transformer encoder.
    """
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 embedding_dim=256,
                 num_blocks=8,
                 num_heads=8,
                 num_experts=8,
                 experts_per_token=2,
                 dropout=0.0
                 ) -> None:
        """
        Initializing the Vision encoder

        Args:
            image_size (int, optional): Input image size, assuming square images.
                                        e.g. Image is 224x224. Defaults to 224.
            patch_size (int, optional): Size of each patch. Defaults to 16.
            in_channels (int, optional): Number of input channels (3 for RGB, 1 for grayscale).
                                         Defaults to 3.
            embedding_dim (int, optional): Dimension of patch embeddings.
                                           Must match text embedding_dim. Defaults to 256.
            num_blocks (int, optional): Number of transformer blocks. Defaults to 8.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        self.num_blocks = num_blocks
        self.num_heads = num_heads

        self.num_experts = num_experts
        self.experts_per_token = experts_per_token

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim
        ) # Patch embeddings

        # Creating dummy EmbeddingLayer for compatibility with TransformerStack
        dummy_embedding = EmbeddingLayer(
            vocab_size=1000,
            embedding_dim=embedding_dim,
            max_seq_length=self.patch_embedding.num_patches + 1
        )

        self.transformer_stack = TransformerStack(
            embedding_layer=dummy_embedding,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout=dropout,
            use_moe=True,
            num_experts=num_experts,
            experts_per_token=experts_per_token
        )

        self.final_gamma = jnp.ones((embedding_dim,))
        self.final_beta = jnp.zeros((embedding_dim,))

    @staticmethod
    @jax.jit
    def fwd(params:dict,
            images:jnp.ndarray,
            ) -> tuple[jnp.ndarray, float]:
        """
        Forward pass through ViT encoder

        Args:
            params (dict): Dictionary containing parameters of ViT model.
            images (jnp.ndarray): Batch of images.
            num_heads (int): Number of attention heads.
            num_blocks (int): Number of transformer blocks.
            head_dim (int): Dimension per head.
            embedding_dim (int): Embedding dimension.
            num_experts (int): Number of experts (unused when use_moe=False).
            experts_per_token (int): Experts per token (unused when use_moe=False).

        Returns:
            tuple: (encoded patch embeddings, total_aux_loss)
        """
        # Get patch embeddings
        patch_embeddings = PatchEmbedding.fwd(params['patch_embedding'], images)

        # Pass through transformer stack (much cleaner!)
        current = patch_embeddings
        total_aux_loss = 0.0

        current, total_aux_loss = TransformerBlock.fwd(
            params['transformer_stack'],
            patch_embeddings
        )

        final_output = TransformerBlock.layer_norm(
            current,
            params['final_ln']['gamma'],
            params['final_ln']['beta']
        )

        return final_output, total_aux_loss

    def get_params(self) -> dict:
        """
        Get all params for JAX functions

        Returns:
            dict: Contains parameters.
        """
        return {
            'patch_embedding': self.patch_embedding.get_params(),
            'transformer_stack': {
                'blocks': [block.get_params() for block in self.transformer_stack.blocks]
            },
            'final_ln': {
                'gamma': self.final_gamma,
                'beta': self.final_beta
            }
        }