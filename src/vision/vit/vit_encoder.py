"""
src/vision/vit/vit_encoder.py

Vision Transformer (ViT) encoder for processing image patches.
Reuses existing TransformerBlock.
"""

import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

import jax
import jax.numpy as jnp

from src.transformer.transformer_block import TransformerBlock
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

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim
        )

        # Creating dummy EmbeddingLayer for compatibility with TransformerBlock
        dummy_embedding = EmbeddingLayer(
            vocab_size=1000,
            embedding_dim=embedding_dim,
            max_seq_length=self.patch_embedding.num_patches + 1
        )

        # Create a stack of self-attention only transformer blocks
        self.blocks = [
            TransformerBlock(
                embedding_layer=dummy_embedding,
                num_heads=num_heads,
                num_blocks=num_blocks,
                dropout=dropout,
                use_moe=False
            )
        ]

        self.final_gamma = jnp.ones((embedding_dim,))
        self.final_beta = jnp.zeros((embedding_dim,))
