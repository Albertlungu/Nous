"""
src/training/multimodal_train.py

Trainer for multimodal vision->language and vice versa.
This class will extend the existing Trainer to be able to handle image inputs.
"""

import os
import sys
from functools import partial
from PIL import Image as PILImage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import jax
import jax.numpy as jnp
import numpy as np

from src.training.train import Trainer
from src.vision.vit.vit_encoder import ViTEncoder
from src.transformer.cross_attention import CrossAttention
from src.transformer.transformer_block import TransformerBlock
from src.training.loss_function import CrossEntropyLoss

class MultimodalTrainer(Trainer):
    """
    Trainer made to handle both text and image inputs.

    Args:
        Trainer (object): Trainer class.
    """
    def __init__(self,
                 # Tokenizer
                 tokenizer,
                 training_data=None,
                 token_ids=None,
                 # Basic transformer
                 num_blocks=8,
                 num_heads=8,
                 embedding_dim=512,
                 max_seq_length=256,
                 # MoE
                 use_moe=True,
                 num_experts=8,
                 experts_per_token=2,
                 load_balance_coef=0.01,
                 # Vision
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 vit_num_blocks=8,
                 # Misc
                 lr=1e-4,
                 min_lr=0.0,
                 use_lr_schedule=True,
                 warmup_steps=500,
                 dropout=0.0
                 ) -> None:
        """
        Initializing the multimodal trainer.

        Args:
            tokenizer (object): Tokenizer.
            training_data (list, optional): The non-tokenized training data. Defaults to None.
            token_ids (list, optional): The tokenized training data as token ids. Defaults to None.

            num_blocks (int, optional): Number of transformer blocks. Defaults to 8.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            embedding_dim (int, optional): Embedding dimension. Defaults to 512.
            max_seq_length (int, optional): Maximum sequence length. Defaults to 256.

            use_moe (bool, optional): Whether or not to use MoE. Defaults to True.
            num_experts (int, optional): Total number of experts. Defaults to 8.
            experts_per_token (int, optional): Experts active per token. Defaults to 2.
            load_balance_coef (float, optional): Weight for auxiliary load balancing loss.
                                                 Defaults to 0.01.

            image_size (int, optional): Image size in pixels. Defaults to 224.
            patch_size (int, optional): Patch size. Defaults to 16.
            in_channels (int, optional): Number of in channels (3 for RGB). Defaults to 3.
            vit_num_blocks (int, optional): Number of transformer blocks in the ViT. Defaults to 8.

            lr (float, optional): Learning rate. Defaults to 1e-4.
            min_lr (float, optional): Minimum learning rate floor.. Defaults to 0.0.
            use_lr_schedule (bool, optional): Whether or not to use learning rate schedule.
                                              Defaults to True.
            warmup_steps (int, optional): Number of warmup steps. Defaults to 500.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        """

        self.vit_num_blocks = vit_num_blocks
        # Initializing the base trainer
        super().__init__(
            tokenizer=tokenizer,
            training_data=training_data,
            token_ids=token_ids,
            # Basic transformer
            num_blocks=num_blocks,
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            max_seq_length=max_seq_length,
            # MoE
            use_moe=use_moe,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            load_balance_coef=load_balance_coef,
            # Misc
            lr=lr,
            min_lr=min_lr,
            use_lr_schedule=use_lr_schedule,
            warmup_steps=warmup_steps,
            dropout=dropout
        )

        # Add ViT encoder for images
        self.vit_encoder = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            num_blocks=vit_num_blocks,
            num_heads=num_heads,
            dropout=dropout
        )

        self.cross_attentions = [
            CrossAttention(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                num_blocks=num_blocks,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ]

        for block in self.transformer_stack.blocks:
            block.gamma_cross = jnp.ones((embedding_dim,))
            block.beta_cross = jnp.zeros((embedding_dim,))

    def compute_loss_and_grads_multimodal(self,
                                          images:jnp.ndarray,
                                          token_ids:jnp.ndarray,
                                          targets:jnp.ndarray
                                          ) -> tuple[float, dict]:
        """
        Compute loss and gradients for image->text gen

        Args:
            images (jnp.ndarray): Batch of images.
                                  Shape: (batch, height, width, channels)
            token_ids (jnp.ndarray): Input text tokens.
            targets (jnp.ndarray): Target next tokens.

        Returns:
            tuple[float, dict]: Tuple containing the model's loss and the grads dictionary.
        """

        # Get ViT encoder outputs (image features)
        vit_params = self.vit_encoder.get_params()
        image_embeddings = ViTEncoder.fwd(
            vit_params,
            images,
            num_heads=self.num_heads,
            head_dim=self.embedding_dim // self.num_heads,
            embedding_dim=self.embedding_dim,
            num_blocks=self.vit_num_blocks
        )

        # Get text embeddings
        embed_params = self.embedding_layer.get_params()
        text_embeddings, _ = self.embedding_layer.embedding_fwd(
            embed_params,
            token_ids
        )

        # Fwd through decoder with x-attn
        current = text_embeddings
        for i, block in enumerate(self.transformer_stack.blocks):
            block_params = block.get_params()

            block_params['cross_attn'] = self.cross_attentions[i].get_params()
            block_params['gamma_cross'] = block.gamma_cross
            block_params['beta_cross'] = block.beta_cross
            # TODO: Make sure the above dict entries actually exist

            current, aux_loss, _ = TransformerBlock.fwd_with_x_attn(
                block_params,
                current,
                image_embeddings,
                num_heads=self.num_heads,
                head_dim=self.embedding_dim // self.num_heads,
                embedding_dim=self.embedding_dim
            )

        # Apply final LayerNorm
        current = TransformerBlock.layer_norm(
            current,
            self.final_gamma,
            self.final_beta
        )

        # Output layer
        logits = self.output_layer.fwd(
            self.output_layer.get_params(),
            current
        )

        # Calculate loss
        loss = CrossEntropyLoss.fwd(
            logits,
            targets,
            ignore_index=0,
            eos_weight=1.0,
            eos_token_id=self.tokenizer.eos_token_id
        )

        return loss