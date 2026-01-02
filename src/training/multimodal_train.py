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
from src.embeddings.embeddings import EmbeddingLayer

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

    # TODO: Add gradient computation
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

    def generate_from_image(self,
                            prompt="",
                            max_len=100,
                            temperature=0.7,
                            top_k=40,
                            repetition_penalty=1.2,
                            debug=False,
                            image=None
                            ) -> str:
        """
        Generate text from image and/or prompt.

        Args:
            prompt (str, optional): User prompt. Defaults to "".
            max_len (int, optional): Maximum response length (in tokens). Defaults to 100.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            top_k (int, optional): Top-k sampling. Defaults to 40.
            repetition_penalty (float, optional): Penalty for model repetition. Defaults to 1.2.
            image (np.ndarray, optional): Image to use in inference. Defaults to None.
            debug (bool, optional): Whether debug messages are active. Defaults to False.

        Returns:
            str: Model response.
        """
        if image is not None:
            if isinstance(image, str):
                image = PILImage.open(image).convert('RGB')
                image = image.resize((224, 224), PILImage.BICUBIC)
                image = np.array(image).astype(np.float16) / 255.0

            image_batch = jnp.array([image])
            vit_params = self.vit_encoder.get_params()
            image_embeddings, _ = ViTEncoder.fwd(vit_params, image_batch)
        else:
            image_embeddings = None

        if prompt:
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = self.tokenizer.encode("Describe this image.")
        prompt_length = len(token_ids)

        if debug:
            print(f"\nDEBUG: Encoded prompt: {token_ids}")
            print(f"DEBUG: Prompt length: {prompt_length}")
            print(f"DEBUG: Mode: {'Multimodal' if image is not None else 'Text'}")

        embed_params = self.embedding_layer.get_params()
        stack_params = [block.get_params() for block in self.transformer_stack.blocks]
        output_params = self.output_layer.get_params()
        final_ln_params = {
            'gamma': self.final_gamma,
            'beta': self.final_beta
        }

        with jax.default_device(jax.devices()[0]):
            for step_idx in range(max_len):
                batch_token_ids = jnp.array([token_ids], dtype=jnp.int16)

                text_embeddings, _ = EmbeddingLayer.fwd(
                    embed_params,
                    batch_token_ids
                )

                current = text_embeddings

                if image_embeddings is not None:
                    # Multimodal
                    for i, block in enumerate(self.transformer_stack.blocks):
                        block_params = stack_params[i]

                        block_params['cross_attn'] = self.cross_attentions[i].get_params()
                        block_params['gamma_cross'] = block.gamma_cross
                        block_params['beta_cross'] = block.beta_cross
                        # TODO: Something weird is happening here:
                            # the autocomplete does not exist for gamma and beta_cross,
                            # suggesting incorrect declaration

                        current, aux_loss, _ = TransformerBlock.fwd_with_x_attn(
                            block_params,
                            current,
                            image_embeddings,
                            num_heads=self.num_heads,
                            head_dim=self.embedding_dim // self.num_heads,
                            embedding_dim=self.embedding_dim,
                            num_experts=self.num_experts,
                            experts_per_token=self.experts_per_token,
                            dropout=self.dropout
                        )
                else:
                    total_aux_loss = 0.0

                    for i in range(self.num_blocks):
                        block_params = stack_params[i]
                        current, aux_loss = TransformerBlock.fwd(
                            block_params,
                            current,
                            self.num_heads,
                            self.embedding_dim // self.num_heads,
                            self.embedding_dim,
                            self.num_experts,
                            self.experts_per_token
                        )
                        total_aux_loss += aux_loss

                current = TransformerBlock.layer_norm(
                    current,
                    final_ln_params['gamma'],
                    final_ln_params['beta']
                )

                logits = self.output_layer.fwd(output_params, current) # TODO: Check if this works as expected

                next_logits = logits[0, -1] / temperature

                vocab_size = self.tokenizer.vocab_size

                # Repetition penalty
                if repetition_penalty != 1.0:
                    unique_tokens = jnp.array(list(set(token_ids)), dtype=jnp.int16)

                    penalty_mask = jnp.zeros(vocab_size, dtype=jnp.bool_)
                    penalty_mask = penalty_mask.at[unique_tokens].set(True)

                    penalties = jnp.where(
                        penalty_mask,
                        jnp.where(next_logits > 0, 1.0 /
                                repetition_penalty, repetition_penalty),
                        1.0
                    )
                    next_logits = next_logits * penalties

                # Top-k filtering
                top_k_indices = jnp.argsort(next_logits)[-top_k:]
                mask = jnp.ones_like(next_logits) * -jnp.inf
                mask = mask.at[top_k_indices].set(next_logits[top_k_indices])
                next_logits = mask

                # Sample next token
                probs = jax.nn.softmax(next_logits)
                probs_np = np.array(probs, dtype=np.float16)

                probs_np = probs_np / probs_np.sum()

                next_token = np.random.choice(len(probs_np), p=probs_np)
                next_token = int(next_token)

                if debug:
                    print(f"DEBUG: Token {len(token_ids) - prompt_length + 1}: {next_token} (prob: {probs_np[next_token]:.4f})")

                if next_token == self.tokenizer.eos_token_id:
                    if debug:
                        print(f"DEBUG: Hit EOS at position {len(token_ids) - prompt_length}")
                    break

                token_ids.append(next_token)

                print(self.tokenizer.decode([next_token]), end="", flush=True)

        generated = token_ids[prompt_length:]
        return "", generated

    def count_parameters(self) -> dict:
        """
        Count total trainable parameters including ViT encoder and cross-attention.

        Returns:
            dict: Dictionary with parameter counts by component and total
        """
        param_counts = super().count_parameters()

        # ==== ViT Encoder Params ====

        param_counts['vit'] = {
            'vit_patch_embedding': 0,
            'vit_attention': 0,
            'vit_feedforward': 0,
            'vit_ln': 0
        }

        # Patch embeddings
        param_counts[['vit']['vit_patch_embedding']] = 0
        param_counts[['vit']['vit_patch_embedding']] += self.vit_encoder.patch_embedding.projection.size
        param_counts[['vit']['vit_patch_embedding']] += self.vit_encoder.patch_embedding.cls_token.size
        param_counts[['vit']['vit_patch_embedding']] += self.vit_encoder.patch_embedding.positional_embeddings.size

        # ViT Transformer Blocks
        param_counts[['vit']['vit_attention']] = 0
        param_counts[['vit']['vit_feedforward']] = 0
        param_counts[['vit']['vit_ln']] = 0

        for block in self.vit_encoder.transformer_stack.blocks:
            # Attention
            param_counts[['vit']['vit_attention']] += block.attention_layer.W_Q.size
            param_counts[['vit']['vit_attention']] += block.attention_layer.W_K.size
            param_counts[['vit']['vit_attention']] += block.attention_layer.W_V.size
            param_counts[['vit']['vit_attention']] += block.attention_layer.W_O.size

            # FFN/MoE
            if block.use_moe:
                param_counts[['vit']['vit_feedforward']] += block.moe.count_params()
            else:
                param_counts[['vit']['vit_feedforward']] += block.ffn.W1.size
                param_counts[['vit']['vit_feedforward']] += block.ffn.B1.size
                param_counts[['vit']['vit_feedforward']] += block.ffn.W2.size
                param_counts[['vit']['vit_feedforward']] += block.ffn.B2.size

            # LN params
            param_counts[['vit']['vit_ln']] += block.gamma_1.size
            param_counts[['vit']['vit_ln']] += block.beta_1.size
            param_counts[['vit']['vit_ln']] += block.gamma_2.size
            param_counts[['vit']['vit_ln']] += block.beta_2.size

        # ViT final LN
        param_counts[['vit']['vit_ln']] += self.vit_encoder.final_gamma.size
        param_counts[['vit']['vit_ln']] += self.vit_encoder.final_beta.size

        # ====== X-Attn =====
        param_counts['x_attn'] = 0
        for x_attn in self.cross_attentions:
            param_counts['x_attn'] += x_attn.W_Q.size
            param_counts['x_attn'] += x_attn.W_K.size
            param_counts['x_attn'] += x_attn.W_V.size
            param_counts['x_attn'] += x_attn.W_O.size

        x_ln_params = 0
        x_ln_params += param_counts['x_attn']
        # ===== X-Attn LN =====
        # Add gamma_cross and beta_cross to LN count
        x_ln_params += sum(
            block.gamma_cross.size + block.beta_cross.size
            for block in self.transformer_stack.blocks
        )
        param_counts['layer_norm'] += x_ln_params

        # ===== Update total =====
        param_counts['total'] = sum(v for k, v in param_counts.items() if k != 'total')

        return param_counts