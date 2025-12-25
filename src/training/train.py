"""
JAX-based trainer for transformer language model with stacked blocks.

Key features:
- Multiple stacked transformer blocks for deeper architecture
- JAX autodiff for automatic gradient computation
- JIT compilation for faster training
- Multi-head attention (8 heads per block)

Architecture:
    Embeddings → TransformerStack → OutputLayer → Loss


src/training/train.py
"""

import os
import sys
import gc
import time as t
import pickle
import functools
from datetime import datetime

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as tree
from tqdm import tqdm

from src.embeddings.embeddings import EmbeddingLayer
from src.transformer.transformer_stack import TransformerStack
from src.transformer.transformer_block import TransformerBlock
from src.transformer.output_layer import OutputLayer
from src.training.loss_function import CrossEntropyLoss
from src.tokenizer.tiktoken_tokenizer import TikToken
from src.optimizers.adam import AdamNested
from src.transformer.mixture_of_experts import MOE
from api.paths import get_models_path


class Trainer:
    """
    JAX-based trainer for transformer language model with stacked blocks.

    Key features:
    - Multiple stacked transformer blocks for deeper architecture
    - JAX autodiff for automatic gradient computation
    - JIT compilation for faster training
    - Multi-head attention (8 heads per block)

    Architecture:
        Embeddings → TransformerStack → OutputLayer → Loss
    """

    def __init__(self,
                tokenizer,
                training_data=None,
                token_ids=None,
                lr=1e-4,
                num_blocks=8,
                num_heads=8,
                embedding_dim=512,
                max_seq_length=256,
                use_moe=True,
                num_experts=8,
                experts_per_token=2,
                use_lr_schedule=True,
                warmup_steps=500,
                dropout=0.0,
                min_lr=0.0,
                load_balance_coef=0.01
                ) -> None:
        """
        Initialize Trainer with model architecture.

        Args:
            tokenizer (object): Tokenizer instance (TikToken or BPETokenizer)
            training_data (list, optional): List of text strings for training. Defaults to None.
            lr (float, optional): Learning rate. Defaults to 1e-4.
            num_blocks (int, optional): Number of transformer blocks to stack. Defaults to 8.
            num_heads (int, optional): Number of attention heads per block.
            embedding_dim (int, optional): Embedding dimension.
                                            Must be divisible by num_heads and num_blocks.
                                            Defaults to 512.
            max_seq_length (int, optional): Maximum sequence length for chunking. Defaults to 256.
            use_moe (bool, optional): Whether to use MoE. Defaults to True.
            num_experts (int, optional): Total number of experts. Defaults to 8.
            experts_per_token (int, optional): How many experts are active per token. Defaults to 2.
            use_lr_schedule (bool, optional): Whether to use learning rate warmup and cosine decay.
                                                Defaults to True.
            warmup_steps (int, optional): Number of warmup steps. Defaults to 500.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            min_lr (float, optional): Minimum learning rate floor. Defaults to 0.0.
            load_balance_coef (float, optional): Weight for auxiliary load balancing loss.
                                                    Defaults to 0.01.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.use_lr_schedule = use_lr_schedule
        self.warmup_steps = warmup_steps
        self.dropout = dropout
        self.min_lr = min_lr

        # Validate that embedding_dim is divisible by num_heads
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")

        self.embedding_layer = EmbeddingLayer(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=embedding_dim,
            max_seq_length=max_seq_length,
            dropout=dropout
        )

        if token_ids and training_data is not None:
            token_ids = []
            for text in tqdm(training_data):
                ids = self.tokenizer.encode(text)
                ids.append(self.tokenizer.eos_token_id)
                self.token_ids.append(ids)

        self.token_ids = token_ids

        self.num_blocks = num_blocks
        self.num_heads = num_heads

        self.use_moe = use_moe
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.lbc = load_balance_coef

        self.transformer_stack = TransformerStack(
            self.embedding_layer,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout=dropout
        ) # TODO: Change this to be able to use MoE.

        self.output_layer = OutputLayer(self.embedding_layer)
        self.loss_fn = CrossEntropyLoss()

        self.final_gamma = jnp.ones(embedding_dim)
        self.final_beta = jnp.zeros(embedding_dim)

        self.lr = lr

        # Initialize training tracking variables
        self.training_history = {
            'losses': [],
            'learning_rates': [],
            'epochs_completed': 0,
            'total_steps': 0
        }

        # Initialize Adam optimizer with beta2=0.95 (nanoGPT value for better LLM training)
        self.optimizer = AdamNested(lr=lr, beta1=0.9, beta2=0.95, epsilon=1e-8, min_lr=min_lr)

        # Create JIT-compiled loss and gradient function
        self._compiled_loss_and_grad = self._create_jit_loss_fn()

        # Create JIT-compiled update function
        self._compiled_update = self._create_jit_update_fn()

        # Initialize optimizer state (Adam moments) using pytree structure
        # This avoids initialization overhead on first batch
        initial_params = self._flatten_params()
        self._adam_m = tree.tree_map(lambda p: jnp.zeros_like(p), initial_params)
        self._adam_v = tree.tree_map(lambda p: jnp.zeros_like(p), initial_params)

        # Capture static values so JAX treats them as concrete during tracing
        num_heads = self.num_heads
        head_dim = self.embedding_layer.embedding_dim // self.num_heads
        embedding_dim = self.embedding_layer.embedding_dim

        @jax.jit
        def fwd_jit(embed_params:dict,
                    stack_params:dict,
                    output_params:dict,
                    final_ln_params:dict,
                    token_ids:jnp.ndarray
                    ) -> tuple[jnp.ndarray, jnp.ndarray]:
            """
            Forward pass through entire model using JAX.

            Args:
                embed_params (dict): Embedding parameters
                stack_params (dict): Transformer stack parameters
                output_params (dict): Output layer parameters
                final_ln_params (dict): Final LayerNorm parameters
                token_ids (jnp.ndarray): Token IDs, shape (batch, seq_len)

            Returns:
                tuple:
                    - transformer_out (jnp.ndarray): Transformer output
                    - logits (jnp.ndarray): Logits
            """
            embeddings, _ = EmbeddingLayer.embedding_fwd(
                embed_params,
                token_ids
            )
            current = embeddings

            # Use the closed-over static values
            num_heads_local = num_heads
            head_dim_local = head_dim
            embedding_dim_local = embedding_dim

            for i in range(len(stack_params)):
                block_params = stack_params[i]
                current = TransformerBlock.fwd(
                    block_params,
                    current,
                    num_heads_local,
                    head_dim_local,
                    embedding_dim_local
                )

            # Apply final LayerNorm after all transformer blocks
            current = TransformerBlock.layer_norm(
                current,
                final_ln_params['gamma'],
                final_ln_params['beta']
            )

            logits = OutputLayer.fwd(output_params, current)

            return current, logits
        # Create a bound wrapper (no-op wrapper — fwd_jit already closes over static values)
        self._fwd = fwd_jit

    def _create_jit_loss_fn(self):
        """
        Create a JIT-compiled function for computing loss and gradients.
        This is created once during initialization for maximum performance.
        """
        num_heads = self.num_heads
        head_dim = self.embedding_layer.embedding_dim // self.num_heads
        embedding_dim = self.embedding_layer.embedding_dim
        num_blocks = self.num_blocks

        # Store tokenizer IDs as static values for JIT compilation
        padding_token_id = 0
        eos_token_id = self.tokenizer.eos_token_id

        @jax.jit
        def loss_and_grad_fn(embed_params, stack_params, output_params, final_ln_params, token_ids, targets):
            """JIT-compiled loss and gradient computation."""
            def loss_fn(embed_params, stack_params, output_params, final_ln_params):
                embeddings, _ = EmbeddingLayer.embedding_fwd(embed_params, token_ids)

                current = embeddings
                for i in range(num_blocks):
                    block_params = stack_params[i]
                    current = TransformerBlock.fwd(
                        block_params,
                        current,
                        num_heads,
                        head_dim,
                        embedding_dim
                    )

                # Apply final LayerNorm after all transformer blocks
                current = TransformerBlock.layer_norm(
                    current,
                    final_ln_params['gamma'],
                    final_ln_params['beta']
                )

                logits = OutputLayer.fwd(output_params, current)
                # Ignore padding (0) during loss calculation
                # Use ignore_index (scalar) instead of ignore_indices (list) for JIT compatibility
                loss = CrossEntropyLoss.fwd(
                    logits,
                    targets,
                    ignore_index=0,
                    eos_weight=1.0,  # Full weight for EOS tokens
                    eos_token_id=eos_token_id
                )
                return loss

            loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3))(
                embed_params, stack_params, output_params, final_ln_params
            )
            return loss, grads

        return loss_and_grad_fn

    def _create_jit_update_fn(self):
        """
        Create a JIT-compiled function for updating parameters with Adam using pytrees.
        This is much faster than the tuple-based approach.
        """
        beta1 = self.optimizer.beta1
        beta2 = self.optimizer.beta2
        lr = self.optimizer.lr
        epsilon = self.optimizer.epsilon

        @jax.jit
        def update_fn(params_pytree, grads_pytree, optimizer_state, t):
            """
            JIT-compiled Adam update using pytrees (works on nested structures).

            Args:
                params_pytree: Nested dict of parameters (from _flatten_params)
                grads_pytree: Nested dict of gradients (same structure)
                optimizer_state: (m_pytree, v_pytree) - nested dicts of moment estimates
                t: Timestep

            Returns:
                (updated_params_pytree, new_optimizer_state)
            """
            m_pytree, v_pytree = optimizer_state

            def adam_update_leaf(param, grad, m, v):
                """Apply Adam update to a single parameter array."""
                # Update biased first moment
                m_new = beta1 * m + (1 - beta1) * grad

                # Update biased second moment
                v_new = beta2 * v + (1 - beta2) * (grad ** 2)

                # Bias correction
                m_hat = m_new / (1 - beta1 ** t)
                v_hat = v_new / (1 - beta2 ** t)

                # Update parameters
                param_new = param - lr * m_hat / (jnp.sqrt(v_hat) + epsilon)

                return param_new, m_new, v_new

            # Apply adam_update_leaf to every leaf in the pytree
            # This returns a pytree of tuples (param_new, m_new, v_new)
            result_pytree = tree.tree_map(
                adam_update_leaf,
                params_pytree,
                grads_pytree,
                m_pytree,
                v_pytree
            )

            # Unzip the tuples to get separate pytrees
            # JAX treats tuples as PyTree nodes, so we need to use tree_transpose
            # to properly unpack them
            from jax.tree_util import tree_transpose, tree_structure

            # Get the structure of the outer pytree (params) and inner structure (tuple of 3)
            outer_treedef = tree_structure(params_pytree)
            inner_treedef = tree_structure((0, 0, 0))  # 3-tuple structure

            # Transpose: outer structure of dicts/lists, inner structure of 3-tuples
            # -> inner structure of 3-tuples, outer structure of dicts/lists
            transposed = tree_transpose(outer_treedef, inner_treedef, result_pytree)

            # Now transposed is a 3-tuple of pytrees
            updated_params, updated_m, updated_v = transposed

            return updated_params, (updated_m, updated_v)

        return update_fn

    def _flatten_params(self):
        """
        Get all parameters as a pytree (tuple structure).
        This structure matches the gradient structure from compute_loss_and_grads.

        Returns:
            tuple: Nested tuple of all model parameters (embeddings, stack, output)
        """
        return (
            self.embedding_layer.get_params(),
            [block.get_params() for block in self.transformer_stack.blocks],
            self.output_layer.get_params(),
            {'gamma': self.final_gamma, 'beta': self.final_beta}
        )

    def _unflatten_params(self, params):
        """
        Set all parameters from a pytree (tuple structure).

        Args:
            params (tuple): Nested tuple of parameters matching _flatten_params structure
                           (embeddings_dict, stack_list, output_dict)
        """
        embeddings_dict, stack_list, output_dict, final_ln_dict = params

        # Update embedding layer
        self.embedding_layer.embeddings = embeddings_dict['embeddings']
        self.embedding_layer.positional_encodings = embeddings_dict['positional_encodings']

        # Update transformer stack
        for i, block_params in enumerate(stack_list):
            block = self.transformer_stack.blocks[i]
            # Update attention
            block.attention_layer.W_Q = block_params['attn']['W_Q']
            block.attention_layer.W_K = block_params['attn']['W_K']
            block.attention_layer.W_V = block_params['attn']['W_V']
            block.attention_layer.W_O = block_params['attn']['W_O']
            # Update FFN
            block.ffn.W1 = block_params['ffn']['W1']
            block.ffn.B1 = block_params['ffn']['B1']
            block.ffn.W2 = block_params['ffn']['W2']
            block.ffn.B2 = block_params['ffn']['B2']
            # Update LayerNorm
            block.gamma_1 = block_params['gamma_1']
            block.beta_1 = block_params['beta_1']
            block.gamma_2 = block_params['gamma_2']
            block.beta_2 = block_params['beta_2']

        # Update output layer
        self.output_layer.W_out = output_dict['W_out']
        self.output_layer.b_out = output_dict['b_out']

        self.final_gamma = final_ln_dict['gamma']
        self.final_beta = final_ln_dict['beta']

    def fwd(self, *args, **kwargs):
        return self._fwd(*args, **kwargs)

    def compute_loss_and_grads(self, token_ids, targets):
        """
        Compute loss and ALL gradients using JIT-compiled JAX autodiff.

        Args:
            token_ids (jnp.ndarray): Input token IDs, shape (batch, seq_len)
            targets (jnp.ndarray): Target token IDs, shape (batch, seq_len)

        Returns:
            tuple: (loss, all_grads)
        """
        embed_params = self.embedding_layer.get_params()
        stack_params = [block.get_params() for block in self.transformer_stack.blocks]
        output_params = self.output_layer.get_params()
        final_ln_params = {'gamma': self.final_gamma, 'beta': self.final_beta}

        # Use the pre-compiled JIT function
        loss, grads = self._compiled_loss_and_grad(
            embed_params, stack_params, output_params, final_ln_params, token_ids, targets
        )

        embed_grads, stack_grads, output_grads, final_ln_grads = grads

        return loss, {
            'embeddings': embed_grads,
            'stack': stack_grads,
            'output': output_grads,
            'final_ln': final_ln_grads
        }

    def update_params(self, grads):
        """
        Update all parameters using JIT-compiled Adam optimizer with pytrees.
        This is much faster than the previous list-based approach.

        Args:
            grads (dict): Gradients from compute_loss_and_grads()
                         Format: {'embeddings': dict, 'stack': list, 'output': dict}
        """
        # Get current parameters as pytree
        params_pytree = self._flatten_params()

        # Convert grads dict to tuple structure matching params_pytree
        grads_pytree = (
            grads['embeddings'],  # This is a dict {'embeddings': ..., 'positional_encodings': ...}
            grads['stack'],        # This is a list of dicts
            grads['output'],        # This is a dict
            grads['final_ln']
        )

        # Increment timestep
        self.optimizer.t += 1

        # Run JIT-compiled update (all parameter updates happen in one JIT call)
        optimizer_state = (self._adam_m, self._adam_v)

        updated_params, new_state = self._compiled_update(
            params_pytree, grads_pytree, optimizer_state, self.optimizer.t
        )

        # Update optimizer state
        self._adam_m, self._adam_v = new_state

        # Unpack updated parameters back to model
        self._unflatten_params(updated_params)
        self.output_layer.W_out = self.embedding_layer.embeddings.T # WEIGHT TYING WAS MESSING UP MY LOSS ASASDJASGAKJDHASAJKDH
        # Anyways force weight tying by making W_out ALWAYS = embeddings.T

        self.output_layer.b_out = jnp.clip(self.output_layer.b_out, -10.0, 10.0)


    def _get_timestamped_checkpoint_path(self, base_path):
        """
        Generate a timestamped checkpoint path.

        Args:
            base_path (str): Base checkpoint path (e.g., "artifacts/training_logs/checkpoint.pkl")

        Returns:
            str: Timestamped path (e.g., "artifacts/training_logs/checkpoint_2025-01-11_14-30-45.pkl")
        """
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Split the path into directory, filename, and extension
        directory = os.path.dirname(base_path)
        filename = os.path.basename(base_path)
        name, ext = os.path.splitext(filename)

        # Create timestamped filename
        timestamped_filename = f"{name}_{timestamp}{ext}"
        timestamped_path = os.path.join(directory, timestamped_filename)

        return timestamped_path

    def train(self, epochs=10, batch_size=20, checkpoint_path=None, save_every=10, prompt=""):
        """
        Train the model with JAX autodiff.
        Automatically saves checkpoints with timestamps.

        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            checkpoint_path (str): Base path for checkpoints (timestamp will be added)
            save_every (int): Save checkpoint every N epochs
        """
        # Ensure checkpoint_path uses centralized models directory when not provided
        if checkpoint_path is None:
            checkpoint_path = get_models_path("training_logs.pkl")

        # Generate timestamped checkpoint path at start of training
        timestamped_checkpoint = self._get_timestamped_checkpoint_path(checkpoint_path)
        print(f"Checkpoints will be saved to: {timestamped_checkpoint}")

        # print(f"Creating batches with batch_size={batch_size}...")
        batches = self.create_batches(batch_size)
        # print(f"Created {len(batches)} batches")

        # Configure learning rate schedule if enabled
        if self.use_lr_schedule:
            new_steps = epochs * len(batches)
            current_step = self.optimizer.t
            total_steps = current_step + new_steps
            self.optimizer.warmup_steps = self.warmup_steps
            self.optimizer.total_steps = total_steps
            self.optimizer.schedule = 'warmup_cosine'
            print(f"Learning rate schedule enabled:")
            print(f"  - Current step: {current_step}")
            print(f"  - Warmup steps: {self.warmup_steps}")
            print(f"  - New steps: {new_steps}")
            print(f"  - Total steps: {total_steps}")
            print(f"  - Base LR: {self.lr}")
            print(f"  - Min LR: {self.min_lr}")
        else:
            print(f"Using constant learning rate: {self.lr}")

        gc.disable()

        try:
            for epoch in range(epochs):
                total_loss = 0
                total_lr = 0  # Track sum of learning rates
                print(f"\nStarting epoch {epoch+1}/{epochs}")

                for batch_idx, batch in enumerate(tqdm(batches, desc=f"Epoch {epoch+1}/{epochs}", leave=False, file=sys.stderr)):
                    start_time = t.time()

                    # print(jax.default_backend())

                    # print(f"[Epoch {epoch+1}, Batch {batch_idx+1}/{len(batches)}] Converting to JAX array...")
                    # Convert to JAX array
                    batch_jax = jnp.array(batch, dtype=jnp.int32)
                    # print(batch_jax)
                    # print(f"    Batch #{batch_idx}")
                    # print(f"  Shape: {batch_jax.shape}")

                    # Create targets (shift by 1 position)
                    input_tokens = batch_jax[:, :-1]
                    target_tokens = batch_jax[:, 1:]
                    # print(f"  Input shape: {input_tokens.shape}, Target shape: {target_tokens.shape}")

                    # print(f"  Computing loss and gradients (JIT compiling on first batch)...")
                    loss, grads = self.compute_loss_and_grads(input_tokens, target_tokens)

                    grads_pytree = (grads['embeddings'], grads['stack'], grads['output'], grads['final_ln'])

                    global_norm = jnp.sqrt(sum(
                        jnp.sum(jnp.square(g))
                        for g in jax.tree_util.tree_leaves(grads_pytree)
                    ))

                    max_norm = 1.0
                    clip_coef = jnp.minimum(1.0, max_norm / (global_norm + 1e-6))

                    grads_pytree_clipped = jax.tree_util.tree_map(
                        lambda g: g * clip_coef,
                        grads_pytree
                    )

                    grads = {
                        'embeddings': grads_pytree_clipped[0],
                        'stack': grads_pytree_clipped[1],
                        'output': grads_pytree_clipped[2],
                        'final_ln': grads_pytree_clipped[3]
                    }
                    # print(f"  Loss computed: {float(loss):.4f}")

                    # Update parameters (this increments self.optimizer.t)
                    # print(f"  Updating parameters...")
                    self.update_params(grads)

                    # Update learning rate AFTER optimizer step (when t is correct)
                    if self.use_lr_schedule:
                        current_lr = self.optimizer.get_lr()
                        self.optimizer.lr = current_lr
                        total_lr += current_lr
                    else:
                        total_lr += self.optimizer.lr

                    total_loss += float(loss)  # Convert JAX scalar to Python float

                    batch_time = t.time() - start_time
                    # print(f"  Batch complete in {batch_time:.2f}s")

                avg_loss = total_loss / len(batches)
                avg_lr = total_lr / len(batches)
                final_lr = self.optimizer.lr if not self.use_lr_schedule else self.optimizer.get_lr()
                print(f"Epoch {epoch+1}/{epochs} complete. Avg loss: {avg_loss:.4f}, Avg LR: {avg_lr:.6f}, Final LR: {final_lr:.6f}")

                # Track training history
                self.training_history['losses'].append(avg_loss)
                self.training_history['learning_rates'].append(float(final_lr))
                self.training_history['epochs_completed'] += 1
                self.training_history['total_steps'] = int(self.optimizer.t)

                # Save checkpoint with timestamp
                if (epoch + 1) % save_every == 0:
                    print(f"Saving checkpoint at epoch {epoch+1}...")
                    self.save_checkpoint(timestamped_checkpoint)

                gc.collect()

                if prompt and (epoch + 1) % 5 == 0: # Run generate every 5 epochs - avoids 100s delay every epoch.
                    generated_text = self.generate(
                        prompt,
                        max_length=150,
                    )
                    print(f"Saving checkpoint at epoch {epoch +1}")
                    self.save_checkpoint(get_models_path(f"alpaca_epoch{epoch + 1}"))

                    print("Prompt: \n", prompt)
                    print("Generated: ", generated_text[0])

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
            print("Saving checkpoint before exit...")
            self.save_checkpoint(timestamped_checkpoint)
            print(f"Checkpoint saved to {timestamped_checkpoint}")
            print(f"Training stopped at epoch {epoch+1}/{epochs}")
            gc.enable()
            raise

        gc.enable()
        print("Training complete! Saving final checkpoint...")
        self.save_checkpoint(timestamped_checkpoint)

    def count_parameters(self):
        """
        Count total number of trainable parameters in the model.

        Returns:
            dict: Dictionary with parameter counts by component and total
        """
        param_counts = {
            'embedding': 0,
            'attention': 0,
            'feedforward': 0,
            'layer_norm': 0,
            'output': 0,
            'total': 0
        }

        # Embedding layer parameters
        param_counts['embedding'] += self.embedding_layer.embeddings.size
        param_counts['embedding'] += self.embedding_layer.positional_encodings.size

        # For each transformer block
        for block in self.transformer_stack.blocks:
            # Attention parameters
            param_counts['attention'] += block.attention_layer.W_Q.size
            param_counts['attention'] += block.attention_layer.W_K.size
            param_counts['attention'] += block.attention_layer.W_V.size
            param_counts['attention'] += block.attention_layer.W_O.size

            # Feedforward parameters
            param_counts['feedforward'] += block.ffn.W1.size
            param_counts['feedforward'] += block.ffn.B1.size
            param_counts['feedforward'] += block.ffn.W2.size
            param_counts['feedforward'] += block.ffn.B2.size

            # Layer normalization parameters
            param_counts['layer_norm'] += block.gamma_1.size
            param_counts['layer_norm'] += block.beta_1.size
            param_counts['layer_norm'] += block.gamma_2.size
            param_counts['layer_norm'] += block.beta_2.size

        # Output layer parameters
        param_counts['output'] += self.output_layer.W_out.size
        param_counts['output'] += self.output_layer.b_out.size

        # Total
        param_counts['total'] = sum(param_counts.values()) - param_counts['total']

        return param_counts

    def print_model_summary(self):
        """Print a summary of the model architecture and parameter counts."""
        counts = self.count_parameters()

        print("="*60)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*60)

        # Print JAX device information
        devices = jax.devices()
        backend = jax.default_backend()
        print(f"JAX Backend:          {backend}")
        print(f"JAX Devices:          {devices}")
        print(f"Device Type:          {devices[0].device_kind if devices else 'Unknown'}")
        print("-"*60)

        print(f"Vocabulary Size:      {self.tokenizer.vocab_size:,}")
        print(f"Embedding Dimension:  {self.embedding_layer.embedding_dim}")
        print(f"Max Sequence Length:  {self.embedding_layer.max_seq_length}")
        print(f"Number of Blocks:     {self.num_blocks}")
        print(f"Number of Heads:      {self.num_heads}")
        print(f"FFN Hidden Dimension: {self.transformer_stack.blocks[0].ffn.ff_dim}")
        print("="*60)
        print("PARAMETER COUNTS")
        print("="*60)
        print(f"Embedding Layer:      {counts['embedding']:>12,} parameters")
        print(f"Attention Layers:     {counts['attention']:>12,} parameters")
        print(f"FeedForward Layers:   {counts['feedforward']:>12,} parameters")
        print(f"Layer Normalization:  {counts['layer_norm']:>12,} parameters")
        print(f"Output Layer:         {counts['output']:>12,} parameters")
        print("-"*60)
        print(f"TOTAL:                {counts['total']:>12,} parameters")
        print("="*60)

        # Calculate model size in MB (assuming float32)
        size_mb = (counts['total'] * 2) / (1024 * 1024)
        print(f"Model Size (float16): ~{size_mb:.2f} MB")
        print("="*60)

    def _generate_metadata(self):
        """Generate comprehensive metadata about the model and training."""
        from datetime import datetime

        # Calculate parameter counts
        counts = self.count_parameters()

        # Calculate dataset statistics
        if self.token_ids:
            total_tokens = sum(len(ids) for ids in self.token_ids)
            avg_tokens = total_tokens / len(self.token_ids)
            max_tokens = max(len(ids) for ids in self.token_ids)
            min_tokens = min(len(ids) for ids in self.token_ids)
        else:
            total_tokens = avg_tokens = max_tokens = min_tokens = 0

        metadata = {
            "model_info": {
                "name": "PyGPT",
                "description": "Transformer-based GPT model trained on instruction-following datasets",
                "version": "1.0",
                "created_date": datetime.now().strftime("%Y-%m-%d"),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "architecture": {
                "type": "Transformer GPT",
                "backend": "JAX",
                "total_parameters": counts['total'],
                "embedding_dim": self.embedding_dim,
                "num_blocks": self.num_blocks,
                "num_heads": self.num_heads,
                "vocab_size": self.tokenizer.vocab_size,
                "max_seq_length": self.max_seq_length,
                "ffn_hidden_dim": self.embedding_dim * 4,
                "dropout": self.dropout,
                "parameter_breakdown": {
                    "embedding_layer": counts['embedding'],
                    "attention_layers": counts['attention'],
                    "feedforward_layers": counts['feedforward'],
                    "layer_normalization": counts['layer_norm'],
                    "output_layer": counts['output']
                }
            },
            "training_data": {
                "tokenizer": type(self.tokenizer).__name__,
                "total_examples": len(self.token_ids) if self.token_ids else 0,
                "total_tokens": total_tokens,
                "avg_tokens_per_example": round(avg_tokens, 1),
                "min_tokens": min_tokens,
                "max_tokens": max_tokens
            },
            "training_config": {
                "batch_size": "varies",  # Not stored in trainer
                "base_lr": self.lr,
                "min_lr": self.min_lr,
                "lr_schedule": "Warmup + Cosine Decay" if self.use_lr_schedule else "Constant",
                "warmup_steps": self.warmup_steps if self.use_lr_schedule else 0,
                "optimizer": "AdamW",
                "beta1": 0.9,
                "beta2": 0.95,
                "epsilon": 1e-8
            },
            "training_history": {
                "epochs_completed": self.training_history['epochs_completed'],
                "total_steps": self.training_history['total_steps'],
                "losses": self.training_history['losses'][-10:] if self.training_history['losses'] else [],  # Last 10 losses
                "learning_rates": self.training_history['learning_rates'][-10:] if self.training_history['learning_rates'] else [],  # Last 10 LRs
                "initial_loss": self.training_history['losses'][0] if self.training_history['losses'] else None,
                "final_loss": self.training_history['losses'][-1] if self.training_history['losses'] else None
            },
            "hardware": {
                "backend": "JAX",
                "devices": [str(d) for d in jax.devices()],
                "precision": "float16"
            }
        }

        return metadata

    def save_checkpoint(self, path=None):
        if path is None:
            path = get_models_path("training_logs.pkl")
        """Save model parameters AND optimizer state to file (for resuming training)."""
        checkpoint = {
            'embeddings': self.embedding_layer.embeddings,
            'positional_encodings': self.embedding_layer.positional_encodings,
            'stack': [block.get_params() for block in self.transformer_stack.blocks],
            'output': self.output_layer.get_params(),
            'final_ln': {'gamma': self.final_gamma, 'beta': self.final_beta},
            'optimizer_t': self.optimizer.t,
            'config': {
                'num_blocks': self.num_blocks,
                'num_heads': self.num_heads,
                'lr': self.lr,
                'vocab_size': self.tokenizer.vocab_size,
                'embedding_dim': self.embedding_layer.embedding_dim
            },
            'training_history': self.training_history,
            'metadata': self._generate_metadata()
        }

        # Save optimizer state if it exists
        if hasattr(self, '_adam_m'):
            checkpoint['adam_m'] = self._adam_m
            checkpoint['adam_v'] = self._adam_v

        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    def save_model_only(self, path=None):
        if path is None:
            path = get_models_path("model.pkl")
        """Save ONLY model weights (smaller file, for inference only)."""
        model_state = {
            'embeddings': self.embedding_layer.embeddings,
            'positional_encodings': self.embedding_layer.positional_encodings,
            'stack': [block.get_params() for block in self.transformer_stack.blocks],
            'output': self.output_layer.get_params(),
            'final_ln': {'gamma': self.final_gamma, 'beta': self.final_beta},
            'config': {
                'num_blocks': self.num_blocks,
                'num_heads': self.num_heads,
                'vocab_size': self.tokenizer.vocab_size,
                'embedding_dim': self.embedding_layer.embedding_dim
            },
            'training_history': self.training_history,
            'metadata': self._generate_metadata()
        }

        with open(path, "wb") as f:
            pickle.dump(model_state, f)

        print(f"Model saved to {path} (weights only, no optimizer state)")

    def save_model_npz(self, path=None):
        if path is None:
            path = get_models_path("model.npz")
        """Save model weights as compressed NumPy arrays (smallest file size)."""
        import numpy as np

        # Collect all parameters as numpy arrays
        save_dict = {
            'embeddings': np.array(self.embedding_layer.embeddings),
            'positional_encodings': np.array(self.embedding_layer.positional_encodings),
            'output_W': np.array(self.output_layer.W_out),
            'output_b': np.array(self.output_layer.b_out),
            'final_ln_gamma': np.array(self.final_gamma),
            'final_ln_beta': np.array(self.final_beta),
        }

        # Add transformer blocks
        for i, block in enumerate(self.transformer_stack.blocks):
            params = block.get_params()
            save_dict[f'block_{i}_W_Q'] = np.array(params['attn']['W_Q'])
            save_dict[f'block_{i}_W_K'] = np.array(params['attn']['W_K'])
            save_dict[f'block_{i}_W_V'] = np.array(params['attn']['W_V'])
            save_dict[f'block_{i}_W_O'] = np.array(params['attn']['W_O'])
            save_dict[f'block_{i}_W1'] = np.array(params['ffn']['W1'])
            save_dict[f'block_{i}_B1'] = np.array(params['ffn']['B1'])
            save_dict[f'block_{i}_W2'] = np.array(params['ffn']['W2'])
            save_dict[f'block_{i}_B2'] = np.array(params['ffn']['B2'])
            save_dict[f'block_{i}_gamma_1'] = np.array(params['gamma_1'])
            save_dict[f'block_{i}_beta_1'] = np.array(params['beta_1'])
            save_dict[f'block_{i}_gamma_2'] = np.array(params['gamma_2'])
            save_dict[f'block_{i}_beta_2'] = np.array(params['beta_2'])

        # Save config as metadata
        config = {
            'num_blocks': self.num_blocks,
            'num_heads': self.num_heads,
            'vocab_size': self.tokenizer.vocab_size,
            'embedding_dim': self.embedding_layer.embedding_dim
        }

        # Save with compression
        np.savez_compressed(path, **save_dict, config=config)
        print(f"Model saved to {path} (compressed NPZ format)")

    def load_checkpoint(self, path="artifacts/model/training_logs.pkl"):
        """Load model parameters from file."""
        print(f"Loading checkpoint from {path}...")
        print("This may take 1-2 minutes for large files...")

        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        print("Checkpoint loaded! Restoring model parameters...")

        self.embedding_layer.embeddings = checkpoint['embeddings']
        self.embedding_layer.positional_encodings = checkpoint['positional_encodings']

        for i, block_params in enumerate(checkpoint['stack']):
            block = self.transformer_stack.blocks[i]
            # Update attention
            block.attention_layer.W_Q = block_params['attn']['W_Q']
            block.attention_layer.W_K = block_params['attn']['W_K']
            block.attention_layer.W_V = block_params['attn']['W_V']
            block.attention_layer.W_O = block_params['attn']['W_O']
            # Update FFN
            block.ffn.W1 = block_params['ffn']['W1']
            block.ffn.B1 = block_params['ffn']['B1']
            block.ffn.W2 = block_params['ffn']['W2']
            block.ffn.B2 = block_params['ffn']['B2']
            # Update LayerNorm
            block.gamma_1 = block_params['gamma_1']
            block.beta_1 = block_params['beta_1']
            block.gamma_2 = block_params['gamma_2']
            block.beta_2 = block_params['beta_2']

        self.output_layer.W_out = self.embedding_layer.embeddings.T
        self.output_layer.b_out = checkpoint['output']['b_out']

        # Load final LayerNorm if it exists
        if 'final_ln' in checkpoint:
            self.final_gamma = checkpoint['final_ln']['gamma']
            self.final_beta = checkpoint['final_ln']['beta']
        else:
            # Old checkpoint - initialize final LayerNorm
            print("Warning: Old checkpoint format without final_ln. Initializing final LayerNorm.")
            self.final_gamma = jnp.ones(self.embedding_layer.embedding_dim)
            self.final_beta = jnp.zeros(self.embedding_layer.embedding_dim)

        # Restore optimizer state
        if 'adam_m' in checkpoint:
            self._adam_m = checkpoint['adam_m']
            self._adam_v = checkpoint['adam_v']
            self.optimizer.t = checkpoint['optimizer_t']
            print("Loaded optimizer state from checkpoint")
        else:
            # Old checkpoint format or no optimizer state - reinitialize
            print("Warning: Old checkpoint format detected. Reinitializing optimizer state.")
            params_pytree = self._flatten_params()
            self._adam_m = tree.tree_map(lambda p: jnp.zeros_like(p), params_pytree)
            self._adam_v = tree.tree_map(lambda p: jnp.zeros_like(p), params_pytree)
            self.optimizer.t = 0

        # Restore training history if available
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
            print(f"Loaded training history: {self.training_history['epochs_completed']} epochs completed")
        else:
            print("Warning: No training history in checkpoint. Initializing new history.")
            self.training_history = {
                'losses': [],
                'learning_rates': [],
                'epochs_completed': 0,
                'total_steps': 0
            }

        print(f"Model parameters restored successfully!")
        if 'config' in checkpoint:
            print(f"Config: {checkpoint['config']}")

        # Display metadata if available
        if 'metadata' in checkpoint:
            meta = checkpoint['metadata']
            print(f"\nModel Metadata:")
            print(f"  Total Parameters: {meta['architecture']['total_parameters']:,}")
            if meta['training_history']['final_loss']:
                print(f"  Final Loss: {meta['training_history']['final_loss']:.4f}")

        print("Ready for inference!")

    def generate(self, prompt, max_length=50, temperature=0.7, top_k=40, repetition_penalty=1.2, debug=False):
        """
        Generate text using the trained model (JAX-based).

        Args:
            prompt (str): Input prompt
            max_length (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_k (int): Top-k filtering
            repetition_penalty (float): Penalty for repeating tokens
            debug (bool): Print debug information

        Returns:
            str: Generated text (excluding the prompt)
        """
        token_ids = self.tokenizer.encode(prompt)
        token_ids = [min(tid, self.tokenizer.vocab_size - 1) for tid in token_ids]
        prompt_length = len(token_ids)  # Track original prompt length

        if debug:
            print(f"\nDEBUG: Encoded prompt: {token_ids}")
            print(f"DEBUG: Prompt length: {prompt_length}")
            print(f"DEBUG: Vocab size: {self.tokenizer.vocab_size}")
            print(f"DEBUG: EOS token ID: {self.tokenizer.eos_token_id}")

        # Extract parameters once to avoid repeated dictionary lookups
        embed_params = self.embedding_layer.get_params()
        stack_params = [block.get_params() for block in self.transformer_stack.blocks]
        output_params = self.output_layer.get_params()
        final_ln_params = {'gamma': self.final_gamma, 'beta': self.final_beta}

        # Force all operations to run on GPU if available
        with jax.default_device(jax.devices()[0]):
            for step_idx in range(max_length):
                # Convert to JAX array (JAX will automatically use GPU if available)
                batch_token_ids = jnp.array([token_ids], dtype=jnp.int32)

                # Forward pass
                transformer_out, logits = self.fwd(
                    embed_params,
                    stack_params,
                    output_params,
                    final_ln_params,
                    batch_token_ids
                )

                # Get logits for last token (keep as float32 for numerical stability)
                next_logits = logits[0, -1] / temperature

                # CRITICAL: Mask out invalid tokens (beyond vocab_size)
                # This ensures we NEVER sample invalid token IDs
                vocab_size = self.tokenizer.vocab_size

                if step_idx == 0 and debug:  # First token only
                    print(f"\nDEBUG First Generation Step:")
                    print(f"  Logits shape: {next_logits.shape}")
                    print(f"  Vocab size: {vocab_size}")
                    print(f"  Top 10 logit values: {jnp.sort(next_logits)[-10:]}")
                    print(f"  Top 10 token indices: {jnp.argsort(next_logits)[-10:]}")
                if len(next_logits) > vocab_size:
                    # Set logits for invalid tokens to -inf (probability = 0)
                    next_logits = next_logits.at[vocab_size:].set(-jnp.inf)

                # Repetition penalty (optimized - vectorized operation)
                if repetition_penalty != 1.0:
                    unique_tokens = jnp.array(list(set(token_ids)), dtype=jnp.int32)
                    # Filter out tokens >= vocab_size
                    unique_tokens = unique_tokens[unique_tokens < vocab_size]

                    # Apply penalty in a vectorized way
                    penalty_mask = jnp.zeros(vocab_size, dtype=jnp.bool_)
                    penalty_mask = penalty_mask.at[unique_tokens].set(True)

                    # Vectorized penalty application
                    penalties = jnp.where(
                        penalty_mask,
                        jnp.where(next_logits > 0, 1.0 / repetition_penalty, repetition_penalty),
                        1.0
                    )
                    next_logits = next_logits * penalties

                # Top-k filtering
                if top_k is not None:
                    top_k_indices = jnp.argsort(next_logits)[-top_k:]
                    mask = jnp.ones_like(next_logits) * -jnp.inf
                    mask = mask.at[top_k_indices].set(next_logits[top_k_indices])
                    next_logits = mask

                # Sample (optimized - keep in float32 for stability)
                probs = jax.nn.softmax(next_logits)
                probs_np = np.array(probs, dtype=np.float32)

                # Normalize to ensure valid probability distribution
                probs_np = probs_np / probs_np.sum()

                next_token = np.random.choice(len(probs_np), p=probs_np)
                next_token = int(next_token)

                if next_token >= vocab_size:
                    next_token = vocab_size - 1

                if debug:
                    print(f"DEBUG: Generated token {len(token_ids) - prompt_length + 1}: {next_token} (prob: {probs_np[next_token]:.4f})")
                    top_5_indices = np.argsort(probs_np)[-5:][::-1]
                    print(f"DEBUG: Top 5 tokens: {[(i, probs_np[i]) for i in top_5_indices]}")

                # Check for EOS BEFORE appending to avoid including it in output
                if next_token == self.tokenizer.eos_token_id:
                    if debug:
                        print(f"DEBUG: Hit EOS token at position {len(token_ids) - prompt_length}")
                    break

                token_ids.append(next_token)

                # Stream the token as it's generated
                print(self.tokenizer.decode([next_token]), end="", flush=True)

        # Return the generated token IDs
        generated_token_ids = token_ids[prompt_length:]
        return "", generated_token_ids

    # def create_batches(self, batch_size=100):
    #     """Create padded batches."""
    #     batches = []
    #     for i in range(0, len(self.token_ids), batch_size):
    #         batch = self.token_ids[i:i+batch_size]
    #         max_len = max(len(seq) for seq in batch)
    #         padded_batches = [
    #             seq + [0] * (max_len - len(seq))
    #             for seq in batch
    #         ]
    #         batches.append(np.array(padded_batches))
    #     return batches
    
    def create_batches(self, batch_size=100):
        """Create padded batches."""
        fixed_len = self.max_seq_length
        batches = []
        for i in range(0, len(self.token_ids), batch_size):
            batch = self.token_ids[i:i+batch_size]
            padded_batches = []
            # max_len = max(len(seq) for seq in batch)
            for seq in batch:
                if len(seq) > fixed_len:
                    seq = seq[:fixed_len]
                padded_batches.append(seq + [0] * (fixed_len - len(seq)))
            batches.append(np.array(padded_batches))
        return batches
        
    
    def extend_training(self, checkpoint_path, epochs=10, batch_size=20, save_every=5, prompt=""):
        print(f"Loading checkpoint from {checkpoint_path}...")
        self.load_checkpoint(checkpoint_path)

        base_name = "artifacts/model/combined_extend.pkl"
        new_checkpoint = self._get_timestamped_checkpoint_path(base_name)
        print(f"Extended training will save to {new_checkpoint}")

        print(f"Continuing training for {epochs} additional epochs...")
        self.train(
            epochs=epochs,
            batch_size=batch_size,
            checkpoint_path=new_checkpoint,
            save_every=save_every,
            prompt=prompt
        )

        print(f"Extended training to {new_checkpoint}")




def main():
    tokenizer = TikToken()
    print(f"Loaded TikToken tokenizer with vocab size: {tokenizer.vocab_size}")

    user_input = ["Hello world"]
    trainer = Trainer(tokenizer, user_input, num_blocks=12, num_heads=12)

    trainer.print_model_summary()

if __name__ == '__main__':
    main()
