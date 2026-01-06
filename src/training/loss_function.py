"""
./src/training/loss_function.py

Loss function class that calculates loss of the model using Cross Entropy Loss. This shows how wrong
    the predictions are. The lower the number, the better the model is able to predict.
"""

import os
import sys

import jax # pylint: disable=no-member
import jax.numpy as jnp # pylint: disable=no-member
from jax import Array
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class CrossEntropyLoss:
    """
    Function that tells the model how wrong its predictions are.
    Cross-entropy measures the difference between the model's prediction
        (from output_layer) and the true next token.
    """
    def __init__(
            self,
            ignore_index=None,
            reduction='mean'
            ):
        """
        Initializing CrossEntropyLoss.

        Args:
            ignore_index (int, optional): Which index to ignore. Defaults to None.
            reduction (str, optional): Type of reduction, 'sum' or 'mean'. Defaults to 'mean'.
        """
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.grad_fn = jax.grad(self.fwd)

    @staticmethod
    @jax.jit
    def fwd(
        logits:jnp.ndarray,
        targets:jnp.ndarray,
        ignore_indices:list,
        ignore_index:int,
        reduction='mean',
        eos_weight=1.0,
        eos_token_id=None
        ):
        """
        Forward pass for the cross entropy loss function. Calculates the actual loss for each token.

        Args:
            logits (jnp.ndarray): Output of the OutputLayer class.
            targets (jnp.ndarray): The true "next-token" index that the model is supposed
                                   to predict. Shape: (batch, seq_len)
            ignore_indices (list): Multiple indices to ignore (e.g., [0, 20000] for padding and EOS.
            ignore_index (int): Single index to ignore.
            reduction (str, optional): Type of reduction, 'sum' or 'mean'. Defaults to 'mean'.
            eos_weight (float, optional): Weight for EOS token loss (e.g., 0.1 to downweight).
                                          Defaults to 1.0.
            eos_token_id (int, optional): EOS token ID to apply weight to.
                                          Required if eos_weight is specified. Defaults to None.

        Returns:
            Array: Loss of the model.

        Raises:
            TypeError
        """
        batch_size, seq_len, vocab_size = logits.shape

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        targets = jnp.clip(targets, 0, vocab_size - 1)

        selected_log_probs = jnp.take_along_axis(
            log_probs, targets[..., None], axis=-1
        )[...,0]

        # Handle both single ignore_index and multiple ignore_indices
        if ignore_indices is not None:
            # Create mask that ignores multiple indices
            mask = jnp.ones_like(targets, dtype=jnp.float32)
            for idx in ignore_indices:
                mask = mask * (targets != idx).astype(jnp.float32)
        elif ignore_index is not None:
            mask = (targets != ignore_index).astype(jnp.float32)
        else:
            mask = jnp.ones_like(targets, dtype=jnp.float32)

        # Apply EOS weighting if specified
        if eos_weight is not None and eos_token_id is not None:
            # Create EOS mask: 1.0 for non-EOS, eos_weight for EOS tokens
            eos_mask = jnp.where(targets == eos_token_id, eos_weight, 1.0).astype(jnp.float32)
            mask = mask * eos_mask

        neg_log_prob = -selected_log_probs * mask

        if reduction == 'mean':
            loss = neg_log_prob.sum()/mask.sum()
        elif reduction == 'sum':
            loss = neg_log_prob.sum()
        else:
            raise TypeError("Please enter either 'mean' or 'sum' as a reduction")

        return loss
