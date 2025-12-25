"""
src/transformer/feed_forward.py

Defines the FeedForward class, a two-layer feed-forward network used in transformer architectures.
The network applies a linear transformation, a gelu activation, and another linear transformation to
token embeddings.

Provides:
- Initialization of weights and biases with proper scaling.
- Forward passes for both static and instance-based computation.
- Gradient computation for training with JAX autodiff.
- Optional dropout support during training.
"""

import os
import sys

import jax # pylint: disable=no-member
import jax.numpy as jnp # pylint: disable=no-member

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.embeddings.embeddings import EmbeddingLayer
from src.training.loss_function import CrossEntropyLoss

class FeedForward():
    """
    A FeedForward neural network module used within transformer architectures.

    This class implements a two-layer feed-forward network with gelu activation.
    It takes token embeddings as input and applies a linear transformation followed
    by a non-linear activation and another linear transformation to produce the output.

    Key components:
    - Two sets of weights and biases (W1, B1 for the first layer, W2, B2 for the second layer).
    - gelu activation function.
    - Forward and backward passes for training with gradient descent.
    """


    def __init__(self, embeddings: EmbeddingLayer, ff_dim = 0, num_blocks=1, dropout=0.0) -> None:
        """
        Initializes the FeedForward network.

        Args:
            token_ids (list): List of token IDs representing input sequences.
            embeddings (EmbeddingLayer): An instance of EmbeddingLayer to convert token
                IDs to embeddings.

        Attributes:
            embedding_dim (int): Dimensionality of the embeddings.
            ff_dim (int): Dimensionality of the feed-forward hidden layer (4 times embedding_dim).
            W1 (jnp.ndarray): Weight matrix for the first linear layer.
                Shape (embedding_dim, ff_dim).
            B1 (jnp.ndarray): Bias vector for the first linear layer. Shape: (ff_dim).
            W2 (jnp.ndarray): Weight matrix for the second linear layer
                Shape: (ff_dim, embedding_dim).
            B2 (jnp.ndarray): Bias vector for the second linear layer Shape: (embedding_dim).
            ff_input (jnp.ndarray): Input embeddings to the feed-forward network.
        """
        # Use the actual embedding dimension from the embeddings instance, not the class default
        self.embedding_dim = embeddings.embedding_dim
        self.ff_dim = ff_dim or self.embedding_dim * 4 # Feed Forward dimension
        self.dropout = dropout

        self.key = jax.random.PRNGKey(0)

        k1, k2 = jax.random.split(self.key)

        self.cross_entropy = CrossEntropyLoss()
        self.loss_fn = self.cross_entropy.fwd

        # Layers with proper initialization
        scale = 0.02
        self.W1 = jax.random.normal(k1, (self.embedding_dim, self.ff_dim)) * scale # Weight first layer
        self.B1 = jnp.zeros(self.ff_dim) # Bias first layer

        # W2 is a residual projection, scale by depth
        residual_scale = scale / jnp.sqrt(2.0 * num_blocks)
        self.W2 = jax.random.normal(k2, (self.ff_dim, self.embedding_dim)) * residual_scale # Weight second layer with residual scaling
        self.B2 = jnp.zeros(self.embedding_dim) # Bias second layer

    @staticmethod
    def gelu(x) -> jnp.ndarray:
        """
        gelu activation function

        Args:
            x (jnp.ndarray): array of vectors to go through activation function (3D matrix)

        Returns:
            jnp.ndarray: activated layer from hidden layer
        """
        return 0.5 * x * (1+jnp.tanh(jnp.sqrt(2/jnp.pi) * (x + 0.044715 * x**3)))

    @staticmethod
    def relu(x) -> jnp.ndarray:
        """Basically gelu but simpler

        Args:
            x (jnp.ndarray): array of vectors to go through activation function (3D matrix)

        Returns:
            jnp.ndarray: activated layer from hidden layer
        """
        return jnp.maximum(0, x)

    @staticmethod
    def fwd(params, x, dropout=0.0, training=True, rng_key=None) -> jnp.ndarray:
        """
        Static forward pass for use in JAX autodiff (called from TransformerBlock.fwd).

        Args:
            params (dict): Dictionary containing 'W1', 'B1', 'W2', 'B2'
            x (jnp.ndarray): Input array of shape (batch_size, seq_len, embedding_dim)

        Returns:
            jnp.ndarray: Output array of shape (batch_size, seq_len, embedding_dim)
        """
        hidden = x @ params['W1'] + params['B1']
        activated = FeedForward.gelu(hidden)
        output = activated @ params['W2'] + params['B2']

        if training and dropout > 0.0 and rng_key is not None:
            keep_prob = 1.0 - dropout
            dropout_mask = jax.random.bernoulli(rng_key, keep_prob, output.shape)
            output = jnp.where(dropout_mask, output / keep_prob, 0.0)

        return output

    @jax.jit
    def fwd_instance(self, x) -> jnp.ndarray:
        """
        Performs the forward pass of the feed-forward network.

        Args:
            x (jnp.ndarray): Input array of shape (batch_size, embedding_dim).

        Returns:
            jnp.ndarray: Output array of shape (batch_size, embedding_dim).
        """
        hidden = x @ self.W1 + self.B1
        activated = self.gelu(hidden)
        output = activated @ self.W2 + self.B2
        return output

    def compute_grads(self, x, target_ids) -> dict:
        """
        Computes gradients of the mean squared error loss w.r.t. the weights and biases.

        Args:
            x (jnp.ndarray): Input embeddings (batch_size, embedding_dim)
            target (jnp.ndarray): Target embeddings of same shape

        Returns:
            dict: Gradients for W1, B1, W2, B2
        """
        def loss_fn(W1, B1, W2, B2):
            hidden = x @ W1 + B1
            activated = self.gelu(hidden)
            logits = activated @ W2 + B2
            return self.loss_fn(logits, target_ids)

        grads = jax.grad(loss_fn, argnums=(0,1,2,3))(self.W1, self.B1, self.W2, self.B2)

        return{
            'dW1': grads[0],
            'dB1': grads[1],
            'dW2': grads[2],
            'dB2': grads[3]
        }

    def get_params_and_grads(self, grads) -> list[dict]:
        """
        Getting parameters and gradients for feedforward network

        Args:
            grads (dict): Dictionary containing gradients from FFN

        Returns:
            list:
                dict: Contains values and gradients of W1, B1, W2, and B2.
        """
        return [
            {'value': self.W1, 'grad': grads['dW1']},
            {'value': self.B1, 'grad': grads['dB1']},
            {'value': self.W2, 'grad': grads['dW2']},
            {'value': self.B2, 'grad': grads['dB2']},
        ]
