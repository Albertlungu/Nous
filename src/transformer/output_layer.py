"""
src/transformer/output_layer.py

Defines the OutputLayer class for a transformer model.

The OutputLayer maps the final hidden states from the transformer blocks to logits. It gives forward
and backward passes, as well as token prediction methods.

Classes:
    OutputLayer:
        The transformer output layer, containing:
            - Weight and bias parameters
            - Forward pass computation
            - Gradient computation for backprop
            - Parameter access
            - Prediction of next token
"""

import os
import sys

import jax # pylint: disable=no-member
import jax.numpy as jnp # pylint: disable=no-member

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.embeddings.embeddings import EmbeddingLayer


class OutputLayer:
    """
    Output Layer of the transformer

    Attributes:
        embedding_layer (object): Instance of the EmbeddingLayer class
        embedding_dim (int): Embedding dimension
        vocab_size (int): Vocabulary size from tokenizer
        W_out (jnp.array[int, int]): Weight matrix, based on embedding_dim and vocab_size
        b_out (jnp.array[int]): Bias vector, shape: (vocab_size)
    """
    def __init__(self, embedding_layer: EmbeddingLayer) -> None:
        """
        Initializes instance attributes for OutputLayer class.

        Args:
            embedding_layer (EmbeddingLayer): EmbeddingLayer class.
        """
        self.embedding_layer = embedding_layer
        self.embedding_dim = self.embedding_layer.embedding_dim
        self.vocab_size = embedding_layer.vocab_size

        self.W_out = embedding_layer.embeddings.T # Weight matrix shape: (embedding_dim, vocab_size)
        # self.W_out = np.random.randn(self.embedding_dim, self.vocab_size) * 0.01
        self.b_out = jnp.zeros(self.vocab_size) # Bias vector

    @staticmethod
    def fwd(params:dict,
            transformer_output:jnp.ndarray
            ) -> jnp.ndarray:
        """

        Args:
            params (dict)
            transformer_output (jnp.ndarray): Output from last transformer block
                Shape: (batch_size, seq_len, embedding_dim)

        Returns:
            jnp.ndarray: Logits over vocabulary,
                Shape: (batch_size, seq_len, vocab_size)
        """
        logits = transformer_output @ params['W_out']  + params['b_out']
        # print(np.shape(self.logits))
        return logits

    def get_params(self) -> dict:
        """
        Gets output layer parameters.

        Returns:
            dict: Dictionary containing output layer parameters (W_out and b_out)
        """
        return {
            'W_out': self.W_out,
            'b_out': self.b_out
        }

    def compute_grads(self,
                    transformer_output:jnp.ndarray,
                    d_output:jnp.ndarray
                    ) -> tuple[dict, jnp.ndarray]:
        """
        Backward function

        Args:
            transformer_output (jnp.array[int, int, int]): Output from last transformer block.
                Shape: [batch_size, seq_len, embedding_dim]
            d_output (jnp.ndarray[int, int, int]): Gradient of loss w.r.t. output layer logits.
                Shape: [batch_size, seq_len, embedding_dim]

        Returns:
            tuple:
                - grads_params (dict[str, jnp.ndarray]): Gradients w.r.t. output layer parameters
                - d_input (jnp.ndarray): Gradient w.r.t. transformer_output, same shape as input
        """
        params = self.get_params()
        output, vjp_fn = jax.vjp(
            lambda p, x: self.fwd(p, x),
            params,
            transformer_output
        )
        grads_params, d_input = vjp_fn(d_output)
        return grads_params, d_input

    def get_params_and_grads(self) -> list[dict]:
        """
        Returns parameters and gradients of OutputLayer

        Returns:
            list:
                dict:
                    - W_out (jnp.array): Updated weight matrix
                    - b_out (jnp.array): Updated bias vector
        """
        if grads is None:
            grads = {
                  'W_out': jnp.zeros_like(self.W_out),
                  'b_out': jnp.zeros_like(self.b_out)
              }

        return [
            {'value': self.W_out, 'grad': grads['W_out']},
            {'value': self.b_out, 'grad': grads['b_out']}
        ]

    def predict_next_token(self,
                           transformer_output:jnp.ndarray,
                           temperature=1.0
                           ) -> jnp.ndarray:
        """
        Samples the next token from the model's predictions.

        Args:
            transformer_output (jnp.ndarray): Output from last TransformerBlock
                                            shape: (batch_size, seq_len, embedding_dim)
            temperature (float16): Sampling temperature (default 1.0)
                                Higher = more random, Lower = more deterministic

        Returns:
            jnp.ndarray: Predicted token IDs
                        shape: (batch_size,) - one prediction per sequence
        """
        params = self.get_params()
        logits = self.fwd(params, transformer_output)[:, -1, :]
        scaled_logits = logits / temperature
        probs = jax.nn.softmax(scaled_logits)
        predicted_tokens = jnp.argmax(probs, axis = -1)
        return predicted_tokens
