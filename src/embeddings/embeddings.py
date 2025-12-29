"""
Token embedding layer with positional encoding for transformer models.

This module implements the embedding layer that converts token IDs to dense vectors
and adds positional information. The embedding dimensions are typically powers of 2
(128, 256, 512, 1024).
"""
import os
import pickle
import sys
import xml.etree.ElementTree as ET

import jax.numpy as jnp # pylint: disable=no-member
import jax # pylint: disable=no-member
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.tokenizer.tokenizer_class import BPETokenizer
from src.embeddings.positional_encoding import PositionalEncoding
import src.utils.config

class EmbeddingLayer:
    """
    Token embedding layer with positional encoding for transformer models.

    Attributes:
        vocab_size (int): Vocabulary size.
        embedding_dim (int): Embedding dimension.
        max_seq_length (int): Maximum sequence length.
        n (int): Positional encoding frequency parameter.
        dropout (float): Dropout probability.
        key (jax.random.PRNGKey): PRNG key from JAX's random module.
        embeddings (jnp.ndarray): Embedding matrix of shape (vocab_size, embedding_dim).
        positional_encoding_class (PositionalEncoding): Instance of PositionalEncoding class.
        positional_encodings (jnp.ndarray): Positional encodings of shape
            (max_seq_length, embedding_dim).
    """

    default_embedding_dim = 256

    def __init__(self,
                 vocab_size=None,
                 embedding_dim=None,
                 max_seq_length=256,
                 n=10000,
                 dropout=0.0
                 ) -> None:
        """
        Initializes an EmbeddingLayer object.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int, optional): Embedding dimension. Defaults to 256.
            max_seq_length (int, optional): Maximum sequence length. Defaults to 256.
            n (int, optional): Positional encoding frequency parameter. Defaults to 10000.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim or self.default_embedding_dim
        self.max_seq_length = max_seq_length
        self.n = n
        self.dropout = dropout

        # Create key inside __init__, not at class level
        self.key = jax.random.PRNGKey(
            68157628006304057045295846951897664502295431894160942124012093587298959185368
            )
        self.embeddings = jax.random.normal(self.key, (self.vocab_size, self.embedding_dim)) * 0.02
            # Basically, random numbers are selected for the vectors right now as placeholder
            # so that the algorithm doesn't see symmetry and simply assign the same
            # vector values to every word upon training

        self.positional_encoding_class = PositionalEncoding(self.embedding_dim, self.max_seq_length)
        self.positional_encodings = self.positional_encoding_class._create_positional_encoding(n)
            # using the function that will be declared later to get the positional encoding of a certain word

    @staticmethod
    def pad_token_ids(max_len,
                      token_ids,
                      pad_token_id=None):
        """
        Pads token ids

        Args:
            max_len (int): Maximum length of embeddings
            token_ids (list): List of token ids to pad
            pad_token_id (int, optional): Pad token id. Defaults to None.

        Returns:
            list: Padded token IDs
        """
        batch_size = len(token_ids)
        padded = jnp.full((batch_size, max_len), pad_token_id, dtype=jnp.int32)
        lengths = jnp.array([min(len(seq), max_len) for seq in token_ids])

        def scatter_seq(i,seq):
            l = lengths[i]
            return padded.at[i, :l].set(jnp.array(seq[:l], dtype=jnp.int32))

        padded = jax.lax.fori_loop(0, batch_size, scatter_seq, padded)
        return padded


    @staticmethod
    def embedding_fwd(params,
                      padded_token_ids,
                      pad_token_id=None,
                      dropout=0.0,
                      training=True,
                      rng_key=None):
        """
        Forward method of the embedding layer

        Args:
            params (dict): Dictionary containing parameters
            padded_token_ids (list): List with padded token ids
            pad_token_id (int, optional): Pad token id. Defaults to None.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            training (bool, optional): Is training? Defaults to True.
            rng_key (int, optional): Seed for random number generation. Defaults to None.

        Returns:
            tuple(jnp.array, list): Tuple containing the output embeddings and the mask
        """
        # Unpack params - params is now a dict for consistency with other layers
        embeddings = params['embeddings']
        positional_encodings = params['positional_encodings']

        # Ensure token_ids are integers for indexing
        padded_token_ids = jnp.asarray(padded_token_ids, dtype=jnp.int32)

        mask = padded_token_ids != pad_token_id

        # Use one-hot encoding approach - fully differentiable and JIT compatible
        # This is the correct way to do embedding lookup in JAX with autodiff
        batch_size, seq_len = padded_token_ids.shape
        vocab_size, embedding_dim = embeddings.shape

        # Create one-hot encoding of token IDs
        # This converts integer indices to a differentiable operation
        token_ids_one_hot = jax.nn.one_hot(padded_token_ids, vocab_size)  # (batch, seq, vocab)

        # Matrix multiply to get embeddings: (batch, seq, vocab) @ (vocab, embed_dim) = (batch, seq, embed_dim)
        token_embeddings = jnp.einsum('bsv,ve->bse', token_ids_one_hot, embeddings)
        token_embeddings = token_embeddings

        seq_len = padded_token_ids.shape[1]

        # Use positional encodings up to seq_len
        # If seq_len > available encodings, we use modulo to wrap around
        # This is JIT-compatible since we avoid dynamic padding
        pos_indices = jnp.arange(seq_len) % positional_encodings.shape[0]
        pos_enc = positional_encodings[pos_indices, :]

        output = token_embeddings + pos_enc[None, :, :]

        if training and dropout > 0.0 and rng_key is not None:
            keep_prob = 1.0 - dropout
            dropout_mask = jax.random.bernoulli(rng_key, keep_prob, output.shape)
            output = jnp.where(dropout_mask, output / keep_prob, 0.0)

        return output, mask

    # Removed loss_fn method - loss computation is now handled in Trainer


    def update(self,
               grads:jnp.ndarray,
               learning_rate:float
               ) -> None:
        """
        Updates embedding weights using gradients (added up from all ids)

        Args:
            grads (jnp.ndarray): gradients from the backward pass
            learning_rate (float): rate at which the machine moves forward
        """
        embedding_grads, pos_enc_grads = grads
        self.embeddings -= learning_rate * embedding_grads
        self.positional_encodings -= learning_rate * pos_enc_grads

    def save(self,
             filepath:str
             ) -> None:
        """
        Save embeddings to a file
        Args:
            filepath (str): file from which embeddings are loaded
        """

        with open(filepath, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'max_seq_length': self.max_seq_length
            }, f)

    def load(self,
             filepath:str
             ) -> None:
        """
        Load embeddings from file
        Args:
            filepath (str): file from which embeddings are loaded
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.embeddings = data['embeddings']
            self.vocab_size = data['vocab_size']
            self.embedding_dim = data['embedding_dim']
            self.max_seq_length = data['max_seq_length']
            self.positional_encodings = self.positional_encoding_class._create_positional_encoding()

    def get_params(self) -> dict:
        """
        Get parameters as a dict for JAX functions

        Returns:
            dict:
                - embeddings (jnp.ndarray)
                - positional_encodings (jnp.ndarray)
        """
        return {
            'embeddings': self.embeddings,
            'positional_encodings': self.positional_encodings
        }

    def get_params_and_grads(self) -> tuple[list[int, int], jnp.ndarray]:
        """
        Gets parameters and gradients from embedding class

        Returns:
            tuple(list[int, int], jnp.ndarray): Tuple containing embeddings and positional encodings
        """
        return (self.embeddings, self.positional_encodings)
