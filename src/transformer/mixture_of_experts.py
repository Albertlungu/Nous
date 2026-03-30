"""
./src/transformer/mixture_of_experts.py

Mixture of Experts implementation in JAX for Nous
"""

import os
import sys
from functools import partial

import jax # pylint: disable=no-member
import jax.numpy as jnp # pylint: disable=no-member

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class MOE:
    """
    Mixture of Experts implementation
    """

    def __init__(
            self,
            embedding_dim=256,
            ff_dim=None,
            num_experts=8,
            experts_per_token=2,
            num_blocks=2,
            scale=0.02,
            dropout=0.0,
            activation="gelu",
            dtype=None
            ):
        """
        Initialization of MoE layer

        Args:
            embedding_layer (EmbeddingLayer class): Embedding layer class
            ff_dim (_type_, optional): Feedforward network dimension. Defaults to None.
            num_experts (int, optional): Total number of experts available. Defaults to 16.
            experts_per_token (int, optional): Number of experts active per token. Defaults to 2.
            num_blocks (int, optional): Number of transformer blocks.. Defaults to 2.
            scale (float, optional): Uhhhhh. Defaults to 0.02.
            dropout (float, optional): Probability of dropout. Defaults to 0.0.
            activation (str, optional): Type of dropout. Defaults to "gelu".
            dtype (jnp.dtype, optional): Data type for weights. Defaults to jnp.bfloat16.
        """
        self.dtype = dtype if dtype is not None else jnp.bfloat16
        self.embedding_dim = embedding_dim
        self.ff_dim = ff_dim if ff_dim is not None else 4 * self.embedding_dim
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.num_blocks = num_blocks
        self.scale = scale
        self.dropout = dropout
        self.activation = activation

        self.router_W = (jax.random.normal(
            jax.random.PRNGKey(45),
            (self.embedding_dim, self.num_experts)
        ) * self.scale).astype(self.dtype)
        self.router_B = jnp.zeros(self.num_experts, dtype=self.dtype)

        # Create experts as stacked arrays
        k1_key, k2_key = jax.random.split(jax.random.PRNGKey(46))
        k1_keys = jax.random.split(k1_key, self.num_experts)
        k2_keys = jax.random.split(k2_key, self.num_experts)
        
        residual_scale = self.scale / jnp.sqrt(2.0 * self.num_blocks)
        
        self.experts_W1 = jax.vmap(lambda k: (jax.random.normal(k, (self.embedding_dim, self.ff_dim)) * self.scale).astype(self.dtype))(k1_keys)
        self.experts_B1 = jnp.zeros((self.num_experts, self.ff_dim), dtype=self.dtype)
        self.experts_W2 = jax.vmap(lambda k: (jax.random.normal(k, (self.ff_dim, self.embedding_dim)) * residual_scale).astype(self.dtype))(k2_keys)
        self.experts_B2 = jnp.zeros((self.num_experts, self.embedding_dim), dtype=self.dtype)


    def get_params(self):
        """
        Gets the parameters for the MoE block.

        Returns:
            dict:
                - router_W: Points to router weights
                - router_B: Points to router biases
                - experts_W1: Stacked array for W1
                - experts_B1: Stacked array for B1
                - experts_W2: Stacked array for W2
                - experts_B2: Stacked array for B2
        """
        return {
            'router_W': self.router_W,
            'router_B': self.router_B,
            'experts_W1': self.experts_W1,
            'experts_B1': self.experts_B1,
            'experts_W2': self.experts_W2,
            'experts_B2': self.experts_B2
        }

    def set_params(
            self,
            params:dict):
        """
        Load params from a dictionary.

        Args:
            params (dict): Parameter dictionary
        """
        self.router_W = params['router_W']
        self.router_B = params['router_B']

        self.experts_W1 = params['experts_W1']
        self.experts_B1 = params['experts_B1']
        self.experts_W2 = params['experts_W2']
        self.experts_B2 = params['experts_B2']



    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def expert_fwd(
        x:jnp.ndarray,
        expert_params:dict,
        activation='gelu'
        ):
        """
        Forward pass through a single expert (normal FFN)

        Args:
            x (jnp.ndarray): Input tensor (batch, seq_len, embedding_dim)
            expert_params (dict): Dictionary with W1, B1, W2, B2
            activation (str, optional): Activation, 'gelu' or 'relu'. Defaults to 'gelu'.

        Returns:
            jnp.ndarray: Output tensor (batch, seq_len, embedding_dim)
        """

        # First layer: transformations are applied to input tensor
        hidden = x @ expert_params['W1'] + expert_params['B1']

        # Activation
        if activation == 'gelu':
            activated = jax.nn.gelu(hidden)
        else:
            activated = jax.nn.relu(hidden)

        # Second layer: transformations are applied to activated layer
        output = activated @ expert_params['W2'] + expert_params['B2']

        return output

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
    def fwd(
        params:dict,
        x:jnp.ndarray,
        num_experts:int,
        experts_per_token:int,
        activation='gelu',
        training=False,
        dropout=0.0,
        key=None
        ):
        """
        Forward pass through full MoE layer.

        Args:
            params (dict): All MoE parameters (from get_params())
            x (jnp.ndarray): Input tensor (batch, seq_len, embedding_dim)
            num_experts (int): Number of experts
            experts_per_token (int): How many experts to activate per token.
            activation (str, optional): Activation function ('gelu' or 'relu'). Defaults to 'gelu'.
            training (bool, optional): Whether in training mode. Defaults to False.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            key (jax.random.PRNGkey, optional): JAX random key for dropout. Defaults to None.

        Returns:
            tuple
                - output (jnp.ndarray): MoE output (batch, seq_len, embedding_dim)
                - aux_loss (jnp.float16): Load balancing auxiliary loss
        """

        batch_size, seq_len, embedding_dim = x.shape

        #======== 1: Router ======== Calculate routing scores (decide which expert to use)
        router_logits = x @ params['router_W'] + params['router_B'] # Calculate score for each expert per token

        # Convert to probabilities
        router_probs = jax.nn.softmax(router_logits, axis=-1) # Shape: (batch, seq_len, num_experts)

        # ======= 2: Top-k selection (picking the selection) ========
        top_k_probs, top_k_indices = jax.lax.top_k(router_probs, experts_per_token)
        # top_k_probs: (batch, seq_len, experts_per_token)
        # top_k_indices: (batch, seq_len, experts_per_token)

        # Probability normalization
        top_k_probs = top_k_probs / jnp.sum(top_k_probs, axis=-1, keepdims=True)

        # ======= 3: Expert processing (run tokens through experts) ========
        def process_expert(i, w1, b1, w2, b2):
            expert_weights = jnp.where(
                top_k_indices == i,
                top_k_probs,
                0.0
            ).sum(axis=-1)

            # Mask to zero out tokens not using this expert (batch, seq_len, 1)
            expert_mask = (expert_weights > 0)[..., None]
            masked_input = jnp.where(expert_mask, x, 0.0)

            expert_params = {
                'W1': w1,
                'B1': b1,
                'W2': w2,
                'B2': b2
            }

            expert_out = MOE.expert_fwd(masked_input, expert_params, activation)
            return expert_out * expert_weights[..., None]

        # Vmap over the experts dimension to execute all experts in parallel
        all_expert_outputs = jax.vmap(process_expert)(
            jnp.arange(num_experts),
            params['experts_W1'],
            params['experts_B1'],
            params['experts_W2'],
            params['experts_B2']
        )
        
        # Sum outputs from all experts
        output = jnp.sum(all_expert_outputs, axis=0)

        # ======= 4: Dropout =======
        if training and dropout > 0.0:
            if key is None:
                key = jax.random.PRNGKey(47)
            keep_prob = 1.0 - dropout
            mask = jax.random.bernoulli(key, keep_prob, output.shape)
            output = jnp.where(mask, output / keep_prob, 0.0)

        # ======= 5: Load balancing loss ========
        # Make sure usage of experts is balanced to prevent "expert collapse"
            # Where all tokens go to the same 1-2 experts
        expert_usage = jnp.mean(router_probs, axis=(0, 1)) # Shape: (num_experts,)
        aux_loss = num_experts * jnp.sum(expert_usage**2) # Weigh the experts already used more

        return output, aux_loss # Type "tuple[Array | Unknown, Array]" is not assignable to return type "tuple[ndarray, float]" "Array" is not assignable to "float"

    def fwd_instance(
            self,
            x:jnp.ndarray,
            training=False,
            key=None
            ):
        """
        Instance method fwd pass (calls the static fwd() method)
        Args:
            x (jnp.ndarray): Input tensor (batch, seq_len, embedding_dim)
            expert_params (dict): Dictionary with W1, B1, W2, B2
            activation (str, optional): Activation, 'gelu' or 'relu'. Defaults to 'gelu'.

        Returns:
            tuple:
                - output (jnp.ndarray): MoE output (batch, seq_len, embedding_dim)
                - aux_loss (jnp.float16): Load balancing auxiliary loss
        """
        params = self.get_params()
        return self.fwd(
            params,
            x,
            self.num_experts,
            self.experts_per_token,
            activation=self.activation,
            training=training,
            dropout=self.dropout,
            key=key
        )

    def count_params(self):
        """
        Count total number of params in MoE layer

        Returns:
            int: Total parameter count
        """
        router_params = self.embedding_dim * self.num_experts + self.num_experts

        expert_params = self.num_experts * (
            self.embedding_dim * self.ff_dim + self.ff_dim + # W1, B1
            self.ff_dim * self.embedding_dim + self.embedding_dim) # W2, B2

        return router_params + expert_params


def main():
    # moe = MOE()
    # print(moe.get_params())
    pass

if __name__ == "__main__":
    main()
