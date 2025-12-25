"""
src/transformer/mixture_of_experts.py

Mixture of Experts implementation in JAX for Nous
"""

import os
import sys

import jax # pylint: disable=no-member
import jax.numpy as jnp # pylint: disable=no-member

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class MOE:
    """
    Mixture of Experts implementation
    """

    def __init__(self,
                embedding_dim=256,
                ff_dim=None,
                num_experts=16,
                experts_per_token=2,
                num_blocks=2,
                scale=0.02,
                dropout=0.0,
                activation="gelu"
                ) -> None:
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
        """

        self.embedding_dim = embedding_dim
        self.ff_dim = ff_dim if ff_dim is not None else 4 * self.embedding_dim
        self.num_experts = num_experts
        self.ept = experts_per_token
        self.num_blocks = num_blocks
        self.scale = scale
        self.dropout = dropout
        self.activation = activation

        # Router: decides which experts to use
        self.router_W = jax.random.normal(
            jax.random.PRNGKey(67),
            (self.embedding_dim, self.num_experts)
        ) * self.scale
        self.router_B = jnp.zeros(self.num_experts)

        # Create experts
        key = jax.random.PRNGKey(0)
        self.experts = []
        for i in range(self.num_experts):
            key, subkey = jax.random.split(key)
            expert = self._init_expert(subkey)
            self.experts.append(expert)


    def _init_expert(self, key:int) -> dict:
        """
        Create a single expert. Each expert is an entire FFN.

        Args:
            key (int): PRNG key used

        Returns:
            dict: Contains W1, B1, W2, and B2 (weights and biases for both base and residual layers)
        """
        k1, k2 = jax.random.split(key)

        residual_scale = self.scale / jnp.sqrt(2.0 * self.num_blocks)

        return {
            'W1': jax.random.normal(k1, (self.embedding_dim, self.ff_dim)) * self.scale,
            'B1': jnp.zeros(self.ff_dim),
            'W2': jax.random.normal(k2, (self.ff_dim, self.embedding_dim)) * residual_scale,
            'B2': jnp.zeros(self.embedding_dim)
        }

    def get_params(self) -> dict:
        """
        Gets the parameters for each expert and returns as a dictionary

        Returns:
            dict:
                - router_W: Points to router weights
                - router_B: Points to router biases
                - expert_{i}_X: Points to W1, B1, W2, and B2
                    (weights and biases for both base and residual layers) for all experts
        """
        params = {
            'router_W': self.router_W,
            'router_B': self.router_B
        }

        for i, expert in enumerate(self.experts):
            params[f'expert_{i}_W1'] = expert['W1']
            params[f'expert_{i}_B1'] = expert['B1']
            params[f'expert_{i}_W2'] = expert['W2']
            params[f'expert_{i}_B2'] = expert['B2']

        return params

    def set_params(self, params) -> None:
        """
        Load params from a dictionary.

        Args:
            params (dict): Parameter dictionary
        """
        self.router_W = params['router_W']
        self.router_B = params['router_B']

        for i in range(self.num_experts):
            self.experts[i] = {
                'W1': params[f'expert_{i}_W1'],
                'B1': params[f'expert_{i}_B1'],
                'W2': params[f'expert_{i}_W2'],
                'B2': params[f'expert_{i}_B2']
            }

    @staticmethod
    def gelu(x) -> jnp.ndarray:
        """
        GELU activation function

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

    @jax.jit
    @staticmethod
    def expert_fwd(x:jnp.ndarray,
                   expert_params:dict,
                   activation='gelu'
                   ) -> jnp.ndarray:
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
            activated = MOE.gelu(hidden)
        else:
            activated = MOE.relu(hidden)

        # Second layer: transformations are applied to activated layer
        output = activated @ expert_params['W2'] + expert_params['B2']

        return output

    @jax.jit
    @staticmethod
    def fwd(params:dict,
            x:jnp.ndarray,
            num_experts:int,
            experts_per_token:int,
            activation='gelu',
            training=False,
            dropout=0.0,
            key=None
            ) -> tuple[jnp.ndarray, float]:
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
        output = jnp.zeros_like(x) # Init output

        # Process each expert
        for i in range(num_experts):
            # Get params
            expert_params = {
                'W1': params[f'expert_{i}_W1'],
                'B1': params[f'expert_{i}_B1'],
                'W2': params[f'expert_{i}_W2'],
                'B2': params[f'expert_{i}_B2']
            }
            # Mask finds which tokens selected this expert (batch, seq_len)
            expert_mask = jnp.any(top_k_indices == i, axis=-1) # True where expert is in top_k

            # Get routing weights for this expert
            expert_weights = jnp.where(
                top_k_indices == i,
                top_k_probs,
                0.0
            ).sum(axis=-1) # shape: (batch_seq_len)

            # Run expert on all tokens
            expert_out = MOE.expert_fwd(x, expert_params, activation)

            # Make a weighted output multiplied by the routing weights
            weighted_out = expert_out * expert_weights[..., None] # ... adds a dimension (batch, seq_len, 1)

            output = output + weighted_out

        # ======= 4: Dropout =======
        if training and dropout > 0.0:
            if key is None:
                key = jax.random.PRNGKey(0)
            keep_prob = 1.0 - dropout
            mask = jax.random.bernoulli(key, keep_prob, output.shape)
            output = jnp.where(mask, output / keep_prob, 0.0)

        # ======= 5: Load balancing loss ========
        # Make sure usage of experts is balanced to prevent "expert collapse"
            # Where all tokens go to the same 1-2 experts
        expert_usage = jnp.mean(router_probs, axis=(0, 1)) # Shape: (num_experts,)
        aux_loss = num_experts * jnp.sum(expert_usage**2) # Weigh the experts already used more

        return output, aux_loss

    def fwd_instance(self,
                    x:jnp.ndarray,
                    training=False,
                    key=None
                    ) -> tuple[jnp.ndarray, float]:
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

    def count_params(self) -> int:
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


def main() -> None:
    # moe = MOE()
    # print(moe.get_params())
    pass

if __name__ == "__main__":
    main()