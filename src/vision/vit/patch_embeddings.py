"""
./src/vision/vit/patch_embeddings.py

Converts images into sequences of patch embeddings for Vision Transformer (ViT)
"""

import jax
import jax.numpy as jnp

from functools import partial

class PatchEmbedding:
    """
    Split images into patches and project these patches onto the embedding dimension
    """
    def __init__(
            self,
            image_size=224,
            patch_size=16,
            in_channels=3,
            embedding_dim=256
            ) -> None:
        """
        Initializing the PatchEmbedding class.

        Args:
            image_size (int, optional): Input image size, assuming square images.
                                        e.g. Image is 224x224. Defaults to 224.
            patch_size (int, optional): Size of each patch. Defaults to 16.
            in_channels (int, optional): Number of input channels (3 for RGB, 1 for grayscale).
                                         Defaults to 3.
            embedding_dim (int, optional): Dimension of patch embeddings.
                                           Must match text embedding_dim. Defaults to 256.
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        self.num_patches = image_size ** 2 // patch_size ** 2
            # Image size and patch size are squared for area of a square
        self.patch_dim = in_channels * patch_size ** 2

        key = jax.random.PRNGKey(
            68157628006304057045295846951897664502295431894160942124012093587298959185368
            )
        scale = 0.02
        self.projection = jax.random.normal(
            key,
            (self.patch_dim, self.embedding_dim)
            ) * scale
            # Creating a random matrix (patch_dim, embedding_dim)

        key, subkey = jax.random.split(key)
        self.cls_token = jax.random.normal(
            subkey,
            (1, 1, embedding_dim)
            ) * scale
            # The [CLS] token is used for global image representation TODO: WHAT THE HELL IS THIS

        key, subkey = jax.random.split(key)
        self.positional_embeddings = jax.random.normal(
            subkey,
            (1, self.num_patches + 1, embedding_dim) # +1 for CLS token
        ) * scale

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 3, 2, 4, 5))
    def fwd(
        params:dict,
        images:jnp.ndarray
        ) -> jnp.ndarray:
        """
        Forward pass: image -> patches -> embeddings

        Args:
            params (dict): Contains 'projection', 'cls_token', 'positional_embeddings',
                            'patch_size'.
            images (jnp.ndarray): A batch of images. Shape: (batch, height, width, channels)

        Returns:
            jnp.ndarray: Patch embeddings. Shape (batch, num_patches + 1, embedding_dim)
        """
        batch_size, h, w, channels = images.shape
        patch_size = params['patch_size']

        ph = h // patch_size # number of patches height
        pw = w // patch_size # number of patches width

        patches = images.reshape(
            batch_size,
            ph, patch_size,
            pw, patch_size,
            channels
        )

        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        patches = patches.reshape(batch_size, ph * pw, -1)

        patch_embeddings = patches @ params['projection']

        cls_tokens = jnp.tile(
            params['cls_token'],
            (batch_size, 1, 1))
        embeddings = jnp.concatenate(
            [cls_tokens, patch_embeddings],
            axis=1)

        embeddings += params['positional_embeddings']

        return embeddings


    def get_params(self) -> dict:
        """
        Get parameters for JAX functions
        """
        return {
            'projection': self.projection,
            'cls_token': self.cls_token,
            'positional_embeddings': self.positional_embeddings,
            'patch_size': self.patch_size
        }
