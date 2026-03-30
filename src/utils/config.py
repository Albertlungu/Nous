"""
./src/utils/config.py

Configuration utilities for JAX models.

Note: The previous jnp.array monkey-patch has been removed as it caused
instability by silently downcasting arrays that should remain float32
(loss values, gradient norms, learning rates). All weight initialization
is now explicitly cast to the desired dtype at creation time.
"""
