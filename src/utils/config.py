"""
./src/utils/config.py

Ensures all jnp array elements have type jnp.float16
"""

import jax.numpy as jnp # pylint: disable=no-member

_old_array = jnp.array
def safe_array(*args, **kwargs):
    """
    Monkey patches all arrays to be a certain float type (e.g., float16, float32, etc.)
    """
    if 'dtype' not in kwargs and args and isinstance(args[0], (list, tuple, jnp.ndarray)):
        first_elem = args[0][0] if len(args[0]) > 0 else None
        if isinstance(first_elem, (float, int)):
            kwargs['dtype'] = jnp.float16
    return _old_array(*args, **kwargs)

jnp.array = safe_array
