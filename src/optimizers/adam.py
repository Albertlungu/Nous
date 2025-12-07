import jax
import jax.numpy as jnp
from typing import Any, Dict, List, Tuple, Union
import numpy as np

class AdamNested:
    def __init__(self, lr=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8, warmup_steps=0, total_steps=None, schedule='constant', min_lr=0.0):
        """
        Adam optimizer with optional learning rate scheduling.

        Args:
            lr: Base learning rate
            beta1: First moment decay rate
            beta2: Second moment decay rate
            epsilon: Small constant for numerical stability
            warmup_steps: Number of steps for linear warmup (default: 0, no warmup)
            total_steps: Total training steps for cosine decay (default: None, no decay)
            schedule: 'constant', 'warmup', or 'warmup_cosine' (default: 'constant')
            min_lr: Minimum learning rate floor (default: 0.0)
        """
        self.base_lr = lr
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.state = {}

        # Learning rate schedule parameters
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.schedule = schedule
        self.min_lr = min_lr

        # Create JIT-compiled Adam step function
        self._jit_adam_step = jax.jit(self._adam_step_fn)

    @staticmethod
    @jax.jit
    def _adam_step_fn(params, grads, m, v, t, beta1, beta2, lr, epsilon):
        """Pure JAX function for Adam update (can be JIT compiled)."""
        # Update biased first moment estimate
        m_new = beta1 * m + (1 - beta1) * grads

        # Update biased second moment estimate
        v_new = beta2 * v + (1 - beta2) * (grads ** 2)

        # Compute bias-corrected first moment
        m_hat = m_new / (1 - beta1 ** t)

        # Compute bias-corrected second moment
        v_hat = v_new / (1 - beta2 ** t)

        # Update parameters
        updated_params = params - lr * m_hat / (jnp.sqrt(v_hat) + epsilon)

        return updated_params, m_new, v_new

    def get_lr(self):
        """Get current learning rate based on schedule and timestep."""
        if self.schedule == 'constant':
            return self.base_lr

        step = self.t

        if self.schedule == 'warmup':
            # Linear warmup only
            if step < self.warmup_steps:
                # Start from 1e-8 to avoid zero learning rate at step 0
                return self.base_lr * max(step / self.warmup_steps, 1e-8)
            else:
                return self.base_lr

        elif self.schedule == 'warmup_cosine':
            # Linear warmup + cosine decay
            if step < self.warmup_steps:
                # Linear warmup - start from small non-zero value
                return self.base_lr * max(step / self.warmup_steps, 1e-8)
            else:
                # Cosine decay after warmup
                if self.total_steps is None:
                    return self.base_lr

                progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                progress = min(progress, 1.0)  # Clamp to [0, 1]

                # Cosine annealing: starts at base_lr, ends at min_lr
                lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + np.cos(np.pi * progress))
                return max(lr, self.min_lr)

        return self.base_lr

    def _get_state_key(self, path):
        return str(path)

    def _step_single(self, params, grads, path=()):
        key = self._get_state_key(path)

        if key not in self.state:
            self.state[key] = {
                'm': jnp.zeros_like(params),
                'v': jnp.zeros_like(params)
            }

        m = self.state[key]['m']
        v = self.state[key]['v']

        # Use JIT-compiled function
        updated_params, m_new, v_new = self._jit_adam_step(
            params, grads, m, v, self.t,
            self.beta1, self.beta2, self.lr, self.epsilon
        )

        self.state[key]['m'] = m_new
        self.state[key]['v'] = v_new

        return updated_params

    def step(self, params, grads, path=()):
        if len(path) == 0:
            self.t += 1

        if isinstance(params, dict):
            return {
                key: self.step(params[key], grads[key], path + (key,))
                for key in params.keys()
            }

        elif isinstance(params, (list, tuple)):
            updated = [
                self.step(p, g, path + (i,))
                for i, (p,g) in enumerate(zip(params, grads))
            ]
            return type(params)(updated)

        else:
            return self._step_single(params, grads, path)
        