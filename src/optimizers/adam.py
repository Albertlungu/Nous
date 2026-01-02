"""
src/optimizers/adam.py

JAX implementation of Adam (Adaptive Moment Estimation) Nested optimizer for Nous.
"""

from typing import Any, Dict, List, Tuple, Union, Mapping

import jax
import jax.numpy as jnp
import numpy as np

GradientType = Union[jnp.ndarray, Mapping[str, Any], Tuple[Any, ...], List[Any]]

class AdamNested:
    def __init__(
            self,
            lr=1e-4,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            warmup_steps=0,
            total_steps=0,
            schedule='constant',
            min_lr=1e-7
            ) -> None:
        """
        Adam optimizer with optional learning rate scheduling.

        Args:
            lr (float, optional): Base learning rate. Defaults to 1e-4.
            beta1 (float, optional): First moment decay rate. Defaults to 0.9.
            beta2 (float, optional): Second moment decay rate. Defaults to 0.999.
            epsilon (float, optional): Small constant for numerical stability. Defaults to 1e-8.
            warmup_steps (int, optional): Number of steps for linear warmup. Defaults to 0.
            total_steps (int, optional): Total training steps for cosine decay. Defaults to 0.
            schedule (str, optional): 'constant', 'warmup', or 'warmup_cosine'.
                                       Defaults to 'constant'.
            min_lr (float): Minimum learning rate floor. Defaults to 1e-7.
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
    def _adam_step_fn(
        params:Mapping[str, Any],
        grads:GradientType,
        m:float,
        v:float,
        t:int,
        beta1:float,
        beta2:float,
        lr:float,
        epsilon:float
        ) -> tuple[Mapping[str, Any], float, float]:
        """
        Update function for Adam optimizer.

        Args:
            params (Mapping[str, Any]): Mapping containing optimizer parameters.
            grads (GradientType): The gradients for the parameters.
            m (float): First moment estimate.
            v (float): Second moment estimate.
            t (int): Time step (iteration counter).
            beta1 (float): First moment decay rate.
            beta2 (float): Second moment decay rate.
            lr (float): Base learning rate.
            epsilon (float): Small constant for numerical stability.

        Returns:
            tuple[Mapping[str, Any], float, float]: Tuple containing the updated parameters and the
                                                    new first and second moment estimates.
        """
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

    def get_lr(self) -> float:
        """
        Get current learning rate based on schedule and timestep.

        Returns:
            float: Learning rate.
        """
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

    def _get_state_key(
            self,
            path:Tuple[Union[str, int]]
            ) -> str:
        """
        Maps a parameter to an optimizer-state key.

        Args:
            path (Tuple[Union[str, int]]): Tuple that represents where the optimizer currently is in
                                           the parameter tree.

        Returns:
            str: A specific location in path as a string.
        """
        return str(path)

    def _step_single(
            self,
            params:Mapping[str, Any],
            grads:GradientType,
            path:Tuple[Union[str, int]]
            ) -> Mapping[str, Any]:
        """
        Make a single step through the Adam optimizer.

        Args:
            params (Mapping[str, Any]): Mapping with Adam parameters.
            grads (GradientType): The gradients for the parameters.
            path (Tuple[Union[str, int]]): Tuple that represents where the optimizer currently is in
                                           the parameter tree.

        Returns:
            Mapping[str, Any]: Updated parameters after a step through optimizer
        """
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

    def step(
            self,
            params:Mapping[str, Any],
            grads:GradientType,
            path:Tuple[Union[str, int]]
            ) -> Union[dict, list, tuple, Any]:
        """
        Applies the Adam optimizer to a nested parameter structure.

        Args:
            params (Mapping[str, Any]): Mapping with Adam parameters.
            grads (GradientType): The gradients for the parameters
            path (Tuple[Union[str, int]]): Tuple that represents where the optimizer currently is in
                                           the parameter tree.

        Returns:
            Union[dict, list, tuple, Any]: _description_
        """
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
