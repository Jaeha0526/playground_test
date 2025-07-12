"""Dummy identity wrapper for testing compatibility with wrap_for_brax_training."""

import jax
import jax.numpy as jp
from typing import Any
from brax.envs.base import Env, State


class DummyIdentityWrapper(Env):
    """Identity wrapper that just passes through to underlying environment.
    
    This is for testing if a Brax Env wrapper is compatible with
    wrap_for_brax_training which expects an MjxEnv.
    """
    
    def __init__(self, env):
        """Initialize wrapper.
        
        Args:
            env: Underlying MjxEnv environment
        """
        super().__init__()
        self._env = env
        
        # Pass through sizes
        self._action_size = env.action_size
        self._observation_size = env.observation_size
    
    @property
    def observation_size(self) -> Any:
        """Pass through observation size."""
        return self._observation_size
    
    @property
    def action_size(self) -> int:
        """Pass through action size."""
        return self._action_size
    
    @property
    def backend(self) -> str:
        """Pass through backend."""
        return getattr(self._env, 'backend', 'mjx')
    
    def reset(self, rng: jax.Array) -> State:
        """Pass through reset."""
        return self._env.reset(rng)
    
    def step(self, state: State, action: jax.Array) -> State:
        """Pass through step."""
        return self._env.step(state, action)
    
    # Pass through any other attributes
    def __getattr__(self, name: str) -> Any:
        """Forward other attributes to wrapped environment."""
        return getattr(self._env, name)
    
    # Important: Properties that might be needed by wrap_for_brax_training
    @property
    def dt(self) -> float:
        """Pass through dt if it exists."""
        return getattr(self._env, 'dt', 0.02)
    
    @property
    def mjx_model(self):
        """Pass through mjx_model if it exists."""
        return getattr(self._env, 'mjx_model', None)
    
    @property
    def mj_model(self):
        """Pass through mj_model if it exists."""
        return getattr(self._env, 'mj_model', None)