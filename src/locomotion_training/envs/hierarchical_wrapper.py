"""Wrapper that integrates hierarchical policy into the environment."""

import jax
import jax.numpy as jp
from typing import Any, Dict, Optional
from brax.envs.base import Env, State


class HierarchicalPolicyWrapper(Env):
    """Wraps navigation environment to integrate frozen locomotion policy.
    
    This wrapper allows us to train only the navigation network (3 outputs)
    while the environment internally uses the frozen locomotion policy
    to convert velocity commands to joint actions.
    """
    
    def __init__(
        self,
        env: Env,
        hierarchical_policy: Any,
        flatten_obs: bool = True
    ):
        """Initialize wrapper.
        
        Args:
            env: Base navigation environment
            hierarchical_policy: HierarchicalNavigationPolicy instance
            flatten_obs: Whether to flatten observations for PPO
        """
        super().__init__()
        self._env = env
        self._hierarchical_policy = hierarchical_policy
        self._flatten_obs = flatten_obs
        
        # Ensure locomotion policy is loaded
        self._hierarchical_policy.load_locomotion_policy()
        
        # Override action size to be velocity commands (3) instead of joints (12)
        self._action_size = 3  # [vx, vy, vyaw]
        
        # Override observation size to be navigation observations (10)
        if flatten_obs:
            self._observation_size = 10  # Flattened navigation observations
        else:
            self._observation_size = {'navigation': (10,)}
    
    @property
    def observation_size(self) -> Any:
        return self._observation_size
    
    @property
    def action_size(self) -> int:
        return self._action_size
    
    @property
    def backend(self) -> str:
        return self._env.backend
    
    def reset(self, rng: jax.Array) -> State:
        """Reset environment and return navigation observations."""
        state = self._env.reset(rng)
        
        # Extract navigation observations
        nav_obs = self._hierarchical_policy.extract_navigation_obs(state.obs)
        
        # Create new state with navigation observations
        if self._flatten_obs:
            obs = nav_obs
        else:
            obs = {'navigation': nav_obs}
        
        # Store full observation in a wrapper-specific attribute
        self._full_obs = state.obs
        
        return state.replace(obs=obs)
    
    def step(self, state: State, action: jax.Array) -> State:
        """Step environment using hierarchical policy.
        
        Args:
            state: Current state
            action: Velocity commands [vx, vy, vyaw] from navigation network
            
        Returns:
            Next state with navigation observations
        """
        # Use stored full observation
        full_obs = self._full_obs if hasattr(self, '_full_obs') else state.obs
        
        # Create locomotion observations with velocity commands
        locomotion_obs = self._hierarchical_policy.extract_locomotion_obs(
            full_obs, action
        )
        
        # Get joint actions from frozen locomotion policy
        joint_actions = self._hierarchical_policy.locomotion_forward(locomotion_obs)
        
        # Step the underlying environment with joint actions
        next_state = self._env.step(state, joint_actions)
        
        # Extract navigation observations from next state
        nav_obs = self._hierarchical_policy.extract_navigation_obs(next_state.obs)
        
        # Create new state with navigation observations
        if self._flatten_obs:
            obs = nav_obs
        else:
            obs = {'navigation': nav_obs}
        
        # Store full observation for next step
        self._full_obs = next_state.obs
        
        return next_state.replace(obs=obs)
    
    @property
    def dt(self) -> float:
        return self._env.dt
    
    def __getattr__(self, name: str) -> Any:
        """Forward other attributes to wrapped environment."""
        return getattr(self._env, name)