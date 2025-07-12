"""Hierarchical navigation wrapper that integrates frozen locomotion policy.

This wrapper handles the conversion from 3D velocity commands to 12D joint actions
using a pre-loaded locomotion policy.
"""

import jax
import jax.numpy as jp
from typing import Any, Dict, Callable
from brax.envs.base import Env, State


class HierarchicalNavigationWrapper(Env):
    """Wrapper that enables hierarchical control for navigation.
    
    This wrapper:
    1. Accepts 3D velocity commands from navigation policy
    2. Converts them to 12D joint actions using frozen locomotion policy
    3. Maintains compatibility with Brax training infrastructure
    """
    
    def __init__(
        self,
        env: Env,
        locomotion_apply_fn: Callable,
        locomotion_params: Any,
        locomotion_obs_size: int = 48,
    ):
        """Initialize hierarchical wrapper.
        
        Args:
            env: Base navigation environment (expects 12D actions)
            locomotion_apply_fn: Locomotion network apply function
            locomotion_params: Pre-loaded locomotion parameters
            locomotion_obs_size: Size of locomotion observations
        """
        super().__init__()
        self._env = env
        self._locomotion_apply_fn = locomotion_apply_fn
        self._locomotion_params = locomotion_params  # Don't use device_put here
        self._locomotion_obs_size = locomotion_obs_size
        
        # Override action size to be velocity commands
        self._action_size = 3  # [vx, vy, vyaw]
        
        # Extract observation size from environment
        if hasattr(env, 'observation_size'):
            env_obs_size = env.observation_size
            if isinstance(env_obs_size, dict) and 'navigation' in env_obs_size:
                self._observation_size = env_obs_size['navigation'][0]
            else:
                self._observation_size = 10  # Default navigation obs size
        else:
            self._observation_size = 10
    
    @property
    def observation_size(self) -> int:
        """Navigation observation size."""
        return self._observation_size
    
    @property
    def action_size(self) -> int:
        """Velocity command size."""
        return self._action_size
    
    @property
    def backend(self) -> str:
        """Backend type."""
        return getattr(self._env, 'backend', 'mjx')
    
    def reset(self, rng: jax.Array) -> State:
        """Reset environment and return navigation observations."""
        # Ensure RNG is properly shaped
        if rng.ndim == 0:
            rng = jax.random.PRNGKey(0)  # Fallback
        
        # Reset underlying environment
        state = self._env.reset(rng)
        
        # Extract navigation observations
        if isinstance(state.obs, dict) and 'navigation' in state.obs:
            nav_obs = state.obs['navigation']
        else:
            # Fallback: create dummy navigation obs
            nav_obs = jp.zeros(self._observation_size)
        
        # Store full observation in state.info for later use
        info = state.info.copy()
        info['full_obs'] = state.obs
        
        # Return state with only navigation observations
        return state.replace(obs=nav_obs, info=info)
    
    def step(self, state: State, action: jax.Array) -> State:
        """Step environment using hierarchical policy.
        
        Args:
            state: Current state
            action: 3D velocity commands [vx, vy, vyaw]
            
        Returns:
            Next state with navigation observations
        """
        # Get full observations from state.info
        full_obs = state.info.get('full_obs', state.obs)
        
        # Create locomotion observations (48D)
        locomotion_obs = self._create_locomotion_obs(full_obs, action)
        
        # Get joint actions from frozen locomotion policy
        joint_actions = self._apply_locomotion_policy(locomotion_obs)
        
        # Step underlying environment with joint actions
        next_state = self._env.step(state, joint_actions)
        
        # Extract navigation observations from next state
        if isinstance(next_state.obs, dict) and 'navigation' in next_state.obs:
            nav_obs = next_state.obs['navigation']
            # Store full observation in info for next step
            info = next_state.info.copy()
            info['full_obs'] = next_state.obs
            next_state = next_state.replace(info=info)
        else:
            nav_obs = jp.zeros(self._observation_size)
        
        # Return state with navigation observations
        return next_state.replace(obs=nav_obs)
    
    def _create_locomotion_obs(self, full_obs: Any, velocity_commands: jax.Array) -> jax.Array:
        """Create locomotion observations from full state and velocity commands.
        
        Args:
            full_obs: Full observation dict from environment
            velocity_commands: 3D velocity commands [vx, vy, vyaw]
            
        Returns:
            48D locomotion observation vector
        """
        if isinstance(full_obs, dict) and 'state' in full_obs:
            # Extract base observation (45D without commands)
            base_obs = full_obs['state']
            
            # Original state is 48D with last 3 being commands
            # Replace last 3 elements with new velocity commands
            if base_obs.shape[-1] == 48:
                locomotion_obs = base_obs.at[..., -3:].set(velocity_commands)
            else:
                # Concatenate if size doesn't match
                locomotion_obs = jp.concatenate([base_obs[..., :-3], velocity_commands], axis=-1)
        else:
            # Fallback: create dummy observation
            locomotion_obs = jp.zeros(self._locomotion_obs_size)
            locomotion_obs = locomotion_obs.at[-3:].set(velocity_commands)
        
        return locomotion_obs
    
    def _apply_locomotion_policy(self, locomotion_obs: jax.Array) -> jax.Array:
        """Apply frozen locomotion policy to get joint actions.
        
        Args:
            locomotion_obs: 48D locomotion observations
            
        Returns:
            12D joint actions
        """
        # Apply locomotion network
        joint_actions = self._locomotion_apply_fn(
            self._locomotion_params,
            locomotion_obs
        )
        
        # Ensure correct shape
        if joint_actions.ndim == 1 and joint_actions.shape[0] == 12:
            return joint_actions
        elif joint_actions.ndim == 2 and joint_actions.shape[-1] == 12:
            return joint_actions.squeeze(0)
        else:
            # Fallback to zeros if something goes wrong
            return jp.zeros(12)
    
    # Pass through other required attributes
    def __getattr__(self, name: str) -> Any:
        """Forward other attributes to wrapped environment."""
        return getattr(self._env, name)