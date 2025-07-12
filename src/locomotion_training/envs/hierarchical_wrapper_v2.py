"""Improved wrapper that integrates hierarchical policy into the environment."""

import jax
import jax.numpy as jp
from typing import Any, Dict, Optional, Tuple
from brax.envs.base import Env, State
import flax.struct as struct


@struct.dataclass
class HierarchicalState:
    """Extended state that carries full observation for hierarchical policy."""
    base_state: State
    full_obs: Dict[str, jax.Array]
    
    # Proxy properties to maintain compatibility
    @property
    def pipeline_state(self):
        return self.base_state.pipeline_state
    
    @property
    def obs(self):
        return self.base_state.obs
    
    @property
    def reward(self):
        return self.base_state.reward
    
    @property
    def done(self):
        return self.base_state.done
    
    @property
    def metrics(self):
        return self.base_state.metrics
    
    @property
    def info(self):
        return self.base_state.info
    
    def replace(self, **kwargs):
        """Replace fields in base state."""
        # Handle special case for obs replacement
        if 'obs' in kwargs:
            new_base = self.base_state.replace(**kwargs)
            return HierarchicalState(base_state=new_base, full_obs=self.full_obs)
        else:
            new_base = self.base_state.replace(**kwargs)
            return HierarchicalState(base_state=new_base, full_obs=self.full_obs)


class HierarchicalPolicyWrapperV2(Env):
    """Improved wrapper that maintains state structure for JAX compatibility."""
    
    def __init__(
        self,
        env: Env,
        hierarchical_policy: Any,
    ):
        """Initialize wrapper.
        
        Args:
            env: Base navigation environment
            hierarchical_policy: HierarchicalNavigationPolicy instance
        """
        super().__init__()
        self._env = env
        self._hierarchical_policy = hierarchical_policy
        
        # Ensure locomotion policy is loaded
        self._hierarchical_policy.load_locomotion_policy()
        
        # Override action size to be velocity commands (3) instead of joints (12)
        self._action_size = 3  # [vx, vy, vyaw]
        
        # Override observation size to be navigation observations (10)
        self._observation_size = 10  # Flattened navigation observations
    
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
        base_state = self._env.reset(rng)
        
        # Extract navigation observations
        nav_obs = self._hierarchical_policy.extract_navigation_obs(base_state.obs)
        
        # Create hierarchical state with navigation observations
        state = HierarchicalState(
            base_state=base_state.replace(obs=nav_obs),
            full_obs=base_state.obs
        )
        
        return state
    
    def step(self, state: HierarchicalState, action: jax.Array) -> State:
        """Step environment using hierarchical policy.
        
        Args:
            state: Current hierarchical state
            action: Velocity commands [vx, vy, vyaw] from navigation network
            
        Returns:
            Next state with navigation observations
        """
        # Get full observation from hierarchical state
        full_obs = state.full_obs
        
        # Create locomotion observations with velocity commands
        locomotion_obs = self._hierarchical_policy.extract_locomotion_obs(
            full_obs, action
        )
        
        # Get joint actions from frozen locomotion policy
        joint_actions = self._hierarchical_policy.locomotion_forward(locomotion_obs)
        
        # Step the underlying environment with joint actions
        # Use the base state for stepping
        next_base_state = self._env.step(state.base_state, joint_actions)
        
        # Extract navigation observations from next state
        nav_obs = self._hierarchical_policy.extract_navigation_obs(next_base_state.obs)
        
        # Create new hierarchical state
        next_state = HierarchicalState(
            base_state=next_base_state.replace(obs=nav_obs),
            full_obs=next_base_state.obs
        )
        
        return next_state
    
    @property
    def dt(self) -> float:
        return self._env.dt
    
    def __getattr__(self, name: str) -> Any:
        """Forward other attributes to wrapped environment."""
        return getattr(self._env, name)