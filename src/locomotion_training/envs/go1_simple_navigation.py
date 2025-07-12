"""Simplified Go1 Navigation Environment that works with existing infrastructure."""

import jax
import jax.numpy as jp
from typing import Dict, Any, Optional, Tuple
from ml_collections import config_dict

from mujoco_playground._src.locomotion.go1 import joystick
from mujoco import mjx


def navigation_config() -> config_dict.ConfigDict:
    """Configuration for navigation environment."""
    # Start with joystick config
    config = joystick.default_config()
    
    # Modify for navigation
    config.episode_length = 500
    
    # Navigation-specific settings
    config.room_size = 10.0
    config.goal_radius = 0.5
    config.min_goal_distance = 2.0
    config.max_goal_distance = 8.0
    
    # Modify rewards for navigation
    config.reward_config.scales.tracking_lin_vel = 0.0  # Don't track joystick
    config.reward_config.scales.tracking_ang_vel = 0.0
    
    # Add navigation-specific reward scales
    config.reward_config.scales.goal_reached = 100.0
    config.reward_config.scales.goal_distance = 0.1
    
    # Disable random commands
    config.command_config.a = [0.0, 0.0, 0.0]
    config.command_config.b = [0.0, 0.0, 0.0]
    
    return config


class Go1SimpleNavigation(joystick.Joystick):
    """Simplified navigation environment that reuses existing Go1 setup."""
    
    def __init__(self, 
                 task: str = "flat_terrain",
                 config: Optional[config_dict.ConfigDict] = None,
                 config_overrides: Optional[dict] = None):
        
        if config is None:
            config = navigation_config()
            
        # Initialize parent normally
        super().__init__(task=task, config=config, config_overrides=config_overrides)
        
        # Navigation attributes
        self._room_size = config.room_size
        self._goal_radius = config.goal_radius
        self._min_goal_distance = config.min_goal_distance
        self._max_goal_distance = config.max_goal_distance
        
        # Goal position (will be set in reset)
        self._goal_position = jp.array([3.0, 3.0])
        
    def reset(self, rng: jax.Array) -> Any:
        """Reset with random goal."""
        # Call parent reset first
        state = super().reset(rng)
        
        # Use the RNG from state.info if available
        if 'rng' in state.info:
            rng = state.info['rng']
        
        # Split RNG for goal
        rng, rng_goal = jax.random.split(rng, 2)
        
        # Random goal position
        goal_position = jax.random.uniform(
            rng_goal, (2,), minval=-4.0, maxval=4.0
        )
        
        # Store goal in info
        info = state.info.copy()
        info['goal_position'] = goal_position
        info['rng'] = rng  # Store for future use
        
        # Update observations
        obs = self._get_obs(state.data, info)
        
        return state.replace(obs=obs, info=info)
    
    def _get_obs(self, data, info):
        """Add navigation observations."""
        # Get base observations
        base_obs = super()._get_obs(data, info)
        
        # Get robot position
        robot_pos = data.qpos[:2]
        robot_heading = data.qpos[5]
        
        # Get goal from info
        goal_pos = info.get('goal_position', jp.array([3.0, 3.0]))
        
        # Compute goal observations
        goal_vec = goal_pos - robot_pos
        goal_distance = jp.linalg.norm(goal_vec)
        goal_direction = goal_vec / (goal_distance + 1e-6)
        
        # Create navigation observations
        nav_obs = jp.concatenate([
            goal_direction,  # 2D
            jp.array([goal_distance]),  # 1D
            robot_pos,  # 2D
            jp.array([jp.sin(robot_heading), jp.cos(robot_heading)]),  # 2D
            jp.zeros(3),  # Padding to make 10D
        ])
        
        # Add to observations
        return {
            'state': base_obs['state'],
            'privileged_state': base_obs['privileged_state'],
            'navigation': nav_obs,
            'goal_distance': goal_distance,
        }
    
    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        """Navigation rewards."""
        # Get base rewards
        base_rewards = super()._get_reward(
            data, action, info, metrics, done, first_contact, contact
        )
        
        # Zero out velocity tracking
        base_rewards['tracking_lin_vel'] = 0.0
        base_rewards['tracking_ang_vel'] = 0.0
        
        # Get positions
        robot_pos = data.qpos[:2]
        goal_pos = info.get('goal_position', jp.array([3.0, 3.0]))
        
        # Goal distance
        goal_distance = jp.linalg.norm(goal_pos - robot_pos)
        
        # Goal reached reward
        goal_reached = goal_distance < self._goal_radius
        base_rewards['goal_reached'] = goal_reached.astype(jp.float32) * 100.0
        
        # Distance penalty
        base_rewards['goal_distance'] = -goal_distance * 0.1
        
        return base_rewards