"""Simplified Go1 Navigation Environment V2 - Fixed RNG handling."""

import jax
import jax.numpy as jp
from typing import Dict, Any, Optional, Tuple
from ml_collections import config_dict

from mujoco_playground._src.locomotion.go1 import joystick
from mujoco_playground._src import mjx_env
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
    
    # Disable random commands
    config.command_config.a = [0.0, 0.0, 0.0]
    config.command_config.b = [0.0, 0.0, 0.0]
    
    return config


class Go1SimpleNavigationV2(joystick.Joystick):
    """Simplified navigation environment with proper RNG handling."""
    
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
        
    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset with random goal - fixed RNG handling."""
        # Split RNG first, before passing to parent
        rng, rng_goal = jax.random.split(rng, 2)
        
        # Call parent reset with its own RNG
        state = super().reset(rng)
        
        # Generate random goal position
        goal_position = jax.random.uniform(
            rng_goal, (2,), minval=-4.0, maxval=4.0
        )
        
        # Store goal in info
        info = {
            'goal_position': goal_position,
            'episode_steps': 0,
            'time': 0.0,
            # Keep existing info
            **state.info
        }
        
        # Update observations with navigation info
        obs = self._get_obs(state.data, info)
        
        return state.replace(obs=obs, info=info)
    
    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, jax.Array]:
        """Get observations including navigation info."""
        # Get base observations
        base_obs = super()._get_obs(data, info)
        
        # Get robot state
        robot_pos = data.qpos[:2]
        robot_heading = data.qpos[5]
        
        # Get goal position (with default fallback)
        goal_pos = info.get('goal_position', jp.array([3.0, 3.0]))
        
        # Compute goal observations
        goal_vec = goal_pos - robot_pos
        goal_distance = jp.linalg.norm(goal_vec)
        goal_direction = goal_vec / (goal_distance + 1e-6)
        
        # Transform to robot frame
        cos_h = jp.cos(robot_heading)
        sin_h = jp.sin(robot_heading)
        goal_direction_local = jp.array([
            goal_direction[0] * cos_h + goal_direction[1] * sin_h,
            -goal_direction[0] * sin_h + goal_direction[1] * cos_h
        ])
        
        # Create navigation observations (10D)
        nav_obs = jp.concatenate([
            goal_direction_local,  # 2D
            jp.array([goal_distance]),  # 1D
            robot_pos,  # 2D
            jp.array([jp.sin(robot_heading), jp.cos(robot_heading)]),  # 2D
            jp.zeros(3),  # 3D padding
        ])
        
        # Return observations dict
        return {
            'state': base_obs['state'],
            'privileged_state': base_obs['privileged_state'],
            'navigation': nav_obs,
            'goal_distance': goal_distance,
        }
    
    def _get_reward(self, data: mjx.Data, action: jax.Array, info: Dict[str, Any],
                    first_contact: jax.Array, done: jax.Array, contact: jax.Array, 
                    qvel_history: jax.Array, metrics: Dict[str, Any]) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Navigation rewards."""
        # Get base rewards
        reward, base_rewards = super()._get_reward(
            data, action, info, first_contact, done, contact, qvel_history, metrics
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
        
        # Progress reward (if previous position available)
        if 'prev_robot_pos' in info:
            prev_pos = info['prev_robot_pos']
            prev_distance = jp.linalg.norm(goal_pos - prev_pos)
            progress = prev_distance - goal_distance
            base_rewards['goal_progress'] = progress * 5.0
        else:
            base_rewards['goal_progress'] = 0.0
        
        # Compute total reward
        total_reward = sum(
            base_rewards[k] * self._config.reward_config.scales.get(k, 1.0)
            for k in base_rewards
        )
        
        return total_reward, base_rewards
    
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step environment with navigation logic."""
        # Store previous position in info
        info = state.info.copy()
        info['prev_robot_pos'] = state.data.qpos[:2]
        
        # Update state with new info before stepping
        state = state.replace(info=info)
        
        # Step parent environment
        next_state = super().step(state, action)
        
        # Update observations with navigation info
        obs = self._get_obs(next_state.data, next_state.info)
        
        # Check if goal reached for early termination
        goal_distance = obs['goal_distance']
        goal_reached = goal_distance < self._goal_radius
        
        # Update done flag
        done = jp.logical_or(next_state.done, goal_reached)
        
        return next_state.replace(obs=obs, done=done)