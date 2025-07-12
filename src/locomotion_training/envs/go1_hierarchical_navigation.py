"""Go1 Hierarchical Navigation Environment.

This environment implements the 4 main changes:
1. Observations include goal-related information
2. Actions are 3D velocity commands (handled by wrapper)
3. Room with random goal positioning
4. Goal-based reward system
"""

import jax
import jax.numpy as jp
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from ml_collections import config_dict
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import joystick
from mujoco_playground._src import mjx_env
from mujoco import mjx
import mujoco


def hierarchical_navigation_config() -> config_dict.ConfigDict:
    """Configuration for hierarchical navigation environment."""
    # Start with joystick config as base
    base_config = joystick.default_config()
    
    # Modify for navigation task
    config = config_dict.create(
        # Base settings
        ctrl_dt=base_config.ctrl_dt,
        sim_dt=base_config.sim_dt,
        episode_length=500,  # Shorter episodes for navigation
        
        # PD controller (same as base)
        Kp=base_config.Kp,
        Kd=base_config.Kd,
        action_repeat=base_config.action_repeat,
        action_scale=base_config.action_scale,
        soft_joint_pos_limit_factor=base_config.soft_joint_pos_limit_factor,
        
        # Room and goal settings
        room_size=10.0,
        wall_buffer=1.0,  # Min distance from walls for spawn/goal
        goal_radius=0.5,  # Success threshold
        min_goal_distance=2.0,  # Min spawn-goal separation
        max_goal_distance=8.0,  # Max goal distance
        
        # Reduced noise for cleaner navigation
        noise_config=config_dict.create(
            level=0.5,  # Reduced from 1.0
            scales=base_config.noise_config.scales,
        ),
        
        # Navigation-focused rewards
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Goal-based rewards (new)
                goal_reached=100.0,      # Sparse success reward
                goal_progress=2.0,       # Dense progress reward
                goal_distance=-0.1,      # Distance penalty
                heading_alignment=0.5,   # Face toward goal
                
                # Wall safety (new)
                wall_collision=-10.0,    # Wall collision penalty
                boundary_penalty=-1.0,   # Near-wall penalty
                
                # Reduced locomotion rewards (from base)
                orientation=-2.0,        # Stay upright (reduced)
                termination=-1.0,        # Don't fall
                lin_vel_z=-0.2,         # Reduced vertical motion penalty
                ang_vel_xy=-0.02,       # Reduced roll/pitch penalty
                
                # Removed tracking rewards (no joystick commands)
                tracking_lin_vel=0.0,
                tracking_ang_vel=0.0,
                
                # Other base rewards zeroed out
                pose=0.0,
                feet_air_time=0.0,
                dof_pos_limits=0.0,
                torques=0.0,
                action_rate=0.0,
                stand_still=0.0,
                energy=0.0,
                feet_clearance=0.0,
                feet_height=0.0,
                feet_slip=0.0,
            ),
            tracking_sigma=0.25,
            max_foot_height=0.1,
        ),
        
        # Disable random commands (we use goal-directed navigation)
        command_config=config_dict.create(
            a=[0.0, 0.0, 0.0],
            b=[0.0, 0.0, 0.0],
        ),
        
        # Add missing perturbation config (required by parent)
        pert_config=config_dict.create(
            enable=False,  # Disable perturbations for navigation
            velocity_kick=[0.0, 3.0],
            kick_wait_times=[1.0, 3.0],
            kick_durations=[0.05, 0.2],
        ),
    )
    
    return config


class Go1HierarchicalNavigation(joystick.Joystick):
    """Go1 environment for hierarchical navigation with goal conditioning."""
    
    def __init__(self, 
                 task: str = "flat_terrain",  # Required by parent
                 config: Optional[config_dict.ConfigDict] = None,
                 config_overrides: Optional[dict] = None):
        """Initialize navigation environment.
        
        Args:
            task: Task type (ignored, we use custom XML)
            config: Environment configuration
            config_overrides: Additional config overrides
        """
        # Use navigation config by default
        if config is None:
            config = hierarchical_navigation_config()
        
        # First, properly initialize parent class
        # This will set up all required attributes
        super().__init__(task=task, config=config, config_overrides=config_overrides)
        
        # Now override with our custom XML
        xml_path = str(Path(__file__).parent / "go1_navigation_room.xml")
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mjx_model = mjx.device_put(self._mj_model)
        
        # Navigation-specific attributes
        self._room_size = config.room_size
        self._wall_buffer = config.wall_buffer
        self._goal_radius = config.reward_config.get('goal_radius', 0.5)
        self._min_goal_distance = config.min_goal_distance
        self._max_goal_distance = config.max_goal_distance
        
        # Get goal marker body ID
        self._goal_body_id = self.mj_model.body('goal_marker').id
        
        # Initialize goal position (will be randomized in reset)
        self._goal_position = jp.array([2.0, 2.0])
        
    
    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset environment with random robot and goal positions."""
        # Get base reset state
        state = super().reset(rng)
        
        # Split RNG
        rng, rng_robot, rng_goal = jax.random.split(rng, 3)
        
        # Randomize robot position
        robot_pos = self._sample_valid_position(rng_robot)
        
        # Randomize goal position (ensuring min distance from robot)
        goal_pos = self._sample_goal_position(rng_goal, robot_pos)
        self._goal_position = goal_pos
        
        # Update robot position in state
        qpos = state.data.qpos.at[:2].set(robot_pos)
        data = state.data.replace(qpos=qpos)
        
        # Update goal marker position in the simulation
        # Note: This is visual only, actual goal logic uses self._goal_position
        
        # Create new state with updated positions
        state = state.replace(data=data)
        
        # Update observations with goal info
        obs = self._get_obs(data, state.info)
        
        # Store goal in info for reward computation
        info = state.info.copy()
        info['goal_position'] = self._goal_position
        info['episode_start_time'] = 0.0
        
        return state.replace(obs=obs, info=info)
    
    def _sample_valid_position(self, rng: jax.Array) -> jax.Array:
        """Sample a valid position within room boundaries."""
        # Sample position with buffer from walls
        valid_range = self._room_size / 2 - self._wall_buffer
        pos = jax.random.uniform(rng, (2,), minval=-valid_range, maxval=valid_range)
        return pos
    
    def _sample_goal_position(self, rng: jax.Array, robot_pos: jax.Array) -> jax.Array:
        """Sample goal position with constraints."""
        def sample_until_valid(carry, _):
            rng, _ = carry
            rng, rng_pos = jax.random.split(rng)
            
            # Sample position
            goal_pos = self._sample_valid_position(rng_pos)
            
            # Check distance constraint
            distance = jp.linalg.norm(goal_pos - robot_pos)
            valid = jp.logical_and(
                distance >= self._min_goal_distance,
                distance <= self._max_goal_distance
            )
            
            return (rng, goal_pos), valid
        
        # Sample until we get a valid position
        _, goal_pos = jax.lax.while_loop(
            lambda carry: ~carry[1],  # Continue while not valid
            lambda carry: sample_until_valid(carry, None)[0],
            (rng, jp.zeros(2))
        )
        
        return goal_pos
    
    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, jax.Array]:
        """Get observations including navigation-specific information.
        
        Returns dict with:
        - 'state': Original 48D locomotion observations (for frozen policy)
        - 'navigation': 10D navigation observations (for navigation policy)
        - 'goal_info': Additional goal information for visualization
        """
        # Get base observations
        base_obs = super()._get_obs(data, info)
        
        # Extract robot position and heading
        robot_pos = data.qpos[:2]  # x, y position
        robot_heading = data.qpos[5]  # yaw angle
        
        # Get goal position from info
        goal_pos = info.get('goal_position', self._goal_position)
        
        # Compute goal-related observations
        goal_vec = goal_pos - robot_pos
        goal_distance = jp.linalg.norm(goal_vec)
        goal_direction = goal_vec / (goal_distance + 1e-6)  # Normalized
        
        # Transform goal direction to robot frame
        cos_h = jp.cos(robot_heading)
        sin_h = jp.sin(robot_heading)
        goal_direction_local = jp.array([
            goal_direction[0] * cos_h + goal_direction[1] * sin_h,
            -goal_direction[0] * sin_h + goal_direction[1] * cos_h
        ])
        
        # Compute wall distances
        wall_distances = jp.array([
            self._room_size / 2 - robot_pos[1],  # North
            self._room_size / 2 + robot_pos[1],  # South
            self._room_size / 2 - robot_pos[0],  # East
            self._room_size / 2 + robot_pos[0],  # West
        ])
        
        # Create navigation observations (10D)
        navigation_obs = jp.concatenate([
            goal_direction_local,      # 2D - goal direction in robot frame
            goal_distance[None],       # 1D - distance to goal
            robot_pos,                 # 2D - robot position
            jp.sin(robot_heading)[None],  # 1D - robot heading (sin)
            jp.cos(robot_heading)[None],  # 1D - robot heading (cos)
            wall_distances[:3],        # 3D - distances to N,S,E walls (W is redundant)
        ])
        
        # Return observations
        return {
            'state': base_obs['state'],  # 48D for locomotion policy
            'privileged_state': base_obs['privileged_state'],  # For value function
            'navigation': navigation_obs,  # 10D for navigation policy
            'goal_info': {
                'goal_position': goal_pos,
                'goal_distance': goal_distance,
                'robot_position': robot_pos,
                'robot_heading': robot_heading,
            }
        }
    
    def _get_reward(
        self, 
        data: mjx.Data, 
        action: jax.Array,
        info: Dict[str, Any],
        first_contact: jax.Array,
        done: jax.Array,
        contact: jax.Array,
        qvel_history: jax.Array,
        metrics: Dict[str, Any]
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Compute navigation-focused rewards."""
        # Get base rewards (minimal locomotion rewards)
        _, base_rewards = super()._get_reward(
            data, action, info, first_contact, done, contact, qvel_history, metrics
        )
        
        # Zero out tracking rewards since we don't use joystick commands
        base_rewards['tracking_lin_vel'] = 0.0
        base_rewards['tracking_ang_vel'] = 0.0
        
        # Get positions
        robot_pos = data.qpos[:2]
        goal_pos = info.get('goal_position', self._goal_position)
        
        # Goal-based rewards
        goal_vec = goal_pos - robot_pos
        goal_distance = jp.linalg.norm(goal_vec)
        
        # Goal reached (sparse)
        goal_reached = goal_distance < self._goal_radius
        rewards = base_rewards.copy()
        rewards['goal_reached'] = goal_reached.astype(jp.float32)
        
        # Goal progress (dense) - reward reducing distance
        prev_pos = info.get('prev_robot_pos', robot_pos)
        prev_distance = jp.linalg.norm(goal_pos - prev_pos)
        progress = prev_distance - goal_distance
        rewards['goal_progress'] = progress
        
        # Goal distance penalty
        rewards['goal_distance'] = -goal_distance
        
        # Heading alignment - reward facing toward goal
        robot_heading = data.qpos[5]
        goal_angle = jp.arctan2(goal_vec[1], goal_vec[0])
        angle_diff = jp.abs(jp.mod(goal_angle - robot_heading + jp.pi, 2 * jp.pi) - jp.pi)
        rewards['heading_alignment'] = jp.exp(-angle_diff)
        
        # Wall collision penalty
        wall_distances = jp.array([
            self._room_size / 2 - jp.abs(robot_pos[1]),  # N/S walls
            self._room_size / 2 - jp.abs(robot_pos[0]),  # E/W walls
        ])
        min_wall_dist = jp.min(wall_distances)
        wall_collision = min_wall_dist < 0.3
        rewards['wall_collision'] = -wall_collision.astype(jp.float32)
        
        # Boundary penalty (soft wall avoidance)
        boundary_penalty = jp.exp(-min_wall_dist) - 1.0
        rewards['boundary_penalty'] = boundary_penalty
        
        return rewards, rewards
    
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step environment.
        
        Note: This expects 12D joint actions. The 3D velocity commands
        are converted to joint actions by the HierarchicalPolicyWrapper.
        """
        # Store previous position for progress reward
        info = state.info.copy()
        info['prev_robot_pos'] = state.data.qpos[:2]
        
        # Step with base implementation
        next_state = super().step(state.replace(info=info), action)
        
        # Update observations with navigation info
        obs = self._get_obs(next_state.data, next_state.info)
        
        # Check if goal reached for early termination
        goal_distance = obs['goal_info']['goal_distance']
        goal_reached = goal_distance < self._goal_radius
        
        # Terminate episode if goal reached
        done = jp.logical_or(next_state.done, goal_reached)
        
        return next_state.replace(obs=obs, done=done)