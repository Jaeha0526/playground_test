"""Go1 Navigation Environment with goal conditioning and room constraints."""

import jax
import jax.numpy as jp
import mujoco
from typing import Dict, Any, Optional, Union
from pathlib import Path
import numpy as np

from ml_collections import config_dict
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import go1_constants as consts
from mujoco_playground._src import mjx_env
from mujoco import mjx


def default_navigation_config() -> config_dict.ConfigDict:
    """Default configuration for Go1 navigation environment."""
    return config_dict.create(
        # Environment settings
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=250,  # Start with Stage 1 curriculum
        
        # PD controller gains (from original Go1)
        Kp=35.0,
        Kd=0.5,
        action_repeat=1,
        action_scale=0.5,
        soft_joint_pos_limit_factor=0.95,
        
        # Room configuration
        room_size=6.0,  # Start with smaller room for Stage 1
        wall_buffer=1.0,  # Minimum distance from walls for spawn/goal
        goal_radius=0.5,  # Success threshold
        min_goal_distance=1.0,  # Minimum spawn-goal separation
        max_goal_distance=3.0,  # Maximum goal distance for Stage 1
        
        # Noise configuration (reduced for navigation)
        noise_config=config_dict.create(
            level=0.5,  # Reduced noise for cleaner navigation signals
            scales=config_dict.create(
                joint_pos=0.02,
                joint_vel=1.0,
                gyro=0.1,
                gravity=0.03,
                linvel=0.05,
            ),
        ),
        
        # Navigation reward configuration
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Main navigation objectives
                goal_reached=100.0,        # Sparse success reward
                goal_progress=2.0,         # Dense progress reward
                goal_distance=-0.1,        # Distance penalty
                
                # Safety constraints
                wall_collision=-10.0,      # Wall collision penalty
                boundary_penalty=-0.5,     # Stay away from walls
                
                # Efficiency incentives
                time_penalty=-0.1,         # Encourage speed
                heading_alignment=0.2,     # Face toward goal
                
                # Keep some locomotion quality (for stability)
                orientation=-1.0,          # Stay upright (reduced weight)
                termination=-1.0,          # Don't fall
            ),
            
            # Navigation-specific parameters
            goal_radius=0.5,
            wall_collision_threshold=0.3,
            boundary_warning_distance=1.0,
        ),
        
        # Command generation (not used - replaced by navigation)
        command_config=config_dict.create(
            # These will be overridden by navigation layer
            a=[0.0, 0.0, 0.0],  # No random commands
            b=[0.0, 0.0, 0.0],
        ),
    )


class Go1NavigationEnv(go1_base.Go1Env):
    """Go1 environment for goal-conditioned navigation in a room."""
    
    def __init__(
        self,
        config: config_dict.ConfigDict = default_navigation_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        hierarchical_policy=None,
    ):
        # Store hierarchical policy
        self._hierarchical_policy = hierarchical_policy
        self._hierarchical_mode = hierarchical_policy is not None
        
        # Get the path to our custom room XML
        xml_path = Path(__file__).parent / "go1_room_environment.xml"
        
        super().__init__(
            xml_path=xml_path.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        
        # Navigation-specific initialization
        self._room_size = self._config.room_size
        self._wall_buffer = self._config.wall_buffer
        self._goal_radius = self._config.reward_config.goal_radius
        self._min_goal_distance = self._config.min_goal_distance
        self._max_goal_distance = self._config.max_goal_distance
        
        # Get wall and goal marker body IDs
        self._goal_marker_id = self.mj_model.body('goal_marker').id
        self._wall_geom_ids = [
            self.mj_model.geom('wall_north_geom').id,
            self.mj_model.geom('wall_south_geom').id,
            self.mj_model.geom('wall_east_geom').id,
            self.mj_model.geom('wall_west_geom').id,
        ]
        
        self._post_init()
    
    def _post_init(self) -> None:
        """Post-initialization setup for navigation environment."""
        # Simplified initialization for navigation
        # Create default pose (standing position)
        self._default_pose = jp.zeros(12)  # 12 joint angles, all zero for standing
        
        # Joint limits (first joint is freejoint) 
        if hasattr(self.mj_model, 'jnt_range') and len(self.mj_model.jnt_range) > 1:
            self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
            self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
            self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor
        else:
            # Default joint limits
            self._lowers = jp.full(12, -jp.pi)
            self._uppers = jp.full(12, jp.pi)
            self._soft_lowers = self._lowers * 0.95
            self._soft_uppers = self._uppers * 0.95

        # Try to get body IDs (with fallback for missing elements)
        try:
            self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
            self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        except:
            self._torso_body_id = 0  # Base body
            self._torso_mass = 1.0   # Default mass
        
        # Try to get floor geom ID
        try:
            self._floor_geom_id = self._mj_model.geom("floor").id
        except:
            self._floor_geom_id = 0  # Default to first geom
        
        # Simplified sensor setup
        self._feet_site_id = np.array([0, 1, 2, 3])  # Default site IDs
        self._feet_geom_id = np.array([0, 1, 2, 3])  # Default geom IDs  
        self._imu_site_id = 0  # Default IMU site
        self._foot_linvel_sensor_adr = np.array([0, 1, 2, 3])  # Default sensor addresses
    
    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset environment with random spawn and goal positions."""
        # Initialize the environment data properly
        data = mjx.make_data(self.mjx_model)
        
        # Generate random spawn position
        rng, spawn_key = jax.random.split(rng)
        spawn_pos = self._generate_random_position(spawn_key, self._wall_buffer)
        
        # Generate random goal position (ensuring minimum distance from spawn)
        rng, goal_key = jax.random.split(rng)
        goal_pos = self._generate_goal_position(goal_key, spawn_pos)
        
        # Set initial robot position
        qpos = data.qpos.at[:2].set(spawn_pos[:2])
        qpos = qpos.at[2].set(0.3)  # Set height above ground
        qpos = qpos.at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))  # Identity quaternion
        qpos = qpos.at[7:].set(self._default_pose)  # Joint positions
        
        # Update data with new positions
        data = data.replace(qpos=qpos)
        
        # Step once to initialize physics
        data = mjx.step(self.mjx_model, data)
        
        # Create info dictionary
        info = {
            "goal_position": goal_pos,
            "spawn_position": spawn_pos[:2],
            "previous_distance_to_goal": jp.linalg.norm(spawn_pos[:2] - goal_pos[:2]),
            "episode_start_time": 0
        }
        
        # Get initial observations (create basic obs to avoid parent class issues)
        # We'll compute full observations properly after parent class is fixed
        obs = {
            "state": jp.zeros(48),  # Placeholder for locomotion state
            "navigation": self._get_navigation_obs(data, info),
        }
        
        # Initialize metrics with all keys that will be used in step
        metrics = {
            "reward/goal_reached": jp.array(0.0),
            "reward/goal_progress": jp.array(0.0),
            "reward/goal_distance": jp.array(0.0),
            "reward/wall_collision": jp.array(0.0),
            "reward/boundary_penalty": jp.array(0.0),
            "reward/time_penalty": jp.array(0.0),
            "reward/heading_alignment": jp.array(0.0),
            "reward/termination": jp.array(0.0),
            "reward/orientation": jp.array(0.0),
            "navigation/goal_distance": jp.linalg.norm(spawn_pos[:2] - goal_pos[:2]),
            "navigation/goal_reached": jp.array(0.0),
            "navigation/episode_time": jp.array(0.0),
        }
        
        # Create initial state
        state = mjx_env.State(
            data=data,
            obs=obs,
            reward=jp.array(0.0),
            done=jp.array(False),
            metrics=metrics,
            info=info
        )
        
        return state
    
    def _generate_random_position(self, rng: jax.Array, buffer: float) -> jax.Array:
        """Generate random position within room bounds."""
        max_coord = self._room_size / 2 - buffer
        min_coord = -max_coord
        
        rng, x_key, y_key = jax.random.split(rng, 3)
        x = jax.random.uniform(x_key, minval=min_coord, maxval=max_coord)
        y = jax.random.uniform(y_key, minval=min_coord, maxval=max_coord)
        
        return jp.array([x, y])
    
    def _generate_goal_position(self, rng: jax.Array, spawn_pos: jax.Array) -> jax.Array:
        """Generate goal position with distance constraints."""
        max_attempts = 20
        
        def attempt_goal_generation(carry, _):
            rng, attempts = carry
            rng, goal_key = jax.random.split(rng)
            
            goal_pos = self._generate_random_position(goal_key, self._wall_buffer)
            distance = jp.linalg.norm(goal_pos - spawn_pos[:2])
            
            # Check distance constraints
            valid = (distance >= self._min_goal_distance) & (distance <= self._max_goal_distance)
            
            return (rng, attempts + 1), (goal_pos, distance, valid)
        
        # Try multiple times to generate valid goal
        (final_rng, _), (goals, distances, valid_flags) = jax.lax.scan(
            attempt_goal_generation, 
            (rng, 0), 
            None, 
            length=max_attempts
        )
        
        # Select first valid goal, or use last attempt if none valid
        valid_idx = jp.argmax(valid_flags)
        selected_goal = goals[valid_idx]
        
        return selected_goal
    
    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, jax.Array]:
        """Get observations including navigation information."""
        # Try to get base locomotion observations, fallback if parent class incomplete
        try:
            base_obs = super()._get_obs(data, info)
        except (AttributeError, NotImplementedError):
            # Create minimal locomotion observations as fallback
            # This includes basic robot state information
            base_obs = {
                "state": jp.concatenate([
                    data.qpos[7:],      # Joint positions (12)
                    data.qvel[6:],      # Joint velocities (12)
                    data.qpos[:3],      # Base position (3)
                    data.qpos[3:7],     # Base orientation quaternion (4)
                    data.qvel[:6],      # Base linear and angular velocity (6)
                    jp.zeros(11),       # Padding to reach 48 dimensions
                ])
            }
        
        # Add navigation observations
        nav_obs = self._get_navigation_obs(data, info)
        
        return {
            **base_obs,
            "navigation": nav_obs,
        }
    
    def _get_navigation_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Get navigation-specific observations (10-dim)."""
        robot_pos = data.qpos[:2]  # [x, y]
        robot_heading = data.qpos[6]  # yaw rotation
        goal_pos = info["goal_position"]
        
        # Goal direction and distance
        goal_vector = goal_pos - robot_pos
        goal_distance = jp.linalg.norm(goal_vector)
        goal_direction = goal_vector / (goal_distance + 1e-6)  # Normalized
        
        # Normalized goal distance (by room diagonal)
        room_diagonal = jp.sqrt(2) * self._room_size
        normalized_distance = goal_distance / room_diagonal
        
        # Wall distances (North, South, East, West)
        room_half = self._room_size / 2
        wall_distances = jp.array([
            room_half - robot_pos[1],  # North wall
            room_half + robot_pos[1],  # South wall  
            room_half - robot_pos[0],  # East wall
            room_half + robot_pos[0],  # West wall
        ])
        
        # Normalize wall distances
        wall_distances = wall_distances / self._room_size
        
        # Navigation observation vector (10-dim)
        nav_obs = jp.concatenate([
            goal_direction,          # 2-dim: [dx, dy] normalized
            jp.array([normalized_distance]),  # 1-dim: distance to goal
            robot_pos / room_half,   # 2-dim: normalized robot position
            jp.array([robot_heading]), # 1-dim: robot yaw angle
            wall_distances,          # 4-dim: distances to walls
        ])
        
        return nav_obs
    
    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: Dict[str, Any],
        metrics: Dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> Dict[str, jax.Array]:
        """Compute navigation rewards."""
        robot_pos = data.qpos[:2]
        goal_pos = info["goal_position"]
        
        # Current distance to goal
        current_distance = jp.linalg.norm(robot_pos - goal_pos)
        previous_distance = info["previous_distance_to_goal"]
        
        # Update distance tracking
        info["previous_distance_to_goal"] = current_distance
        
        rewards = {}
        
        # 1. Goal reached (sparse)
        goal_reached = current_distance < self._goal_radius
        rewards["goal_reached"] = goal_reached.astype(jp.float32)
        
        # 2. Progress toward goal (dense)
        progress = previous_distance - current_distance
        rewards["goal_progress"] = progress
        
        # 3. Distance penalty (encourage proximity)
        rewards["goal_distance"] = -current_distance
        
        # 4. Wall collision detection
        wall_collision = self._check_wall_collision(robot_pos)
        rewards["wall_collision"] = wall_collision.astype(jp.float32)
        
        # 5. Boundary warning (stay away from walls)
        boundary_warning = self._check_boundary_warning(robot_pos)
        rewards["boundary_penalty"] = boundary_warning.astype(jp.float32)
        
        # 6. Time penalty (encourage efficiency)
        rewards["time_penalty"] = jp.ones_like(current_distance)
        
        # 7. Heading alignment (face toward goal)
        heading_alignment = self._compute_heading_alignment(data, goal_pos)
        rewards["heading_alignment"] = heading_alignment
        
        # 8. Basic stability (reduced weight since locomotion is frozen)
        upvector = self.get_upvector(data)
        rewards["orientation"] = -jp.sum(jp.square(upvector[:2]))
        
        # 9. Termination penalty
        rewards["termination"] = done.astype(jp.float32)
        
        return rewards
    
    def _check_wall_collision(self, robot_pos: jax.Array) -> jax.Array:
        """Check if robot is too close to walls."""
        room_half = self._room_size / 2
        threshold = self._config.reward_config.wall_collision_threshold
        
        # Distance to each wall
        distances_to_walls = jp.array([
            room_half - jp.abs(robot_pos[0]),  # Distance to east/west walls
            room_half - jp.abs(robot_pos[1]),  # Distance to north/south walls
        ])
        
        # Collision if any wall distance < threshold
        collision = jp.any(distances_to_walls < threshold)
        return collision
    
    def _check_boundary_warning(self, robot_pos: jax.Array) -> jax.Array:
        """Check if robot is approaching walls."""
        room_half = self._room_size / 2
        warning_distance = self._config.reward_config.boundary_warning_distance
        
        distances_to_walls = jp.array([
            room_half - jp.abs(robot_pos[0]),
            room_half - jp.abs(robot_pos[1]),
        ])
        
        # Warning if any wall distance < warning_distance
        warning = jp.any(distances_to_walls < warning_distance)
        return warning
    
    def _compute_heading_alignment(self, data: mjx.Data, goal_pos: jax.Array) -> jax.Array:
        """Compute reward for facing toward goal."""
        robot_pos = data.qpos[:2]
        robot_heading = data.qpos[6]  # yaw
        
        # Desired heading toward goal
        goal_vector = goal_pos - robot_pos
        desired_heading = jp.arctan2(goal_vector[1], goal_vector[0])
        
        # Heading error
        heading_error = jp.abs(jp.mod(robot_heading - desired_heading + jp.pi, 2*jp.pi) - jp.pi)
        
        # Exponential reward for good alignment
        alignment_reward = jp.exp(-heading_error / (jp.pi/4))  # Max error = pi/4 for good reward
        
        return alignment_reward
    
    def _get_termination(self, data: mjx.Data) -> jax.Array:
        """Check for episode termination."""
        # Fall termination (from base class)
        fall_termination = self.get_upvector(data)[-1] < 0.0
        
        # Goal reached termination
        robot_pos = data.qpos[:2]
        # Note: goal_position is in info, but info not available here
        # We'll handle goal termination in the step function
        
        return fall_termination
    
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step the environment with navigation logic."""
        # Use base step function but override command generation
        # The action here should be velocity commands from navigation layer
        
        # Convert navigation commands to motor targets
        motor_targets = self._default_pose + action * self._config.action_scale
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )
        
        # Contact detection (from base class)
        contact = jp.array([
            # Note: collision detection would need floor_geom_id
            # For now, assume no contact issues with walls
            False, False, False, False  # [FR, FL, RR, RL]
        ])
        first_contact = jp.array([False, False, False, False])
        
        # Get observations
        obs = self._get_obs(data, state.info)
        
        # Check termination
        fall_done = self._get_termination(data)
        
        # Check goal reached
        robot_pos = data.qpos[:2]
        goal_pos = state.info["goal_position"]
        goal_distance = jp.linalg.norm(robot_pos - goal_pos)
        goal_done = goal_distance < self._goal_radius
        
        # Combined termination
        done = fall_done | goal_done
        
        # Compute rewards
        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        
        # Scale rewards
        scaled_rewards = {
            k: v * self._config.reward_config.scales[k] 
            for k, v in rewards.items()
        }
        
        # Total reward
        reward = jp.clip(sum(scaled_rewards.values()) * self.dt, 0.0, 10000.0)
        
        # Update metrics
        for k, v in scaled_rewards.items():
            state.metrics[f"reward/{k}"] = v
        
        # Add navigation metrics
        state.metrics["navigation/goal_distance"] = goal_distance
        state.metrics["navigation/goal_reached"] = goal_done.astype(jp.float32)
        state.metrics["navigation/episode_time"] = jp.array(state.info.get("episode_start_time", 0) + 1, dtype=jp.float32)
        
        # Update info
        state.info["episode_start_time"] = state.info.get("episode_start_time", 0) + 1
        
        # Create new state
        # Keep done as bool, don't convert to float
        new_state = state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
        )
        
        return new_state