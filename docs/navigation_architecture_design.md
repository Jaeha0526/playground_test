# Go1 Navigation Architecture & Training Design

This document summarizes the hierarchical navigation architecture and training design for the Go1 quadruped robot, based on leveraging the existing locomotion policy for goal-conditioned navigation tasks.

## 🏗️ Architecture Overview

### Core Concept: Hierarchical Policy
We designed a **two-layer hierarchical architecture** that leverages the existing well-trained Go1 locomotion policy:

```
Goal + Robot State → [Navigation Layer] → Velocity Commands → [Locomotion Layer] → Joint Actions
```

### Key Design Decisions:
- **Preserve locomotion skills**: Keep existing Go1 policy frozen
- **Add navigation layer**: Small network that outputs velocity commands
- **Maintain interface**: Navigation outputs same `[vx, vy, vyaw]` format as original joystick commands
- **Modular training**: Train only the navigation layer while locomotion stays fixed

## 🧠 Policy Architecture

### Navigation Layer (Trainable)
```python
class NavigationController:
    Input:  goal_direction(2) + goal_distance(1) + robot_position(2) + 
            robot_heading(1) + wall_distances(4) = 10 dimensions
    
    Network: [64] → ReLU → [32] → ReLU → [3]
    
    Output: velocity_commands [vx, vy, vyaw]  # Same as joystick interface
```

### Locomotion Layer (Frozen)
```python
# Existing Go1 policy (pre-trained checkpoint)
locomotion_policy(velocity_commands + proprioceptive_sensors) → joint_actions

# Architecture: [512, 256, 128] (policy) + [512, 256, 128] (value)
# Parameters: ~100K (frozen, excluded from optimizer)
```

### Parameter Management
```python
# Only navigation parameters are trainable (~1K parameters)
train_state = TrainState.create(
    apply_fn=navigation_network.apply,
    params=navigation_params,  # Only these get gradients
    tx=optimizer
)

# Locomotion parameters loaded from checkpoint (frozen)
locomotion_params = load_checkpoint("path/to/go1_checkpoint")['params']
frozen_locomotion_params = jax.lax.stop_gradient(locomotion_params)
```

## 🏠 Environment Design

### Square Room Setup
```python
environment_specs = {
    'room_size': 10.0,           # 10m × 10m square room
    'wall_height': 0.5,          # 0.5m high walls
    'spawn_buffer': 1.0,         # 1m from walls for spawn/goal
    'goal_radius': 0.5,          # 0.5m success threshold
    'min_goal_distance': 2.0,    # Minimum spawn-goal separation
}
```

### XML Structure
```xml
<!-- Room with 4 walls -->
<geom name="wall_north" pos="0 5 0.25" size="5 0.1 0.25" type="box"/>
<geom name="wall_south" pos="0 -5 0.25" size="5 0.1 0.25" type="box"/>
<geom name="wall_east" pos="5 0 0.25" size="0.1 5 0.25" type="box"/>
<geom name="wall_west" pos="-5 0 0.25" size="0.1 5 0.25" type="box"/>
```

### Randomization Strategy
- **Random spawn**: Anywhere in room (>1m from walls)
- **Random goals**: Anywhere in room (>1m from walls, >2m from spawn)
- **Each episode**: New spawn and goal positions

## 🎯 Reward Design

### Navigation-Only Rewards (Locomotion Frozen)
```python
navigation_rewards = {
    # Main objectives
    'goal_reached': +100.0,      # Sparse: reach goal position
    'goal_progress': +2.0,       # Dense: distance improvement per step
    'goal_distance': -0.1,       # Dense: penalty for being far from goal
    
    # Safety constraints
    'wall_collision': -10.0,     # Penalty for hitting walls
    'boundary_penalty': -0.5,    # Stay away from wall proximity
    
    # Efficiency incentives  
    'time_penalty': -0.1,        # Encourage fast completion
    'heading_alignment': +0.2,   # Reward facing toward goal
}
```

### Reward Implementation
```python
def compute_navigation_reward(state, goal, prev_state):
    # Dense progress reward (every step)
    dist = jnp.linalg.norm(state.pos[:2] - goal[:2])
    prev_dist = jnp.linalg.norm(prev_state.pos[:2] - goal[:2])
    progress = (prev_dist - dist) * 2.0
    
    # Distance penalty (encourage proximity)
    distance_penalty = -dist * 0.1
    
    # Sparse success reward
    goal_reached = dist < 0.5
    sparse_reward = 100.0 if goal_reached else 0.0
    
    # Wall collision detection
    wall_collision = check_wall_proximity(state.pos, threshold=0.3)
    collision_penalty = -10.0 if wall_collision else 0.0
    
    return {
        'total': progress + distance_penalty + sparse_reward + collision_penalty,
        'goal_reached': goal_reached,
        'components': {
            'progress': progress,
            'distance': distance_penalty, 
            'sparse': sparse_reward,
            'collision': collision_penalty
        }
    }
```

### Removed Locomotion Rewards
Since locomotion parameters are frozen, we removed:
- `tracking_lin_vel`, `tracking_ang_vel` (not optimizing these)
- `orientation`, `pose`, `feet_*` (locomotion quality preserved)
- `stand_still` (conflicts with navigation goals)

## 🎓 Curriculum Learning Design

### Progressive Training Stages
```python
curriculum_stages = [
    {
        # Stage 1: Close Goals (Learn basic navigation)
        'goal_distance_range': (1.0, 3.0),
        'episode_length': 250,        # 5 seconds
        'trajectory_length': 25,      # 0.5 seconds
        'room_size': 6.0,            # Smaller room
        'obstacles': None,
        'expected_success_rate': 0.8,
    },
    {
        # Stage 2: Medium Goals (Learn path planning)  
        'goal_distance_range': (2.0, 5.0),
        'episode_length': 400,        # 8 seconds
        'trajectory_length': 40,      # 0.8 seconds
        'room_size': 8.0,
        'obstacles': None,
        'expected_success_rate': 0.7,
    },
    {
        # Stage 3: Full Room Navigation
        'goal_distance_range': (3.0, 8.0),
        'episode_length': 600,        # 12 seconds
        'trajectory_length': 50,      # 1.0 second
        'room_size': 10.0,
        'obstacles': 'simple',        # Add static obstacles
        'expected_success_rate': 0.6,
    },
    {
        # Stage 4: Complex Navigation
        'goal_distance_range': (5.0, 12.0),
        'episode_length': 1000,       # 20 seconds
        'trajectory_length': 50,      # 1.0 second
        'room_size': 10.0,
        'obstacles': 'complex',       # Multiple obstacles
        'expected_success_rate': 0.5,
    }
]
```

### Stage Advancement Criteria
```python
advancement_criteria = {
    'success_rate': 0.8,          # 80% episodes reach goal
    'time_efficiency': 0.7,       # Reach goal in 70% of optimal time
    'stability': 100,             # Consistent performance over 100 episodes
}
```

## ⚙️ Training Configuration

### PPO Hyperparameters (Navigation-Specific)
```python
navigation_ppo_config = {
    # Reduced scale for navigation-only training
    'num_timesteps': 50_000_000,     # 50M steps (vs 200M for locomotion)
    'num_envs': 4096,                # 4K parallel envs (vs 8K for locomotion)
    'unroll_length': 25-50,          # Longer trajectories (vs 20 for locomotion)
    'episode_length': 250-1000,      # Variable by curriculum stage
    
    # Standard PPO settings
    'batch_size': 128,               # Environments per minibatch
    'num_minibatches': 32,           # 4096 / 128 = 32
    'num_updates_per_batch': 4,      # Epochs through data
    'learning_rate': 1e-3,           # Higher LR (smaller network)
    'discounting': 0.97,
    'entropy_cost': 1e-2,
    'max_grad_norm': 1.0,
}
```

### Network Configuration
```python
navigation_network_config = {
    'hidden_layers': [64, 32],       # Much smaller than locomotion
    'input_dim': 10,                 # Navigation observations
    'output_dim': 3,                 # [vx, vy, vyaw] commands
    'activation': 'relu',
    'parameters': ~1000,             # vs ~100K for locomotion
}
```

### Training Timeline
```python
training_phases = {
    'Stage 1': '10M steps (close goals)',
    'Stage 2': '15M steps (medium goals)', 
    'Stage 3': '15M steps (full room)',
    'Stage 4': '10M steps (complex)',
    'Total': '50M steps (~2-4 hours on GPU)'
}
```

## 📊 Observation Space Design

### Navigation Observations (10-dim)
```python
navigation_obs = jnp.concatenate([
    goal_direction,      # 2-dim: normalized vector to goal [dx, dy]
    goal_distance,       # 1-dim: distance to goal (normalized by room size)
    robot_position,      # 2-dim: [x, y] in room coordinates  
    robot_heading,       # 1-dim: yaw angle (radians)
    wall_distances,      # 4-dim: distance to each wall [N, S, E, W]
])
```

### Locomotion Observations (48-dim, unchanged)
```python
# Existing Go1 format (preserved)
locomotion_obs = jnp.concatenate([
    velocity_commands,   # 3-dim: from navigation layer
    joint_positions,     # 12-dim: relative to default pose
    joint_velocities,    # 12-dim: with noise
    imu_data,           # 9-dim: linear vel, gyro, gravity  
    previous_action,     # 12-dim: last joint targets
])
```

## 🔄 Training Process

### Hierarchical Forward Pass
```python
def hierarchical_policy(obs, locomotion_params, navigation_params):
    # 1. Extract navigation observations
    nav_obs = extract_navigation_obs(obs)  # 10-dim
    
    # 2. Navigation layer (trainable)
    velocity_commands = navigation_network.apply(navigation_params, nav_obs)
    
    # 3. Locomotion layer (frozen)
    loco_obs = extract_locomotion_obs(obs, velocity_commands)  # 48-dim
    joint_actions = locomotion_network.apply(locomotion_params, loco_obs)
    
    return joint_actions
```

### Training Loop
```python
def train_step(train_state, locomotion_params, batch):
    def loss_fn(navigation_params):
        # Forward pass (locomotion params frozen)
        actions = hierarchical_policy(
            batch['obs'], 
            jax.lax.stop_gradient(locomotion_params),  # No gradients
            navigation_params                          # Trainable
        )
        
        # Compute navigation rewards only
        rewards = compute_navigation_rewards(batch, actions)
        loss = compute_ppo_loss(rewards, actions, batch)
        return loss
    
    # Update only navigation parameters
    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    
    return train_state, loss
```

## 🎯 Expected Outcomes

### Performance Targets
```python
success_metrics = {
    'Stage 1': 'Success rate > 90% for goals 1-3m away',
    'Stage 2': 'Success rate > 80% for goals 2-5m away',
    'Stage 3': 'Success rate > 70% for goals 3-8m away with obstacles',
    'Stage 4': 'Success rate > 60% for complex navigation scenarios',
}

efficiency_metrics = {
    'Path efficiency': '> 70% of optimal path length',
    'Time efficiency': '> 70% of optimal time',
    'Collision rate': '< 5% episodes with wall collisions',
}
```

### Generalization Testing
```python
generalization_tests = {
    'Room sizes': 'Test on 8m, 12m, 15m rooms',
    'Goal distances': 'Test on longer distances than training',
    'Obstacle configurations': 'Test on unseen obstacle layouts',
    'Dynamic obstacles': 'Test with moving obstacles',
}
```

## 🔧 Implementation Files

### Recommended File Structure
```
src/locomotion_training/
├── envs/
│   ├── go1_navigation.py          # Custom navigation environment
│   └── room_environments.xml      # Room XML with walls
├── policies/
│   ├── hierarchical_policy.py     # Combined navigation + locomotion
│   └── navigation_network.py      # Small navigation network
├── training/
│   ├── navigation_trainer.py      # Training orchestration
│   └── curriculum_manager.py      # Stage progression logic
└── config/
    ├── navigation_config.py       # Training configurations
    └── curriculum_config.py       # Curriculum stage definitions
```

## 💡 Key Benefits

1. **Leverage existing skills**: Preserve 200M steps of locomotion training
2. **Fast training**: Only 50M steps needed for navigation vs 200M+ for full policy
3. **Modular design**: Navigation layer can be swapped/improved independently
4. **Stable foundation**: Locomotion quality guaranteed by frozen parameters
5. **Clear debugging**: Can isolate navigation vs locomotion issues
6. **Curriculum learning**: Progressive difficulty ensures robust learning

## 🚀 Next Steps

1. **Implement environment**: Create room environment with goal randomization
2. **Load locomotion checkpoint**: Set up parameter freezing mechanism
3. **Design navigation network**: Implement small 2-layer network
4. **Start Stage 1 training**: Begin with close goals and short episodes
5. **Monitor metrics**: Track success rate, path efficiency, collision rate
6. **Progress through curriculum**: Advance stages based on performance criteria

This design provides a solid foundation for training robust goal-conditioned navigation while leveraging the existing locomotion capabilities of the Go1 robot.