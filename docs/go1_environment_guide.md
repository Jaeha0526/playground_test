# Go1JoystickFlatTerrain Environment Analysis

This guide provides a detailed breakdown of the Go1JoystickFlatTerrain environment for research modifications.

## 📁 Key Files Location

All files are in: `venv/lib/python3.12/site-packages/mujoco_playground/_src/locomotion/go1/`

- **`joystick.py`** - Main environment implementation
- **`base.py`** - Base Go1 environment class
- **`go1_constants.py`** - Constants and configuration
- **`randomize.py`** - Domain randomization
- **`xmls/`** - MuJoCo XML model files

## 🎛️ Configuration System

### Default Configuration (lines 32-93 in joystick.py)

```python
def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,           # Control timestep (50 Hz)
      sim_dt=0.004,           # Simulation timestep (250 Hz)
      episode_length=1000,    # Max episode steps
      Kp=35.0,               # PD controller P gain
      Kd=0.5,                # PD controller D gain
      action_repeat=1,        # Action repeat
      action_scale=0.5,       # Action scaling factor
      
      # Observation noise
      noise_config=config_dict.create(
          level=1.0,          # Noise multiplier
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      
      # Reward system
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking rewards (positive)
              tracking_lin_vel=1.0,    # Track linear velocity commands
              tracking_ang_vel=0.5,    # Track angular velocity commands
              pose=0.5,                # Stay close to default pose
              feet_air_time=0.1,       # Reward proper foot swing
              
              # Penalty terms (negative)
              lin_vel_z=-0.5,          # Penalize vertical motion
              ang_vel_xy=-0.05,        # Penalize roll/pitch
              orientation=-5.0,        # Keep body upright
              dof_pos_limits=-1.0,     # Stay within joint limits
              termination=-1.0,        # Avoid falling
              stand_still=-1.0,        # Move when commanded
              torques=-0.0002,         # Minimize torque usage
              action_rate=-0.01,       # Smooth actions
              energy=-0.001,           # Minimize energy consumption
              feet_clearance=-2.0,     # Proper foot clearance
              feet_height=-0.2,        # Consistent foot height
              feet_slip=-0.1,          # Avoid foot slipping
          ),
          tracking_sigma=0.25,         # Tracking tolerance
          max_foot_height=0.1,         # Target foot swing height
      ),
      
      # Command generation
      command_config=config_dict.create(
          a=[1.5, 0.8, 1.2],          # Max command amplitude [vx, vy, vyaw]
          b=[0.9, 0.25, 0.5],         # Command probability weights
      ),
  )
```

## 🏃 Environment Architecture

### 1. State Representation (lines 295-377)

**Observation (`state`)**: 48 dimensions
```python
state = jp.hstack([
    noisy_linvel,                    # 3 - Local linear velocity (with noise)
    noisy_gyro,                      # 3 - Angular velocity (with noise)
    noisy_gravity,                   # 3 - Gravity vector (with noise)
    noisy_joint_angles - default_pose, # 12 - Joint angles relative to default
    noisy_joint_vel,                 # 12 - Joint velocities (with noise)
    last_act,                        # 12 - Previous action
    command,                         # 3 - Current velocity command [vx, vy, vyaw]
])
```

**Privileged State**: 95 dimensions (includes ground truth data for training)

### 2. Action Space
- **12 joint targets** for the 12 actuated joints
- Actions are scaled by `action_scale=0.5`
- Applied as offsets to the default pose

### 3. Reward System (lines 379-417)

The reward function combines multiple terms:

```python
def _get_reward(self, ...):
    return {
        # Tracking rewards (encourage following commands)
        "tracking_lin_vel": self._reward_tracking_lin_vel(...),  # Exponential reward
        "tracking_ang_vel": self._reward_tracking_ang_vel(...),  # Exponential reward
        
        # Base stability costs
        "lin_vel_z": self._cost_lin_vel_z(...),                 # Squared penalty
        "ang_vel_xy": self._cost_ang_vel_xy(...),               # Squared penalty
        "orientation": self._cost_orientation(...),             # Squared penalty
        
        # Control costs
        "torques": self._cost_torques(...),                     # L1 + L2 penalty
        "action_rate": self._cost_action_rate(...),             # Action smoothness
        "energy": self._cost_energy(...),                       # Power consumption
        
        # Gait costs
        "feet_slip": self._cost_feet_slip(...),                 # Contact slip
        "feet_clearance": self._cost_feet_clearance(...),       # Foot clearance
        "feet_height": self._cost_feet_height(...),             # Swing height
        "feet_air_time": self._reward_feet_air_time(...),       # Air time reward
        
        # Other
        "pose": self._reward_pose(...),                         # Default pose
        "stand_still": self._cost_stand_still(...),             # Move when commanded
        "termination": self._cost_termination(...),             # Avoid falling
        "dof_pos_limits": self._cost_joint_pos_limits(...),     # Joint limits
    }
```

### Key Reward Functions

**1. Tracking Rewards (lines 421-437)**
```python
def _reward_tracking_lin_vel(self, commands, local_vel):
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self.tracking_sigma)  # Exponential reward

def _reward_tracking_ang_vel(self, commands, ang_vel):
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self.tracking_sigma)
```

**2. Energy Cost (lines 459-463)**
```python
def _cost_energy(self, qvel, qfrc_actuator):
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))  # Power = |velocity| * |force|
```

**3. Foot Slip Cost (lines 498-505)**
```python
def _cost_feet_slip(self, data, contact, info):
    cmd_norm = jp.linalg.norm(info["command"])
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    return jp.sum(vel_xy_norm_sq * contact) * (cmd_norm > 0.01)  # Only when moving
```

## 🔧 How to Modify for Research

### 1. Custom Reward Function

Create your own reward function by modifying `_get_reward()`:

```python
# Add custom reward terms
def _get_reward(self, ...):
    rewards = {
        # Existing rewards...
        "tracking_lin_vel": self._reward_tracking_lin_vel(...),
        
        # Your custom rewards
        "custom_efficiency": self._reward_custom_efficiency(...),
        "custom_stability": self._reward_custom_stability(...),
        "custom_gait": self._reward_custom_gait(...),
    }
    return rewards

# Add custom reward implementations
def _reward_custom_efficiency(self, data, action, info):
    # Example: Reward efficient gaits
    power = jp.sum(jp.abs(data.qvel[6:]) * jp.abs(data.actuator_force))
    velocity = jp.linalg.norm(self.get_local_linvel(data)[:2])
    efficiency = velocity / (power + 1e-6)  # Avoid division by zero
    return efficiency

def _reward_custom_stability(self, data):
    # Example: Reward stable base orientation
    upvector = self.get_upvector(data)
    return jp.exp(-jp.sum(jp.square(upvector[:2])))

def _reward_custom_gait(self, contact, info):
    # Example: Reward specific gait patterns
    # contact is [FR, FL, RR, RL] boolean array
    # Reward diagonal gaits (trot)
    diagonal1 = contact[0] * contact[3]  # FR + RL
    diagonal2 = contact[1] * contact[2]  # FL + RR
    return diagonal1 + diagonal2
```

### 2. Custom Observation Space

Modify `_get_obs()` to add custom observations:

```python
def _get_obs(self, data, info):
    # Existing observations...
    state = jp.hstack([
        noisy_linvel,
        noisy_gyro,
        # ... existing observations
        
        # Add custom observations
        self._get_custom_obs(data, info),  # Your custom observations
    ])
    
def _get_custom_obs(self, data, info):
    # Example: Add terrain information, gait phase, etc.
    terrain_height = self._get_terrain_height(data)
    gait_phase = self._get_gait_phase(info)
    return jp.hstack([terrain_height, gait_phase])
```

### 3. Custom Environment Configuration

```python
def custom_config():
    config = default_config()
    
    # Modify existing parameters
    config.reward_config.scales.tracking_lin_vel = 2.0  # Increase tracking reward
    config.reward_config.scales.energy = -0.01         # Increase energy penalty
    
    # Add custom parameters
    config.custom_params = config_dict.create(
        efficiency_weight=1.0,
        stability_weight=0.5,
        gait_weight=0.2,
    )
    
    return config
```

### 4. Domain Randomization

Modify `randomize.py` to add custom randomization:

```python
def custom_randomize(model, rng):
    # Existing randomization...
    
    # Add custom randomization
    # Randomize friction
    rng, key = jax.random.split(rng)
    friction_multiplier = jax.random.uniform(key, minval=0.5, maxval=1.5)
    model = model.replace(geom_friction=model.geom_friction * friction_multiplier)
    
    # Randomize mass
    rng, key = jax.random.split(rng)
    mass_multiplier = jax.random.uniform(key, minval=0.8, maxval=1.2)
    model = model.replace(body_mass=model.body_mass * mass_multiplier)
    
    return model
```

## 🎯 Research Directions

### 1. Energy Efficiency
- Modify energy cost function
- Add metabolic cost models
- Reward specific gait patterns

### 2. Terrain Adaptation
- Add terrain observations
- Modify XML files for different terrains
- Add terrain-aware rewards

### 3. Gait Analysis
- Track contact patterns
- Reward specific gaits (trot, pace, bound)
- Add gait transition rewards

### 4. Robustness
- Increase domain randomization
- Add external disturbances
- Test on rough terrain

## 📊 Training Parameters

Recommended training configurations (from `locomotion_params.py`):

```python
# High-level training config
ppo_config = {
    'num_timesteps': 100_000_000,    # 100M steps
    'num_envs': 2048,                # Parallel environments
    'batch_size': 1024,              # Training batch size
    'learning_rate': 3e-4,           # Learning rate
    'episode_length': 1000,          # Max episode length
}
```

## 🔬 Debugging Tools

### 1. Reward Analysis
```python
# Log individual reward components
for k, v in rewards.items():
    state.metrics[f"reward/{k}"] = v
```

### 2. Observation Analysis
```python
# Monitor key observations
state.metrics["cmd_x"] = info["command"][0]
state.metrics["vel_x"] = self.get_local_linvel(data)[0]
state.metrics["tracking_error"] = jp.abs(info["command"][0] - self.get_local_linvel(data)[0])
```

### 3. Contact Analysis
```python
# Monitor foot contacts
state.metrics["contact_fr"] = contact[0]
state.metrics["contact_fl"] = contact[1]
state.metrics["air_time_avg"] = jp.mean(info["feet_air_time"])
```

This environment provides a solid foundation for locomotion research with well-designed reward functions, domain randomization, and modular architecture for easy modification.