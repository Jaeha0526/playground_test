# Go1 Structure and Customization Guide

This document answers key questions about the Go1 locomotion system structure and how to customize it for research.

## 🏗️ Question 1: General Structure of Go1 Folder

### **File Organization and Responsibilities**

```
mujoco_playground/_src/locomotion/go1/
├── __init__.py                 # Package initialization
├── README.md                   # Go1-specific documentation
├── base.py                     # 🏛️ Base environment class
├── joystick.py                 # 🎮 Joystick task implementation
├── handstand.py               # 🤸 Handstand task implementation  
├── getup.py                   # 🏃 Recovery task implementation
├── go1_constants.py           # 📋 Constants and paths
├── randomize.py               # 🎲 Domain randomization
└── xmls/                      # 🤖 MuJoCo model files
    ├── go1_mjx_feetonly.xml   # Robot model (feet-only collision)
    ├── scene_mjx_feetonly_flat_terrain.xml  # Flat terrain scene
    ├── scene_mjx_feetonly_rough_terrain.xml # Rough terrain scene
    └── assets/                # Textures and heightmaps
```

### **Where Each Component is Defined:**

#### **1. Environment Definition** 🏛️
**File:** `base.py` (lines 40-100+)
```python
class Go1Env(mjx_env.MjxEnv):
    """Base class for Go1 environments."""
    
    def __init__(self, xml_path, config, config_overrides=None):
        # Load MuJoCo model from XML
        self._mj_model = mujoco.MjModel.from_xml_string(...)
        # Set PD gains, timesteps, etc.
```

**Responsibilities:**
- Load MuJoCo model from XML files
- Set up physics parameters (PD gains, timesteps)
- Define sensor reading methods
- Provide base functionality for all Go1 tasks

#### **2. Observation & Action Interface** 🎮
**File:** `joystick.py` (lines 295-377 for obs, 234-289 for actions)

**Observations:**
```python
def _get_obs(self, data, info):
    state = jp.hstack([
        noisy_linvel,        # 3 - Local linear velocity
        noisy_gyro,          # 3 - Angular velocity  
        noisy_gravity,       # 3 - Gravity vector
        noisy_joint_angles,  # 12 - Joint positions
        noisy_joint_vel,     # 12 - Joint velocities
        last_act,           # 12 - Previous action
        command,            # 3 - Velocity command
    ])  # Total: 48 dimensions
```

**Actions:**
```python
def step(self, state, action):
    # action is 12D joint angle targets
    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(self.mjx_model, state.data, motor_targets, ...)
```

#### **3. Policy Model** 🧠
**File:** `mujoco_playground/config/locomotion_params.py` (lines 42-47)
```python
network_factory=config_dict.create(
    policy_hidden_layer_sizes=(128, 128, 128, 128),     # 4-layer policy network
    value_hidden_layer_sizes=(256, 256, 256, 256, 256), # 5-layer value network
    policy_obs_key="state",                              # Uses 48D state
    value_obs_key="state",
)
```

**Note:** The actual neural network implementation is in Brax:
- **Policy Network:** 4 layers × 128 units (actor)
- **Value Network:** 5 layers × 256 units (critic)
- **Input:** 48D observation vector
- **Output:** 12D action vector (joint targets)

#### **4. Training Code** 🏋️
**Files:** 
- `locomotion_params.py` - Training hyperparameters
- Our `src/locomotion_training/training/trainer.py` - Training orchestration

**Training Configuration:**
```python
def brax_ppo_config(env_name):
    return config_dict.create(
        num_timesteps=100_000_000,    # 100M training steps
        num_envs=8192,                # Parallel environments
        batch_size=256,               # Training batch size
        learning_rate=3e-4,           # Learning rate
        unroll_length=20,             # Trajectory length
        num_minibatches=32,           # Mini-batches per update
        discounting=0.97,             # Discount factor
    )
```

#### **5. Reward Function** 🎯
**File:** `joystick.py` (lines 379-533)
```python
def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
    return {
        # Tracking rewards
        "tracking_lin_vel": self._reward_tracking_lin_vel(...),
        "tracking_ang_vel": self._reward_tracking_ang_vel(...),
        
        # Stability costs  
        "orientation": self._cost_orientation(...),
        "lin_vel_z": self._cost_lin_vel_z(...),
        
        # Energy costs
        "torques": self._cost_torques(...),
        "energy": self._cost_energy(...),
        
        # Gait costs
        "feet_slip": self._cost_feet_slip(...),
        "feet_air_time": self._reward_feet_air_time(...),
        
        # ... 17 total reward components
    }
```

---

## 🚧 Question 2: Adding Obstacles for Avoidance Training

### **Where to Modify for Obstacles:**

#### **Option 1: Modify XML Scene Files** (Recommended)
**File:** `xmls/scene_mjx_feetonly_flat_terrain.xml`

**Current terrain (lines 22-25):**
```xml
<worldbody>
  <geom name="floor" size="0 0 0.01" type="plane" material="groundplane" 
        contype="1" conaffinity="0" priority="1" friction="0.6" condim="3"/>
</worldbody>
```

**Add obstacles:**
```xml
<worldbody>
  <!-- Existing floor -->
  <geom name="floor" size="0 0 0.01" type="plane" material="groundplane" 
        contype="1" conaffinity="0" priority="1" friction="0.6" condim="3"/>
        
  <!-- Static obstacles -->
  <body name="obstacle1" pos="2.0 0.0 0.1">
    <geom name="obs1" type="box" size="0.2 0.2 0.1" rgba="1 0 0 1"
          contype="1" conaffinity="1" friction="0.8"/>
  </body>
  
  <body name="obstacle2" pos="4.0 1.0 0.15">
    <geom name="obs2" type="cylinder" size="0.15 0.15" rgba="0 1 0 1"
          contype="1" conaffinity="1" friction="0.8"/>
  </body>
  
  <!-- Dynamic obstacles (can move) -->
  <body name="moving_obstacle" pos="6.0 0.0 0.1">
    <joint name="obs_slide" type="slide" axis="0 1 0" range="-2 2"/>
    <geom name="obs_moving" type="sphere" size="0.2" rgba="0 0 1 1"
          contype="1" conaffinity="1" friction="0.8"/>
  </body>
</worldbody>
```

#### **Option 2: Procedural Obstacle Generation**
**File:** Create custom environment class

```python
class Go1ObstacleAvoidance(go1_base.Go1Env):
    def __init__(self, task="flat_terrain_obstacles", ...):
        # Load base XML and add obstacles programmatically
        super().__init__(xml_path=..., config=...)
        self._add_random_obstacles()
    
    def _add_random_obstacles(self):
        # Add obstacles to the MuJoCo model programmatically
        obstacle_positions = self._generate_obstacle_positions()
        for i, pos in enumerate(obstacle_positions):
            self._add_obstacle(f"obstacle_{i}", pos)
    
    def reset(self, rng):
        # Randomize obstacle positions each episode
        state = super().reset(rng)
        state = self._randomize_obstacles(state, rng)
        return state
```

#### **Option 3: Create New Scene File**
**File:** `xmls/scene_mjx_feetonly_obstacles.xml`

```xml
<mujoco model="go1 feetonly obstacle course">
  <include file="go1_mjx_feetonly.xml"/>
  
  <!-- Add obstacle course layout -->
  <worldbody>
    <geom name="floor" size="0 0 0.01" type="plane" material="groundplane"/>
    
    <!-- Obstacle course pattern -->
    <body name="wall1" pos="3.0 0.0 0.2">
      <geom type="box" size="0.1 2.0 0.2" rgba="0.8 0.4 0.2 1"/>
    </body>
    
    <!-- Maze-like obstacles -->
    <body name="maze_wall1" pos="5.0 -1.0 0.15">
      <geom type="box" size="2.0 0.1 0.15" rgba="0.6 0.6 0.6 1"/>
    </body>
  </worldbody>
</mujoco>
```

### **Adding Obstacle Detection to Observations**

**File:** `joystick.py` - Modify `_get_obs()` method

```python
def _get_obs(self, data, info):
    # Existing observations...
    state = jp.hstack([
        noisy_linvel, noisy_gyro, noisy_gravity,
        noisy_joint_angles, noisy_joint_vel,
        last_act, command,
        
        # Add obstacle information
        self._get_obstacle_observations(data),  # Distance/direction to obstacles
    ])

def _get_obstacle_observations(self, data):
    robot_pos = data.qpos[:2]  # x, y position
    
    # Get distances to known obstacles
    obstacle_distances = []
    for obs_name in ["obstacle1", "obstacle2", "moving_obstacle"]:
        obs_body_id = self.mj_model.body(obs_name).id
        obs_pos = data.xpos[obs_body_id][:2]
        distance = jp.linalg.norm(robot_pos - obs_pos)
        direction = (obs_pos - robot_pos) / (distance + 1e-6)
        obstacle_distances.extend([distance, direction[0], direction[1]])
    
    return jp.array(obstacle_distances)
```

---

## 🎨 Question 3: Custom Reward Functions

### **Where to Change for Custom Rewards:**

#### **Primary Location:** `joystick.py` - `_get_reward()` method (lines 379-417)

**Step 1: Add Custom Reward to Configuration**
```python
def default_config():
    return config_dict.create(
        # ... existing config ...
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Existing rewards...
                tracking_lin_vel=1.0,
                energy=-0.001,
                
                # Add your custom rewards
                obstacle_avoidance=2.0,        # Reward avoiding obstacles
                forward_progress=0.5,          # Reward forward movement
                energy_efficiency=1.0,         # Reward efficient movement
                gait_regularity=0.3,          # Reward consistent gait
            ),
            # Custom parameters
            obstacle_safety_distance=1.0,     # Minimum safe distance
            efficiency_baseline=10.0,         # Energy efficiency baseline
        )
    )
```

**Step 2: Implement Custom Reward Functions**
```python
def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
    rewards = {
        # Existing rewards...
        "tracking_lin_vel": self._reward_tracking_lin_vel(...),
        "energy": self._cost_energy(...),
        
        # Your custom rewards
        "obstacle_avoidance": self._reward_obstacle_avoidance(data, info),
        "forward_progress": self._reward_forward_progress(data, info),
        "energy_efficiency": self._reward_energy_efficiency(data, action, info),
        "gait_regularity": self._reward_gait_regularity(contact, info),
    }
    return rewards

# Implement custom reward functions
def _reward_obstacle_avoidance(self, data, info):
    """Reward staying away from obstacles."""
    robot_pos = data.qpos[:2]
    min_distance = float('inf')
    
    for obs_name in ["obstacle1", "obstacle2"]:
        obs_body_id = self.mj_model.body(obs_name).id
        obs_pos = data.xpos[obs_body_id][:2]
        distance = jp.linalg.norm(robot_pos - obs_pos)
        min_distance = jp.minimum(min_distance, distance)
    
    safety_distance = self._config.reward_config.obstacle_safety_distance
    # Exponential reward for maintaining safe distance
    return jp.exp(-(safety_distance - min_distance) ** 2)

def _reward_forward_progress(self, data, info):
    """Reward forward movement in the desired direction."""
    cmd_direction = info["command"][:2]  # x, y velocity commands
    actual_velocity = self.get_local_linvel(data)[:2]
    
    # Dot product rewards movement in commanded direction
    progress = jp.dot(cmd_direction, actual_velocity)
    return jp.maximum(progress, 0.0)  # Only positive progress

def _reward_energy_efficiency(self, data, action, info):
    """Reward energy-efficient movement."""
    power = jp.sum(jp.abs(data.qvel[6:]) * jp.abs(data.actuator_force))
    velocity = jp.linalg.norm(self.get_local_linvel(data)[:2])
    
    # Efficiency = distance / energy
    efficiency = velocity / (power + 1e-6)
    baseline = self._config.reward_config.efficiency_baseline
    return jp.clip(efficiency / baseline, 0.0, 2.0)

def _reward_gait_regularity(self, contact, info):
    """Reward consistent gait patterns."""
    # Reward alternating contact patterns (trot gait)
    diagonal1 = contact[0] * contact[3]  # FR + RL
    diagonal2 = contact[1] * contact[2]  # FL + RR
    
    # Penalize all feet on ground or no feet on ground
    total_contact = jp.sum(contact)
    ideal_contact = jp.abs(total_contact - 2.0)  # Prefer 2 feet on ground
    
    return (diagonal1 + diagonal2) * jp.exp(-ideal_contact)
```

### **Alternative: Create Custom Environment Class**

```python
# Create: src/locomotion_training/envs/go1_custom.py
from mujoco_playground._src.locomotion.go1 import joystick

class Go1CustomRewards(joystick.Joystick):
    """Go1 with custom reward functions for research."""
    
    def __init__(self, task="flat_terrain", **kwargs):
        super().__init__(task=task, **kwargs)
    
    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        # Get base rewards
        base_rewards = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
        
        # Add your custom rewards
        custom_rewards = {
            "my_custom_reward1": self._my_custom_reward1(data, action, info),
            "my_custom_reward2": self._my_custom_reward2(data, contact),
        }
        
        # Combine rewards
        return {**base_rewards, **custom_rewards}
    
    def _my_custom_reward1(self, data, action, info):
        # Your custom reward implementation
        return 0.0
```

### **Quick Customization Tips:**

1. **Reward Scaling:** Adjust weights in `reward_config.scales`
2. **New Terms:** Add to `_get_reward()` return dictionary
3. **Configuration:** Add parameters to `default_config()`
4. **Testing:** Use `state.metrics[f"reward/{name}"] = value` for logging

This structure allows you to easily experiment with different reward formulations while maintaining the existing training infrastructure!