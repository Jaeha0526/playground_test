# Go1 Navigation Training

A hierarchical reinforcement learning system for training the Unitree Go1 quadruped robot to perform goal-conditioned navigation in indoor environments.

## 🎯 What is this Experiment?

This project implements **hierarchical navigation** for the Go1 robot by building a small navigation layer on top of an existing locomotion policy. Instead of training navigation from scratch, we leverage the robot's proven walking skills and only learn high-level navigation decisions.

### Key Innovation
- **Frozen Locomotion:** Keep the pre-trained Go1 locomotion policy (200M steps of training) completely frozen
- **Navigation Layer:** Train a small network (6.7K parameters) to output velocity commands `[vx, vy, vyaw]`  
- **Curriculum Learning:** Progressive training through 4 stages of increasing difficulty
- **Goal Conditioning:** Robot learns to navigate to randomly placed goals in a room

### Architecture Overview
```
Goal Position + Robot State → [Navigation Network] → Velocity Commands
                                     ↓
Proprioceptive Sensors ← [Frozen Go1 Policy] ← Velocity Commands  
                                     ↓
                              Joint Actions (12-dim)
```

## 🏗️ System Components

### Environment
- **Room:** 10m × 10m square room with walls
- **Goals:** Randomly placed target positions
- **Observations:** Goal direction, distance, robot pose, wall distances (10-dim)
- **Rewards:** Goal reaching (+100), progress (+2.0), wall collision (-10.0), time penalty (-0.1)

### Curriculum Stages
1. **Stage 1 (Close Goals):** 1-3m goals, 6m room, 250 steps/episode
2. **Stage 2 (Medium Goals):** 2-5m goals, 8m room, 400 steps/episode  
3. **Stage 3 (Full Room):** 3-8m goals, 10m room + obstacles, 600 steps/episode
4. **Stage 4 (Complex):** 5-12m goals, 10m room + multiple obstacles, 1000 steps/episode

### Training Schedule
- **Total:** 50M steps across 4 stages (6-10 hours)
- **Stage 1:** 10M steps (1-2 hours)
- **Stage 2:** 15M steps (2-3 hours)
- **Stage 3:** 15M steps (2-3 hours)  
- **Stage 4:** 10M steps (1-2 hours)

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure you have a trained Go1 locomotion checkpoint
ls checkpoints/Go1JoystickFlatTerrain_*/best/

# Install dependencies (should already be available)
# - JAX/Flax for neural networks
# - Brax for RL training  
# - MuJoCo Playground for Go1 environment
```

### Run Training

#### Option 1: Full Curriculum Training (Recommended)
```bash
# Start from Stage 1 and progress automatically
python -m src.locomotion_training.training.navigation_trainer \
    --locomotion_checkpoint checkpoints/Go1JoystickFlatTerrain_20250630_224046/best/ \
    --stage 1 \
    --checkpoint_dir checkpoints/navigation_full \
    --total_timesteps 50000000
```

#### Option 2: Test Single Stage
```bash
# Train only Stage 1 for testing
python -m src.locomotion_training.training.navigation_trainer \
    --locomotion_checkpoint checkpoints/Go1JoystickFlatTerrain_20250630_224046/best/ \
    --stage 1 \
    --checkpoint_dir checkpoints/navigation_test \
    --total_timesteps 10000000
```

#### Option 3: Resume from Checkpoint
```bash
# Resume interrupted training
python -m src.locomotion_training.training.navigation_trainer \
    --checkpoint_dir checkpoints/navigation_full \
    --resume True
```

### Quick Test (No Training)
```bash
# Test the complete system setup
python test_navigation_system.py
```

## 📊 Monitoring Training

### Real-time Progress
Monitor training progress through console output:
```
Stage 1 Progress: Close Goals
   Episodes: 1,250
   Training Steps: 2,500,000 / 10,000,000
   Step Progress: 25.0%
   Time in Stage: 0.8 hours
   Success Rate: 78.5%
   Time Efficiency: 72.3%
   Collision Rate: 8.2%
   🎯 READY FOR ADVANCEMENT!
```

### Curriculum Advancement
The system automatically advances stages when criteria are met:
- **Success Rate:** Stage 1: >80%, Stage 2: >70%, Stage 3: >60%, Stage 4: >50%
- **Time Efficiency:** Robot reaches goals in <70% of optimal time
- **Stability:** Consistent performance over 50+ episodes

### Key Metrics
- **Success Rate:** Percentage of episodes where robot reaches goal
- **Time Efficiency:** Actual time / optimal time to reach goal
- **Path Efficiency:** Actual path length / optimal path length  
- **Collision Rate:** Percentage of episodes with wall collisions

## 📁 File Structure

```
src/locomotion_training/
├── envs/
│   ├── go1_navigation.py              # Navigation environment
│   └── go1_room_environment.xml       # Room with walls XML
├── networks/
│   ├── navigation_network.py          # Small navigation network (6.7K params)
│   └── hierarchical_policy.py         # Combined navigation + locomotion
├── training/
│   ├── navigation_trainer.py          # Main training orchestration
│   └── curriculum_manager.py          # Automatic stage progression
└── config/
    └── navigation_config.py           # Training configurations

docs/
└── navigation_architecture_design.md  # Detailed system design

test_navigation_system.py              # Quick system test
```

## 🎛️ Configuration

### Modify Training Settings
Edit `src/locomotion_training/config/navigation_config.py`:

```python
# Adjust Stage 1 difficulty
stage_configs[1] = {
    "num_timesteps": 20_000_000,      # Longer training
    "env_config": {
        "max_goal_distance": 2.0,     # Easier goals
        "episode_length": 300,        # Longer episodes
    }
}
```

### Custom Reward Weights
```python
reward_config = {
    "goal_reached": 200.0,        # Increase goal reward
    "goal_progress": 1.0,         # Reduce progress reward
    "wall_collision": -20.0,      # Stronger collision penalty
}
```

### Environment Variations
```python
env_config = {
    "room_size": 8.0,             # Smaller room
    "wall_buffer": 1.5,           # More wall clearance
    "goal_radius": 0.3,           # Tighter success criteria
}
```

## 📈 Expected Results

### Stage 1 (Close Goals)
- **Target Success Rate:** >80%
- **Typical Training:** 8-12M steps to advancement
- **Behaviors Learned:** Basic goal approach, wall avoidance

### Stage 2 (Medium Goals)  
- **Target Success Rate:** >70%
- **Typical Training:** 12-18M steps to advancement
- **Behaviors Learned:** Path planning, efficient navigation

### Stage 3 (Full Room + Obstacles)
- **Target Success Rate:** >60%
- **Typical Training:** 10-20M steps to advancement  
- **Behaviors Learned:** Obstacle avoidance, complex navigation

### Stage 4 (Complex Environments)
- **Target Success Rate:** >50%
- **Final Capability:** Navigate complex environments with multiple obstacles

## 🐛 Troubleshooting

### Common Issues

#### 1. Locomotion Checkpoint Not Found
```bash
Error: Checkpoint not found: checkpoints/Go1JoystickFlatTerrain_*/best/
```
**Solution:** Ensure you have a trained Go1 locomotion checkpoint. Update the path in the training command.

#### 2. Low Success Rate in Stage 1
```bash
Success Rate: 30% (Expected: >80%)
```
**Solutions:**
- Increase goal reward: `goal_reached: 200.0`
- Reduce goal distance: `max_goal_distance: 2.0`
- Extend episode length: `episode_length: 300`

#### 3. Training Stalls
```bash
Stage metrics haven't improved for 1M steps
```
**Solutions:**
- Lower learning rate: `learning_rate: 5e-4`
- Increase exploration: `entropy_cost: 2e-2`
- Reset to earlier checkpoint

#### 4. Wall Collision Issues
```bash
Collision Rate: >20% (Expected: <10%)
```
**Solutions:**
- Increase collision penalty: `wall_collision: -20.0`
- Add boundary warning: `boundary_penalty: -1.0`
- Increase wall buffer: `wall_buffer: 1.5`

### Performance Optimization

#### GPU Memory Issues
```python
# Reduce parallel environments
config.num_envs = 2048  # Instead of 4096

# Reduce batch size  
config.batch_size = 128  # Instead of 256
```

#### Speed Up Training
```python
# Increase learning rate for navigation
config.learning_rate = 2e-3  # Instead of 1e-3

# Reduce evaluation frequency
config.eval_frequency = 200_000  # Instead of 100_000
```

## 🔬 Research Extensions

### Advanced Features
1. **Dynamic Obstacles:** Add moving obstacles to environments
2. **Multi-Room Navigation:** Connect multiple rooms with doorways
3. **Semantic Goals:** Navigate to object categories ("go to chair")
4. **Outdoor Terrain:** Extend to uneven outdoor environments

### Architecture Improvements
1. **Vision Integration:** Add camera inputs for obstacle detection
2. **Memory Networks:** LSTM/Transformer for long-term spatial memory
3. **Hierarchical Planning:** Multi-level goal decomposition
4. **Transfer Learning:** Apply to other quadruped robots

### Evaluation Benchmarks
1. **Generalization Tests:** Unseen room sizes and layouts
2. **Robustness:** Performance under noise and perturbations  
3. **Sample Efficiency:** Compare vs end-to-end training
4. **Real Robot:** Deploy on physical Go1 hardware

## 📚 References

- [Go1 Environment Guide](docs/go1_environment_guide.md) - Detailed locomotion system analysis
- [Architecture Design](docs/navigation_architecture_design.md) - Complete system design document
- [MuJoCo Playground](https://github.com/deepmind/mujoco_playground) - Base environment framework
- [Brax](https://github.com/google/brax) - JAX-based RL training framework

## 🤝 Contributing

### Adding New Curriculum Stages
1. Define stage in `navigation_config.py`
2. Update advancement criteria in `curriculum_manager.py`
3. Test with `test_navigation_system.py`

### Custom Environments
1. Inherit from `Go1NavigationEnv`
2. Override `_get_reward()` for custom rewards
3. Modify XML for different layouts

### Bug Reports
Please include:
- Training configuration used
- Console output with error
- Hardware specifications (GPU, RAM)
- JAX/Brax versions

---

**Happy Navigation Training!** 🤖🎯

For questions or issues, please refer to the troubleshooting section or check the detailed architecture documentation.