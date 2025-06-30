# Locomotion Training

> **Note**: This code is converted from the notebook in https://github.com/google-deepmind/mujoco_playground

A Python framework for training and evaluating locomotion policies using MuJoCo Playground. This repository converts the functionality from the original Jupyter notebook into a structured Python codebase with CLI interface.

## Features

- **Multi-environment support**: Train policies for quadrupedal (Go1, Spot, Barkour) and bipedal (Berkeley Humanoid, Unitree G1/H1) robots
- **PPO training**: Robust reinforcement learning with domain randomization
- **Comprehensive evaluation**: Detailed analysis with plotting and video generation
- **Checkpointing**: Save and restore training progress
- **Finetuning**: Improve policies with curriculum learning
- **CLI interface**: Easy-to-use command-line tools

## Installation

1. Clone this repository:
```bash
git clone <repo-url>
cd playground_test
```

2. Run the setup script:
```bash
bash setup.sh
```

3. Activate the virtual environment and verify installation:
```bash
source venv/bin/activate
python main.py list-envs
```

## Quick Start

### List Available Environments
```bash
python main.py list-envs
```

### Train a Policy
```bash
# Train Go1 joystick policy
python main.py train Go1JoystickFlatTerrain --timesteps 100000000 --checkpoint-dir checkpoints

# Train with custom config
python main.py create-config Go1JoystickFlatTerrain config.json
python main.py train Go1JoystickFlatTerrain --config config.json
```

### Evaluate a Policy
```bash
# Evaluate with forward motion
python main.py evaluate Go1JoystickFlatTerrain checkpoints/Go1JoystickFlatTerrain/100000000 \
    --x-vel 1.0 --y-vel 0.0 --yaw-vel 0.0 --episodes 10

# Evaluate with turning
python main.py evaluate Go1JoystickFlatTerrain checkpoints/Go1JoystickFlatTerrain/100000000 \
    --x-vel 0.0 --y-vel 0.0 --yaw-vel 3.14 --episodes 5
```

### Train Handstand Policy
```bash
# Train base handstand policy
python main.py train-handstand --checkpoint-dir checkpoints

# Finetune with energy penalties
python main.py train-handstand --finetune --checkpoint-dir checkpoints
```

## Project Structure

```
playground_test/
├── src/locomotion_training/
│   ├── config/
│   │   └── config.py          # Configuration management
│   ├── training/
│   │   └── trainer.py         # Training logic
│   ├── evaluation/
│   │   └── evaluator.py       # Evaluation and analysis
│   └── utils/
│       ├── plotting.py        # Visualization utilities
│       └── video.py           # Video rendering
├── checkpoints/               # Saved model checkpoints
├── videos/                    # Generated evaluation videos
├── main.py                    # CLI entry point
└── requirements.txt           # Dependencies
```

## Usage Examples

### Training Examples

```bash
# Basic training
python main.py train Go1JoystickFlatTerrain

# Training with custom parameters
python main.py train Go1JoystickFlatTerrain \
    --timesteps 200000000 \
    --checkpoint-dir my_checkpoints \
    --seed 42

# Resume training from checkpoint
python main.py train Go1JoystickFlatTerrain \
    --restore-from checkpoints/Go1JoystickFlatTerrain/50000000

# Train bipedal humanoid
python main.py train BerkeleyHumanoidJoystickFlatTerrain \
    --timesteps 200000000 \
    --checkpoint-dir humanoid_checkpoints
```

### Evaluation Examples

```bash
# Basic evaluation
python main.py evaluate Go1JoystickFlatTerrain checkpoints/Go1JoystickFlatTerrain/100000000

# Evaluation with specific commands
python main.py evaluate Go1JoystickFlatTerrain checkpoints/Go1JoystickFlatTerrain/100000000 \
    --x-vel 1.5 --y-vel 0.5 --yaw-vel 1.0 \
    --episodes 20 --video-path custom_videos

# Command sequence evaluation
python main.py command-sequence Go1JoystickFlatTerrain checkpoints/Go1JoystickFlatTerrain/100000000 \
    --commands "0,0,0;1,0,0;1,0,1;0,1,0;-1,0,0" \
    --steps-per-command 300
```

### Configuration Management

```bash
# Create default training config
python main.py create-config Go1JoystickFlatTerrain train_config.json --type training

# Create default evaluation config
python main.py create-config Go1JoystickFlatTerrain eval_config.json --type eval

# Use custom config
python main.py train Go1JoystickFlatTerrain --config train_config.json
```

## Environment Details

### Quadrupedal Environments
- **Go1JoystickFlatTerrain**: Unitree Go1 with joystick control
- **Go1Handstand**: Unitree Go1 handstand policy
- **SpotJoystick**: Boston Dynamics Spot with joystick control
- **BarkourJoystick**: Google Barkour robot with joystick control

### Bipedal Environments
- **BerkeleyHumanoidJoystickFlatTerrain**: Berkeley Humanoid with joystick control
- **UnitreeG1Joystick**: Unitree G1 humanoid with joystick control
- **UnitreeH1Joystick**: Unitree H1 humanoid with joystick control

## Training Parameters

Key training parameters can be configured:

- **num_timesteps**: Total training steps (default: 100M)
- **num_envs**: Parallel environments (default: 2048)
- **learning_rate**: PPO learning rate (default: 3e-4)
- **batch_size**: Training batch size (default: 1024)
- **episode_length**: Max episode length (default: 1000)

## Evaluation Metrics

The evaluation system provides comprehensive metrics:

- **Tracking errors**: How well the robot follows velocity commands
- **Power consumption**: Energy efficiency analysis
- **Gait analysis**: Foot swing patterns and contact states
- **Reward breakdown**: Individual reward component analysis
- **Action smoothness**: Policy stability metrics

## Video Generation

Evaluation automatically generates videos showing:
- Robot locomotion with command visualization
- Contact point indicators
- Perturbation force visualization
- Joystick command overlays

## Advanced Usage

### Custom Environment Configuration

```python
# Modify environment parameters
custom_config = {
    'energy_termination_threshold': 400,
    'reward_config.energy': -0.003,
    'reward_config.dof_acc': -2.5e-7,
}

# Use in training
trainer.train(custom_env_config=custom_config)
```

### Programmatic API

```python
from src.locomotion_training.training.trainer import LocomotionTrainer
from src.locomotion_training.evaluation.evaluator import LocomotionEvaluator
from src.locomotion_training.config.config import TrainingConfig

# Setup training
config = TrainingConfig(env_name='Go1JoystickFlatTerrain')
trainer = LocomotionTrainer(config)
make_inference_fn, params, metrics = trainer.train()

# Evaluate
eval_config = EvalConfig(env_name='Go1JoystickFlatTerrain')
evaluator = LocomotionEvaluator(eval_config)
results = evaluator.evaluate_policy(make_inference_fn, params)
```

## Troubleshooting

### Common Issues

1. **Setup Script Timeout**: If `bash setup.sh` hangs during verification, the installation is likely complete. Press Ctrl+C and test with:
   ```bash
   source venv/bin/activate
   python main.py list-envs
   ```

2. **GPU Memory Issues**: Reduce `num_envs` or `batch_size`
3. **Training Instability**: Adjust learning rate or use different seed
4. **Evaluation Errors**: Ensure checkpoint path is correct
5. **Video Generation Fails**: Check OpenGL/MuJoCo rendering setup

### GPU Requirements

Training requires GPU acceleration. Make sure you have:
- CUDA-compatible GPU
- Proper JAX installation with GPU support
- Sufficient GPU memory (8GB+ recommended)

## Performance Tips

- Use larger `num_envs` for faster training on high-memory GPUs
- Adjust `render_every` in evaluation to balance quality vs. speed
- Use checkpointing for long training runs
- Monitor training progress with real-time plotting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the original MuJoCo Playground license for details.

## Acknowledgments

- Based on the MuJoCo Playground locomotion notebook: [locomotion.ipynb](https://github.com/google-deepmind/mujoco_playground/blob/main/learning/notebooks/locomotion.ipynb)
- Original repository: [google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground)
- Uses Brax for reinforcement learning
- Built on MuJoCo physics simulation
- Inspired by sim-to-real robotics research
