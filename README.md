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
# Train Go1 joystick policy (checkpointing enabled by default)
python main.py train Go1JoystickFlatTerrain --timesteps 100000000

# Train without checkpointing
python main.py train Go1JoystickFlatTerrain --timesteps 100000000 --checkpoint-off

# Resume from checkpoint
python main.py train Go1JoystickFlatTerrain --restore-from checkpoints/Go1JoystickFlatTerrain_20241230_143000/50000000

# Train with custom config
python main.py create-config Go1JoystickFlatTerrain config.json
python main.py train Go1JoystickFlatTerrain --config config.json
```

### 📊 Training Progress Monitoring

Training automatically saves **real-time progress** to organized directories:

#### Reward Graphs & Data
- **Directory**: `reward_graphs/{env_name}_{timestamp}/`
- **Files**:
  - `training_progress_current.png` - Real-time reward graph (updates every evaluation)
  - `training_progress.json` - Complete training data including:
    - Training configuration
    - Reward progression (steps, rewards, standard deviations)
    - Current metrics and progress percentage
    - Timing information
    - Full evaluation metrics

#### Checkpoints
- **Directory**: `checkpoints/{env_name}_{timestamp}/` (matches reward graph timestamp)
- **Files**: 
  - `{step_number}/` directories containing model parameters
  - `best/` directory with the best-performing checkpoint
  - `best_info.json` tracking best reward and step number
- **Frequency**: Every evaluation interval (synced with reward graphs)
- **Best tracking**: Automatically saves best-performing model
- **Control**: Use `--checkpoint-off` to disable

#### Example Directory Structure
```
reward_graphs/Go1JoystickFlatTerrain_20241230_143000/
├── training_progress_current.png    # Real-time reward plot
└── training_progress.json           # Complete training data

checkpoints/Go1JoystickFlatTerrain_20241230_143000/
├── 10000000/                        # Checkpoint at 10M steps
├── 20000000/                        # Checkpoint at 20M steps
├── 30000000/                        # Checkpoint at 30M steps
├── best/                            # Best-performing checkpoint
└── best_info.json                   # Best model metadata
```

#### Monitoring Your Training
1. **View real-time progress**: Open `training_progress_current.png` in any image viewer
2. **Analyze detailed data**: Check `training_progress.json` for comprehensive metrics
3. **Track best performance**: Check `best_info.json` for best reward achieved
4. **Resume training**: Use any checkpoint directory with `--restore-from`
5. **Use best model**: Load from `checkpoints/{env_name}_{timestamp}/best/` for evaluation

### Evaluate a Policy
```bash
# Set environment for headless video generation (SSH/Docker environments)
export MUJOCO_GL=osmesa

# Basic evaluation with video
python main.py evaluate Go1JoystickFlatTerrain checkpoints/Go1JoystickFlatTerrain/100000000 \
    --x-vel 1.0 --y-vel 0.0 --yaw-vel 0.0 --episodes 10

# Evaluate with custom camera and video settings
python main.py evaluate Go1JoystickFlatTerrain checkpoints/Go1JoystickFlatTerrain/100000000 \
    --camera side --width 1920 --height 1080 --save-video

# Evaluate best checkpoint from training
python main.py evaluate Go1JoystickFlatTerrain checkpoints/Go1JoystickFlatTerrain_20241230_143000/best \
    --x-vel 1.5 --episodes 5 --camera track
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
# Basic training (checkpointing enabled by default)
python main.py train Go1JoystickFlatTerrain

# Training with custom parameters
python main.py train Go1JoystickFlatTerrain \
    --timesteps 200000000 \
    --seed 42

# Training without checkpoints (for quick experiments)
python main.py train Go1JoystickFlatTerrain \
    --timesteps 1000000 --checkpoint-off

# Resume training from checkpoint
python main.py train Go1JoystickFlatTerrain \
    --restore-from checkpoints/Go1JoystickFlatTerrain_20241230_143000/50000000

# Train bipedal humanoid
python main.py train BerkeleyHumanoidJoystickFlatTerrain \
    --timesteps 200000000
```

### Evaluation Examples

```bash
# Basic evaluation
python main.py evaluate Go1JoystickFlatTerrain checkpoints/Go1JoystickFlatTerrain/100000000

# Evaluation with specific commands
python main.py evaluate Go1JoystickFlatTerrain checkpoints/Go1JoystickFlatTerrain/100000000 \
    --x-vel 1.5 --y-vel 0.5 --yaw-vel 1.0 \
    --episodes 20 --video-path custom_videos

# Command sequence evaluation with video
python main.py command-sequence Go1JoystickFlatTerrain checkpoints/Go1JoystickFlatTerrain/100000000 \
    --commands "0,0,0;1,0,0;1,0,1;0,1,0;-1,0,0" \
    --steps-per-command 300 --show-video

# Complex command sequence with custom video settings
python main.py command-sequence Go1JoystickFlatTerrain checkpoints/Go1JoystickFlatTerrain/100000000 \
    --commands "0,0,0;1.5,0,0;1.5,0.5,1.0;0,1,0;-1,0,0;0,0,3.14" \
    --steps-per-command 200 --camera track --width 1280 --height 720
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

- **num_timesteps**: Total training steps (default: 200M for Go1, proven from original notebook)
- **num_envs**: Parallel environments (default: 8192)
- **learning_rate**: PPO learning rate (default: 3e-4)
- **batch_size**: Training batch size (default: 256)
- **episode_length**: Max episode length (default: 1000)

## Evaluation Metrics

The evaluation system provides comprehensive metrics:

- **Tracking errors**: How well the robot follows velocity commands
- **Power consumption**: Energy efficiency analysis
- **Gait analysis**: Foot swing patterns and contact states
- **Reward breakdown**: Individual reward component analysis
- **Action smoothness**: Policy stability metrics

## Video Generation

### Features (Matching Jupyter Notebook)
Evaluation automatically generates videos showing:
- Robot locomotion with command visualization
- Contact point indicators  
- Perturbation force visualization
- Joystick command overlays
- **Interactive video display** (like `media.show_video()` in notebook)
- **Command sequence videos** showing policy following different commands
- **Multiple camera angles** (track, side, front)
- **Custom video resolution** and quality

### Video Options
```bash
# Save video only (default)
--save-video --no-show-video

# Display video interactively (like notebook)
--show-video

# Both save and display
--save-video --show-video

# Custom camera angles
--camera track    # Follow robot (default)
--camera side     # Side view
--camera front    # Front view

# Custom resolution
--width 1920 --height 1080    # Full HD
--width 1280 --height 720     # HD
--width 640 --height 480      # Standard (default)
```

### Headless Video Generation

For SSH/headless environments (like cloud servers or Docker), video generation requires specific setup:

#### Prerequisites
Graphics libraries are automatically installed by `setup.sh` if you have sudo access. 

**With sudo access:**
```bash
sudo apt update
sudo apt install -y libegl1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg
```

**Without sudo access:**
- Use Google Colab (has graphics libraries pre-installed)
- Use Docker containers with graphics libraries
- Ask system administrator to install the libraries
- Run evaluation without video: `--no-video`

#### Check Video Generation Support
```bash
# Check if graphics libraries are available
ldconfig -p | grep -E "libegl|libGL|libOSMesa"

# If libraries are found, you can generate videos
# If not, you'll need to install them or use alternatives
```

#### Usage in Headless Environment
```bash
# Set MuJoCo to use OSMesa backend for headless rendering
export MUJOCO_GL=osmesa

# Evaluate with video generation
python main.py evaluate Go1JoystickFlatTerrain checkpoints/Go1JoystickFlatTerrain/best --episodes 5

# The video will be saved to videos/ directory
ls videos/

# If video generation fails, run without video
python main.py evaluate Go1JoystickFlatTerrain checkpoints/Go1JoystickFlatTerrain/best --episodes 5 --no-video
```

#### Troubleshooting Video Generation
1. **"No OpenGL context"**: Ensure graphics libraries are installed
2. **"MUJOCO_GL not set"**: Export `MUJOCO_GL=osmesa`
3. **Empty videos directory**: Check evaluation logs for errors
4. **Video quality issues**: Adjust `--width` and `--height` parameters

#### Environment Variables
```bash
# For headless systems (SSH, Docker, etc.)
export MUJOCO_GL=osmesa

# For systems with display (optional)
export MUJOCO_GL=glfw  # or leave unset
```

### Command Sequence Videos
Create videos showing the robot following different velocity commands:
```bash
# Show robot adapting to different commands
python main.py command-sequence Go1JoystickFlatTerrain checkpoint.pkl \
    --commands "0,0,0;1,0,0;0,1,0;0,0,1.57;-1,0,0" \
    --show-video --camera track
```

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
