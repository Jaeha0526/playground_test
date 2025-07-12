# Navigation Training and Evaluation Commands

This guide provides all the commands needed to train, evaluate, and generate videos for the hierarchical navigation system.

## Prerequisites

Before running any commands, ensure you have:
1. Activated the virtual environment: `source venv/bin/activate`
2. Set up the environment for video generation: `source navigation_guide/setup_video_env.sh`

## 1. Training Commands

### Quick Training (2-3 minutes)
For a quick test with 30,000 timesteps:
```bash
python train_hierarchical_binary_success.py
```
*Note: The script automatically uses the default locomotion checkpoint*

### Standard Training (3-4 minutes)
For standard training with 300,000 timesteps (achieves 100% success):
```bash
python train_hierarchical_binary_success.py --num_timesteps 300000
```

### Full Training (30+ minutes)
For extensive training with 5 million timesteps:
```bash
python train_hierarchical_binary_success.py --num_timesteps 5000000
```

### Custom Locomotion Checkpoint
If you have a different locomotion checkpoint:
```bash
python train_hierarchical_binary_success.py \
    --locomotion_checkpoint path/to/your/checkpoint \
    --num_timesteps 300000
```

## 2. Evaluation Commands

### Basic Evaluation
Evaluate the trained navigation model:
```bash
python evaluate_navigation_final.py
```

### Test with Close Goals
Test navigation with closer goal distances:
```bash
python test_navigation_close_goals.py
```

## 3. Video Generation Commands

### Basic Video Generation
Generate a simple navigation video:
```bash
source setup_cuda12_env.sh
export MUJOCO_GL=osmesa
python generate_navigation_video_osmesa.py
```

### Video with Information Overlay
Generate a video with goal position and distance information:
```bash
source setup_cuda12_env.sh
export MUJOCO_GL=osmesa
python generate_navigation_video_with_info.py
```

## 4. Important Setup Steps

### Initial Setup (One-time)
Run the setup script to install all dependencies:
```bash
source navigation_guide/setup_video_env.sh
```

### Before Each Video Generation Session
Always run these commands before generating videos:
```bash
# Set up CUDA 12 environment
source setup_cuda12_env.sh

# Set headless rendering backend
export MUJOCO_GL=osmesa
```

## 5. Output Locations

- **Checkpoints**: `checkpoints/navigation_binary_YYYYMMDD_HHMMSS/`
- **Training plots**: `plots/navigation_binary_YYYYMMDD_HHMMSS/`
- **Videos**: `videos/navigation_*.mp4`
- **Evaluation results**: `videos/navigation_evaluation_*.json`

## 6. Common Parameters

### Training Parameters
- `--num_timesteps`: Number of training steps (default: 300,000)
- `--num_envs`: Number of parallel environments (default: 512)
- `--learning_rate`: Learning rate (default: 5e-4)
- `--seed`: Random seed (default: 0)

### Environment Configuration
- Room size: 10m x 10m
- Goal radius: 0.5m
- Goal distance range: 2-8m (configurable)
- Episode length: 500 steps

## 7. Troubleshooting

If video generation fails:
1. Ensure you've run `source setup_cuda12_env.sh`
2. Check that `MUJOCO_GL=osmesa` is set
3. Verify libraries are installed: `ldconfig -p | grep osmesa`
4. If still failing, check the video generation issues document

## 8. Example Full Workflow

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Set up video generation environment (first time only)
source navigation_guide/setup_video_env.sh

# 3. Train navigation model
python train_hierarchical_binary_success.py \
    --locomotion_checkpoint checkpoints/Go1JoystickFlatTerrain_20250630_224046/best \
    --num_timesteps 300000

# 4. Set up for video generation
source setup_cuda12_env.sh
export MUJOCO_GL=osmesa

# 5. Generate video
python generate_navigation_video_with_info.py

# 6. Check results
ls -la videos/navigation_*.mp4
```