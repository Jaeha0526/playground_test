# Navigation Training and Video Generation Guide

This folder contains comprehensive documentation and setup scripts for training hierarchical navigation models and generating evaluation videos.

## 📁 Contents

1. **`navigation_commands.md`** - Complete command reference for training, evaluation, and video generation
2. **`video_generation_issues.md`** - Detailed troubleshooting guide documenting all issues encountered and their solutions
3. **`setup_video_env.sh`** - Automated setup script for video generation environment
4. **`generate_video.sh`** - Helper script for quick video generation (created by setup script)

## 🚀 Quick Start

### 1. Initial Setup (One-time)
```bash
# From project root directory
source navigation_guide/setup_video_env.sh
```

### 2. Train Navigation Model
```bash
python train_hierarchical_binary_success.py \
    --locomotion_checkpoint checkpoints/Go1JoystickFlatTerrain_20250630_224046/best \
    --num_timesteps 300000
```

### 3. Generate Video
```bash
source setup_cuda12_env.sh
export MUJOCO_GL=osmesa
python generate_navigation_video_with_info.py
```

## 📋 Key Requirements

- **System Libraries**: libosmesa6-dev, libgl1-mesa-glx, ffmpeg
- **Python Libraries**: opencv-python, mediapy
- **Environment Variables**: MUJOCO_GL=osmesa, CUDA paths
- **GPU**: NVIDIA GPU with CUDA 12 support (or CPU fallback)

## 🔧 Troubleshooting

If you encounter issues:
1. Check `video_generation_issues.md` for known problems and solutions
2. Ensure all libraries are installed: `ldconfig -p | grep osmesa`
3. Verify CUDA setup: `python -c "import jax; print(jax.devices())"`
4. Run setup script again: `source navigation_guide/setup_video_env.sh`

## 📊 Training Results

The hierarchical navigation system achieves:
- **100% success rate** in 3-4 minutes of training
- **300,000 timesteps** for optimal performance
- Binary success tracking (goal reached or not)

## 🎥 Video Output

Generated videos include:
- Robot navigation behavior
- Goal position and distance overlay
- Step counter and success indicator
- Room boundary information

Videos are saved to the `videos/` directory with timestamps.

## 📝 Notes

- The navigation environment uses a flat terrain without visual walls or goal markers
- Goals are represented as coordinates, not visual elements
- The information overlay provides visual feedback about the task
- Training uses a frozen locomotion policy with a trainable navigation policy on top