# Video Generation Issues and Resolutions

This document details all the issues encountered during navigation video generation and their resolutions.

## Issue 1: Scripts Hanging on env.reset()

### Problem
Scripts would hang indefinitely when calling `env.reset()`, making it impossible to run evaluation or generate videos.

### Root Cause
CUDA version mismatch - the system had CUDA 11.8 installed but JAX expected CUDA 12.

### Resolution
Created `setup_cuda12_env.sh` script that uses CUDA 12 libraries from pip packages:
```bash
source setup_cuda12_env.sh
```

## Issue 2: Checkpoint Format Incompatibility

### Problem
Navigation checkpoints were saved in custom format while PPO expected a different format:
- Navigation saved: `{'params': params, 'config': config}`
- PPO expected: `[normalizer_params, policy_params, value_params]`

### Resolution
Created `train_navigation_ppo_format.py` that saves checkpoints in PPO-compatible format.

## Issue 3: Video Rendering Without Display

### Problem
Initial attempts to generate videos failed with errors:
- `GLFWError: The DISPLAY environment variable is missing`
- `AttributeError: 'NoneType' object has no attribute 'glGetError'`

### Root Cause
The environment lacked:
1. A display (X11) for rendering
2. Proper OpenGL libraries for headless rendering
3. Correct environment variables

### Resolution
1. Installed required libraries:
```bash
apt install -y libosmesa6-dev libgl1-mesa-glx libglu1-mesa-dev
apt install -y ffmpeg
```

2. Set OSMesa backend BEFORE importing any modules:
```python
import os
os.environ['MUJOCO_GL'] = 'osmesa'
```

## Issue 4: Missing FFmpeg for Video Encoding

### Problem
`RuntimeError: Program 'ffmpeg' is not found`

### Resolution
Installed ffmpeg:
```bash
apt install -y ffmpeg
```

## Issue 5: Navigation Network Architecture Mismatch

### Problem
Navigation network expected different output dimensions than checkpoint had.

### Resolution
Corrected network architecture to match training:
- Hidden layers: 32-32-6
- Output: 6 dimensions (3 mean + 3 log_std for action distribution)

## Issue 6: Action Sampling from Distribution

### Problem
Network outputs distribution parameters, not actions directly.

### Resolution
Implemented proper action sampling:
```python
dist_params = network.apply(params, obs)
mean = dist_params[..., :3]
log_std = dist_params[..., 3:]
std = jp.exp(log_std)
eps = jax.random.normal(key, mean.shape)
action = mean + std * eps
action = jp.tanh(action)  # Apply tanh squashing
```

## Issue 7: No Visual Goal/Wall Markers

### Problem
Videos showed robot movement but no visual indication of goals or walls.

### Root Cause
The navigation environment is based on flat terrain without visual elements for walls or goals.

### Resolution
Created `generate_navigation_video_with_info.py` that adds text overlay showing:
- Goal position
- Robot position
- Distance to goal
- Step counter
- Room boundaries (text)

## Issue 8: PyOpenGL Platform Detection

### Problem
PyOpenGL couldn't detect the platform properly for EGL/OSMesa rendering.

### Resolution
Used OSMesa backend which worked after installing proper libraries:
```bash
export MUJOCO_GL=osmesa
```

## Summary of Key Solutions

1. **CUDA Issues**: Use `setup_cuda12_env.sh` to set correct CUDA paths
2. **Headless Rendering**: Install OSMesa libraries and set `MUJOCO_GL=osmesa`
3. **Video Encoding**: Install ffmpeg
4. **Missing Visual Elements**: Use information overlay for goal/position display

## Verification Commands

Check if everything is properly set up:
```bash
# Check CUDA
python -c "import jax; print(jax.devices())"

# Check OSMesa
ldconfig -p | grep osmesa

# Check ffmpeg
which ffmpeg

# Test video generation
export MUJOCO_GL=osmesa
python -c "import os; print(os.environ.get('MUJOCO_GL'))"
```