#!/usr/bin/env python3
"""Test video generation using the exact working pattern from the notebook."""

import os
import subprocess
from pathlib import Path

# Set up environment exactly like the notebook
print("Setting up environment like the Colab notebook...")

# 1. Check GPU
print("1. Checking GPU...")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("   ✅ GPU detected")
        print(f"   Driver: {result.stdout.split('Driver Version:')[1].split('|')[0].strip()}")
    else:
        print("   ❌ No GPU detected")
        exit(1)
except:
    print("   ❌ nvidia-smi failed")
    exit(1)

# 2. Set up Nvidia EGL ICD configuration
print("2. Setting up Nvidia EGL ICD configuration...")
nvidia_icd_config_path = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'

if not os.path.exists(nvidia_icd_config_path):
    print(f"   Creating {nvidia_icd_config_path}...")
    try:
        Path(nvidia_icd_config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(nvidia_icd_config_path, 'w') as f:
            f.write("""{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
""")
        print("   ✅ ICD configuration created")
    except PermissionError:
        print("   ❌ Permission denied")
        exit(1)
else:
    print("   ✅ ICD configuration already exists")

# 3. Set MuJoCo GL backend
print("3. Setting MUJOCO_GL=egl...")
os.environ['MUJOCO_GL'] = 'egl'
print(f"   MUJOCO_GL = {os.environ.get('MUJOCO_GL')}")

# 4. Test basic MuJoCo
print("4. Testing basic MuJoCo...")
try:
    import mujoco
    model = mujoco.MjModel.from_xml_string('<mujoco/>')
    print("   ✅ Basic MuJoCo test passed")
except Exception as e:
    print(f"   ❌ Basic MuJoCo test failed: {e}")
    exit(1)

# 5. Test video generation using exact notebook pattern
print("5. Testing video generation...")
try:
    import jax
    import jax.numpy as jp
    import mediapy as media
    from mujoco_playground import registry
    
    # Load environment exactly like notebook
    env_name = "Go1JoystickFlatTerrain"
    env = registry.load(env_name)
    
    print(f"   Environment loaded: {env_name}")
    
    # Create a simple rollout
    rng = jax.random.PRNGKey(0)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    
    state = reset_fn(rng)
    trajectory = [state]
    
    # Run for 50 steps with small random actions
    for i in range(50):
        rng, act_rng = jax.random.split(rng)
        action = jax.random.normal(act_rng, (12,)) * 0.1  # Small random actions
        state = step_fn(state, action)
        trajectory.append(state)
    
    print(f"   Created trajectory with {len(trajectory)} states")
    
    # Render using exact notebook pattern
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    
    render_every = 2
    traj = trajectory[::render_every]
    
    print("   Rendering frames...")
    frames = env.render(
        traj,
        camera="track",
        scene_option=scene_option,
        width=640,
        height=480,
    )
    
    print(f"   ✅ Rendered {len(frames)} frames, shape: {frames.shape}")
    
    # Save video
    fps = 1.0 / env.dt / render_every
    video_path = "test_video_working.mp4"
    
    print(f"   Saving video to {video_path}...")
    media.write_video(video_path, frames, fps=int(fps))
    
    if os.path.exists(video_path):
        file_size = os.path.getsize(video_path)
        print(f"   ✅ Video saved successfully! ({file_size} bytes)")
        print(f"   Duration: {len(frames)/fps:.1f} seconds at {fps:.1f} FPS")
        
        # Clean up
        os.remove(video_path)
        print("   🧹 Test video cleaned up")
        
        print("\n🎉 VIDEO GENERATION WORKING!")
        print("Your environment is properly configured for headless video generation.")
        print("\nTo use in your scripts:")
        print("export MUJOCO_GL=egl")
        print("# (and ensure the ICD configuration file exists)")
        
    else:
        print("   ❌ Video file was not created")
        
except Exception as e:
    print(f"   ❌ Video generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✅ All tests passed! Video generation is working.")