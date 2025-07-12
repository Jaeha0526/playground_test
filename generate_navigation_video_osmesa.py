#!/usr/bin/env python3
"""Generate navigation video using OSMesa backend (headless rendering)."""

import os
# Set OSMesa backend BEFORE any imports - this is the key!
os.environ['MUJOCO_GL'] = 'osmesa'

print("=== Navigation Video Generation (OSMesa Headless) ===\n")

# Now do the CUDA setup
if 'nvidia/cuda_runtime/lib' not in os.environ.get('LD_LIBRARY_PATH', ''):
    print("Setting up CUDA 12 environment...")
    import subprocess
    result = subprocess.run(['bash', '-c', 'source setup_cuda12_env.sh && env'], 
                          capture_output=True, text=True)
    for line in result.stdout.strip().split('\n'):
        if '=' in line:
            key, value = line.split('=', 1)
            os.environ[key] = value

import jax
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
print(f"MuJoCo GL backend: {os.environ.get('MUJOCO_GL')}\n")

from datetime import datetime
from pathlib import Path
import jax.numpy as jp
from flax import linen as nn
from orbax import checkpoint as ocp

from mujoco_playground import registry, locomotion
from src.locomotion_training.envs.go1_simple_navigation import Go1SimpleNavigation, navigation_config
from src.locomotion_training.envs.hierarchical_navigation_wrapper import HierarchicalNavigationWrapper
from src.locomotion_training.utils.video import VideoRenderer, create_joystick_command_overlay

print("Starting video generation process...\n")

# Load policies
print("1. Loading checkpoints...")
checkpointer = ocp.PyTreeCheckpointer()

# Locomotion
loc_data = checkpointer.restore(os.path.abspath("checkpoints/Go1JoystickFlatTerrain_20250630_224046/best"))
normalizer_params = loc_data[0]
policy_params = loc_data[1]

class LocomotionPolicyNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512, name='hidden_0')(x)
        x = nn.swish(x)
        x = nn.Dense(256, name='hidden_1')(x)
        x = nn.swish(x)
        x = nn.Dense(128, name='hidden_2')(x)
        x = nn.swish(x)
        x = nn.Dense(24, name='hidden_3')(x)
        return x

locomotion_network = LocomotionPolicyNetwork()

def locomotion_apply_fn(full_params, obs):
    mean = normalizer_params['mean']['state']
    summed_var = normalizer_params['summed_variance']['state']
    count = normalizer_params['count']
    
    normalized_obs = (obs - mean) / jp.sqrt(summed_var / count + 1e-5)
    actions = locomotion_network.apply({'params': policy_params['params']}, normalized_obs)
    return jp.tanh(actions[..., :12])

# Navigation
nav_data = checkpointer.restore(os.path.abspath("checkpoints/navigation_ppo_20250711_235524/best"))
nav_normalizer = nav_data[0]
nav_policy_params = nav_data[1]

print("✓ Checkpoints loaded\n")

# Create environment
print("2. Creating environment...")
locomotion.register_environment(
    env_name="Go1SimpleNavigation",
    env_class=Go1SimpleNavigation,
    cfg_class=navigation_config
)

base_env = registry.load("Go1SimpleNavigation")
env = HierarchicalNavigationWrapper(
    env=base_env,
    locomotion_apply_fn=locomotion_apply_fn,
    locomotion_params=(normalizer_params, policy_params),
    locomotion_obs_size=48
)
print("✓ Environment created\n")

# Create navigation network
class NavigationPolicyNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(32, name='hidden_0')(x)
        x = nn.swish(x)
        x = nn.Dense(32, name='hidden_1')(x)
        x = nn.swish(x)
        x = nn.Dense(6, name='hidden_2')(x)
        return x

navigation_network = NavigationPolicyNetwork()

@jax.jit
def navigation_inference_fn(obs, key):
    mean = nav_normalizer['mean']
    std = nav_normalizer['std']
    normalized_obs = (obs - mean) / (std + 1e-8)
    
    dist_params = navigation_network.apply({'params': nav_policy_params['params']}, normalized_obs)
    mean = dist_params[..., :3]
    log_std = dist_params[..., 3:]
    
    std = jp.exp(log_std)
    eps = jax.random.normal(key, mean.shape)
    action = mean + std * eps
    
    return jp.tanh(action)

print("3. Running evaluation episode...")

video_renderer = VideoRenderer(
    camera="track",
    width=640,
    height=480,
    render_every=2,
    show_contacts=True
)

# Run episode
rng = jax.random.PRNGKey(123)  # Different seed for different goal
rng, reset_rng = jax.random.split(rng)

print("   Resetting environment...")
reset_fn = jax.jit(env.reset)
state = reset_fn(reset_rng)
print("   ✓ Reset complete")

# Get goal info
goal_pos = state.info['goal_position']
initial_robot_pos = state.data.xpos[base_env._torso_body_id]
initial_distance = jp.sqrt((goal_pos[0] - initial_robot_pos[0])**2 + 
                          (goal_pos[1] - initial_robot_pos[1])**2)
print(f"   Goal position: ({float(goal_pos[0]):.2f}, {float(goal_pos[1]):.2f})")
print(f"   Initial distance: {float(initial_distance):.2f}m")

rollout = []
overlays = []
success = False

step_fn = jax.jit(env.step)

print("\n   Running episode steps...")
max_steps = 100  # 100 steps for proper evaluation
for step in range(max_steps):
    rng, act_rng = jax.random.split(rng)
    action = navigation_inference_fn(state.obs, act_rng)
    
    state = step_fn(state, action)
    rollout.append(state)
    
    # Create overlay for joystick command visualization
    overlay = create_joystick_command_overlay(state, base_env, action, 1.0)
    overlays.append(overlay)
    
    # Check success
    if hasattr(state.info, 'goal_reached') and bool(state.info['goal_reached']):
        success = True
        print(f"   ✓ Goal reached at step {step}!")
        break
    
    if state.done:
        print(f"   Episode terminated at step {step}")
        break
    
    if step % 20 == 0 and step > 0:
        robot_pos = state.data.xpos[base_env._torso_body_id]
        current_distance = jp.sqrt((goal_pos[0] - robot_pos[0])**2 + 
                                  (goal_pos[1] - robot_pos[1])**2)
        print(f"   Step {step}/{max_steps} - Distance to goal: {float(current_distance):.2f}m")

# Final status
final_robot_pos = state.data.xpos[base_env._torso_body_id]
final_distance = jp.sqrt((goal_pos[0] - final_robot_pos[0])**2 + 
                        (goal_pos[1] - final_robot_pos[1])**2)

print(f"\n   Episode complete:")
print(f"   - Success: {'Yes' if success else 'No'}")
print(f"   - Final distance: {float(final_distance):.2f}m")
print(f"   - Total steps: {len(rollout)}")

# Generate video
print("\n4. Generating video with OSMesa...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_path = f"videos/navigation_osmesa_{timestamp}.mp4"
Path("videos").mkdir(exist_ok=True)

try:
    print("   Calling video_renderer.render()...")
    frames = video_renderer.render(
        env=base_env,
        trajectory=rollout,
        modify_scene_fns=overlays,
        save_path=video_path
    )
    
    print(f"\n✓ Video saved successfully!")
    print(f"  Path: {video_path}")
    print(f"  Frames: {len(frames)}")
    print(f"  Duration: {len(frames) / video_renderer.fps:.1f}s")
    print(f"  Resolution: {video_renderer.width}x{video_renderer.height}")
    print(f"\n🎉 Navigation video generation complete!")
    
except Exception as e:
    print(f"\n❌ Video generation failed: {e}")
    import traceback
    traceback.print_exc()
    print("\nNote: OSMesa backend requires proper graphics libraries.")
    print("If this fails, you may need to install: libosmesa6-dev")