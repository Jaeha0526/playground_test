#!/usr/bin/env python3
"""Generate navigation video with info overlay showing goal and distance."""

import os
os.environ['MUJOCO_GL'] = 'osmesa'

print("=== Navigation Video Generation with Info Overlay ===\n")

# CUDA setup
if 'nvidia/cuda_runtime/lib' not in os.environ.get('LD_LIBRARY_PATH', ''):
    import subprocess
    result = subprocess.run(['bash', '-c', 'source setup_cuda12_env.sh && env'], 
                          capture_output=True, text=True)
    for line in result.stdout.strip().split('\n'):
        if '=' in line:
            key, value = line.split('=', 1)
            os.environ[key] = value

import jax
import jax.numpy as jp
from flax import linen as nn
from orbax import checkpoint as ocp
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import mediapy as media

from mujoco_playground import registry, locomotion
from src.locomotion_training.envs.go1_simple_navigation import Go1SimpleNavigation, navigation_config
from src.locomotion_training.envs.hierarchical_navigation_wrapper import HierarchicalNavigationWrapper
from src.locomotion_training.utils.video import VideoRenderer, create_joystick_command_overlay

print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}\n")

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
print(f"  Room size: {base_env._room_size}m x {base_env._room_size}m")
print(f"  Goal radius: {base_env._goal_radius}m")
print(f"  Goal distance range: {base_env._min_goal_distance}m - {base_env._max_goal_distance}m\n")

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
rng = jax.random.PRNGKey(789)
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
distances = []
robot_positions = []
success = False

step_fn = jax.jit(env.step)

print("\n   Running episode steps...")
max_steps = 200
for step in range(max_steps):
    rng, act_rng = jax.random.split(rng)
    action = navigation_inference_fn(state.obs, act_rng)
    
    state = step_fn(state, action)
    rollout.append(state)
    
    # Track robot position and distance
    robot_pos = state.data.xpos[base_env._torso_body_id]
    robot_positions.append([float(robot_pos[0]), float(robot_pos[1])])
    distance = jp.sqrt((goal_pos[0] - robot_pos[0])**2 + (goal_pos[1] - robot_pos[1])**2)
    distances.append(float(distance))
    
    # Create joystick overlay
    overlay = create_joystick_command_overlay(state, base_env, action, 1.0)
    overlays.append(overlay)
    
    if hasattr(state.info, 'goal_reached') and bool(state.info['goal_reached']):
        success = True
        print(f"   ✓ Goal reached at step {step}!")
        break
    
    if state.done:
        print(f"   Episode terminated at step {step}")
        break
    
    if step % 40 == 0 and step > 0:
        print(f"   Step {step}/{max_steps} - Distance to goal: {distance:.2f}m")

final_distance = distances[-1] if distances else initial_distance
print(f"\n   Episode complete:")
print(f"   - Success: {'Yes' if success else 'No'}")
print(f"   - Final distance: {final_distance:.2f}m")
print(f"   - Distance improvement: {float(initial_distance - final_distance):.2f}m")

# Generate video
print("\n4. Generating video...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_path = f"videos/navigation_info_{timestamp}.mp4"
Path("videos").mkdir(exist_ok=True)

# Render frames
frames = video_renderer.render(
    env=base_env,
    trajectory=rollout,
    modify_scene_fns=overlays,
    save_path=None  # Don't save yet, we'll add overlay first
)

print(f"   Adding information overlay...")

# Add text overlay to frames
annotated_frames = []
for i, frame in enumerate(frames):
    # Create a copy to modify
    frame_copy = frame.copy()
    
    # Add semi-transparent background for text
    overlay = frame_copy.copy()
    cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
    frame_copy = cv2.addWeighted(frame_copy, 0.7, overlay, 0.3, 0)
    
    # Add text
    step_num = i * 2  # render_every=2
    distance = distances[min(step_num, len(distances)-1)]
    robot_x, robot_y = robot_positions[min(step_num, len(robot_positions)-1)]
    
    cv2.putText(frame_copy, f"Navigation Task", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame_copy, f"Goal: ({float(goal_pos[0]):.1f}, {float(goal_pos[1]):.1f})", 
                (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame_copy, f"Robot: ({robot_x:.1f}, {robot_y:.1f})", 
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame_copy, f"Distance: {distance:.2f}m", 
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame_copy, f"Step: {step_num}", 
                (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Add success indicator if reached
    if success and step_num >= len(rollout) - 2:
        cv2.putText(frame_copy, "GOAL REACHED!", (200, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    
    # Add boundary indicators (text only since we can't draw in 3D)
    cv2.putText(frame_copy, "Room: 10m x 10m", (480, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    annotated_frames.append(frame_copy)

# Save video with annotations
media.write_video(video_path, annotated_frames, fps=video_renderer.fps)

print(f"\n✓ Video saved successfully!")
print(f"  Path: {video_path}")
print(f"  Frames: {len(frames)}")
print(f"  Duration: {len(frames) / video_renderer.fps:.1f}s")
print(f"  Resolution: {video_renderer.width}x{video_renderer.height}")
print(f"\n🎉 Navigation video with information overlay complete!")
print("\nNote: The environment uses a flat terrain without visual walls or goal markers.")
print("The robot navigates based on internal goal coordinates shown in the overlay.")