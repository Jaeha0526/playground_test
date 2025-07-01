#!/usr/bin/env python3
"""Evaluation script using exact notebook video generation pattern."""

import jax
import jax.numpy as jp
import functools
import numpy as np
from pathlib import Path

import mediapy as media
import mujoco

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import wrapper

def evaluate_checkpoint_with_video(checkpoint_path: str, x_vel: float = 1.0, y_vel: float = 0.0, yaw_vel: float = 0.0, num_episodes: int = 3):
    """Evaluate checkpoint using exact notebook pattern with video generation."""
    
    env_name = "Go1JoystickFlatTerrain"
    print(f"🚀 Evaluating {env_name}")
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"🎯 Command: x={x_vel}, y={y_vel}, yaw={yaw_vel}")
    
    # Load environment and config
    env = registry.load(env_name)
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    
    # Setup network factory
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )
    
    # Load checkpoint using ppo.train (the working method)
    train_params = dict(ppo_params)
    train_params.update({
        'num_timesteps': 0,  # Skip training, just load
        'num_evals': 1,
        'network_factory': network_factory,
    })
    
    print("📦 Loading checkpoint...")
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        eval_env=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        restore_checkpoint_path=checkpoint_path,
        **train_params
    )
    
    print("✅ Checkpoint loaded successfully!")
    
    # Create inference function
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)
    
    # Setup environment functions
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    
    # Setup scene options for video (exact notebook pattern)
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    
    # Video parameters (exact notebook pattern)
    render_every = 2
    fps = 1.0 / env.dt / render_every
    
    print(f"🎮 Running {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    all_trajectories = []
    
    for episode in range(num_episodes):
        rng = jax.random.PRNGKey(episode)
        rng, reset_rng = jax.random.split(rng)
        
        # Reset environment
        state = reset_fn(reset_rng)
        
        episode_reward = 0.0
        step_count = 0
        max_steps = 1000
        trajectory = [state]  # Store full trajectory
        
        print(f"  Episode {episode + 1}/{num_episodes}: ", end="", flush=True)
        
        for step in range(max_steps):
            # Get action from policy
            rng, act_rng = jax.random.split(rng)
            action, _ = jit_inference_fn(state.obs, act_rng)
            
            # Step environment
            state = step_fn(state, action)
            trajectory.append(state)
            episode_reward += state.reward
            step_count += 1
            
            if state.done:
                break
        
        episode_rewards.append(float(episode_reward))
        episode_lengths.append(step_count)
        all_trajectories.append(trajectory)
        
        print(f"Reward: {episode_reward:.3f}, Length: {step_count}")
    
    # Print results
    print(f"\n📊 Results:")
    print(f"  Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"  Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Max reward: {np.max(episode_rewards):.3f}")
    print(f"  Min reward: {np.min(episode_rewards):.3f}")
    
    # Generate video for best episode (exact notebook pattern)
    best_episode_idx = np.argmax(episode_rewards)
    best_trajectory = all_trajectories[best_episode_idx]
    
    print(f"\n🎬 Generating video for best episode (Episode {best_episode_idx + 1})...")
    
    # Subsample trajectory for rendering (exact notebook pattern)
    traj = best_trajectory[::render_every]
    
    print(f"  Rendering {len(traj)} frames...")
    
    # Render frames using exact notebook method
    frames = env.render(
        traj,
        camera="track", 
        scene_option=scene_option,
        width=640,
        height=480,
    )
    
    # Save video
    Path("videos").mkdir(exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"videos/Go1_best_episode_{timestamp}.mp4"
    
    print(f"  Saving video to: {video_path}")
    media.write_video(video_path, frames, fps=int(fps))
    
    print(f"✅ Video saved! ({len(frames)} frames, {len(frames)/fps:.1f}s duration)")
    print(f"🎉 Evaluation completed successfully!")
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'video_path': video_path,
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_with_video.py <checkpoint_path> [x_vel] [y_vel] [yaw_vel] [num_episodes]")
        print("Example: python evaluate_with_video.py /path/to/checkpoint 1.0 0.0 0.0 3")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    x_vel = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    y_vel = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    yaw_vel = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    num_episodes = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    
    results = evaluate_checkpoint_with_video(checkpoint_path, x_vel, y_vel, yaw_vel, num_episodes)