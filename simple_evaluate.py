#!/usr/bin/env python3
"""Simple evaluation script that works using the notebook pattern."""

import jax
import jax.numpy as jp
import functools
import numpy as np
from pathlib import Path
import os

# Set up headless rendering
os.environ['MUJOCO_GL'] = 'osmesa'  # Use OSMesa for software rendering

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import wrapper
import mediapy as media
from datetime import datetime

def evaluate_checkpoint(checkpoint_path: str, x_vel: float = 1.0, y_vel: float = 0.0, yaw_vel: float = 0.0, num_episodes: int = 5, save_video: bool = True):
    """Evaluate checkpoint using the working notebook pattern."""
    
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
    
    # Run evaluation episodes
    print(f"🎮 Running {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    all_frames = []  # For video generation
    
    for episode in range(num_episodes):
        rng = jax.random.PRNGKey(episode)
        rng, reset_rng = jax.random.split(rng)
        
        # Reset environment
        state = reset_fn(reset_rng)
        
        # Override command in state to use our specified command
        command = jp.array([x_vel, y_vel, yaw_vel])
        
        episode_reward = 0.0
        step_count = 0
        max_steps = 1000
        episode_frames = []
        
        print(f"  Episode {episode + 1}/{num_episodes}: ", end="", flush=True)
        
        for step in range(max_steps):
            # Render frame for video (every 5 steps to reduce video size)
            if save_video and step % 5 == 0:
                frame = env.render(state, camera="track", width=640, height=480)
                episode_frames.append(frame)
            
            # Get action from policy
            rng, act_rng = jax.random.split(rng)
            action, _ = jit_inference_fn(state.obs, act_rng)
            
            # Step environment
            state = step_fn(state, action)
            episode_reward += state.reward
            step_count += 1
            
            if state.done:
                break
        
        if save_video and episode_frames:
            all_frames.extend(episode_frames)
        
        episode_rewards.append(float(episode_reward))
        episode_lengths.append(step_count)
        
        print(f"Reward: {episode_reward:.3f}, Length: {step_count}")
    
    # Print results
    print(f"\n📊 Results:")
    print(f"  Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"  Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Max reward: {np.max(episode_rewards):.3f}")
    print(f"  Min reward: {np.min(episode_rewards):.3f}")
    
    # Save video if requested
    if save_video and all_frames:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f"videos/Go1_evaluation_{timestamp}.mp4"
        Path("videos").mkdir(exist_ok=True)
        
        print(f"🎬 Saving video to: {video_path}")
        print(f"   Frames: {len(all_frames)}, Duration: ~{len(all_frames)/12:.1f}s")
        
        # Save video at 12 fps (since we render every 5 steps at 20Hz = 4Hz, then *3 for reasonable speed)
        media.write_video(video_path, all_frames, fps=12)
        print(f"✅ Video saved!")
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python simple_evaluate.py <checkpoint_path> [x_vel] [y_vel] [yaw_vel] [num_episodes]")
        print("Example: python simple_evaluate.py /path/to/checkpoint 1.0 0.0 0.0 5")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    x_vel = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    y_vel = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    yaw_vel = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    num_episodes = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    save_video = sys.argv[6].lower() != 'false' if len(sys.argv) > 6 else True
    
    results = evaluate_checkpoint(checkpoint_path, x_vel, y_vel, yaw_vel, num_episodes, save_video)