#!/usr/bin/env python3
"""Generate video for trained locomotion policy."""

import os
from pathlib import Path
from datetime import datetime
import jax
import jax.numpy as jp
from orbax import checkpoint as ocp
from flax import linen as nn

from mujoco_playground import registry
from src.locomotion_training.utils.video import VideoRenderer, create_joystick_command_overlay


def load_locomotion_policy(checkpoint_path: str):
    """Load locomotion policy from checkpoint."""
    print(f"Loading locomotion policy from: {checkpoint_path}")
    
    abs_checkpoint_path = os.path.abspath(checkpoint_path)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpoint_data = checkpointer.restore(abs_checkpoint_path)
    
    normalizer_params = checkpoint_data[0]
    policy_params = checkpoint_data[1]
    
    # Create policy network
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
    
    policy_network = LocomotionPolicyNetwork()
    
    def inference_fn(obs):
        """Apply locomotion policy."""
        # Extract state observation from dict
        if isinstance(obs, dict):
            state_obs = obs['state']
        else:
            state_obs = obs
            
        # Normalize observations
        mean = normalizer_params['mean']['state']
        summed_var = normalizer_params['summed_variance']['state']
        count = normalizer_params['count']
        
        normalized_obs = (state_obs - mean) / jp.sqrt(summed_var / count + 1e-5)
        
        # Apply policy network
        outputs = policy_network.apply({'params': policy_params['params']}, normalized_obs)
        # Take only first 12 outputs (joint actions) and apply tanh
        actions = jp.tanh(outputs[..., :12])
        return actions
    
    print("✓ Locomotion policy loaded")
    return inference_fn, normalizer_params, policy_params


def main():
    """Generate locomotion video."""
    print("🎬 Locomotion Video Generation\n")
    
    # Load locomotion policy
    checkpoint_path = "checkpoints/Go1JoystickFlatTerrain_20250630_224046/best"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    inference_fn, normalizer_params, policy_params = load_locomotion_policy(checkpoint_path)
    
    # Load environment
    print("\n🌍 Creating environment...")
    env_name = "Go1JoystickFlatTerrain"
    env = registry.load(env_name)
    
    print(f"✓ Environment created: {env_name}")
    print(f"  - Action size: {env.action_size}")
    print(f"  - Observation size: {env.observation_size}")
    
    # Create video renderer
    video_renderer = VideoRenderer(
        camera="track",
        width=640,
        height=480,
        render_every=2,
        show_contacts=True,
        show_perturbations=False
    )
    
    # JIT compile functions
    print("\n⚡ Compiling functions...")
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference = jax.jit(inference_fn)
    
    # Generate episode with specific command
    print("\n🎮 Generating locomotion episode...")
    
    # Define velocity command [x_vel, y_vel, yaw_vel]
    command = jp.array([1.0, 0.0, 0.0])  # Forward at 1 m/s
    print(f"  Command: {command} (forward walking)")
    
    # Run episode
    rng = jax.random.PRNGKey(42)
    state = jit_reset(rng)
    
    # Set command
    state.info['command'] = command
    
    rollout = []
    modify_scene_fns = []
    
    max_steps = 300
    for step in range(max_steps):
        # Get observation
        obs = state.obs
        
        # Get action from policy
        action = jit_inference(obs)
        
        # Step environment
        state = jit_step(state, action)
        state.info['command'] = command  # Keep command constant
        
        rollout.append(state)
        
        # Create command overlay
        if hasattr(state, 'data'):
            overlay_fn = create_joystick_command_overlay(
                state, env, command,
                scale_factor=abs(command[0]) / 2.0 if abs(command[0]) > 0 else 1.0
            )
            modify_scene_fns.append(overlay_fn)
    
    print(f"  Episode length: {len(rollout)} steps")
    
    # Check if we have valid states
    if rollout and hasattr(rollout[0], 'data'):
        # Create video
        print("\n🎥 Rendering video...")
        
        video_dir = Path("videos/locomotion_demo")
        video_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = video_dir / f"locomotion_{timestamp}.mp4"
        
        try:
            # Use the video renderer
            frames = video_renderer.render(
                env=env,
                trajectory=[s.data for s in rollout],
                modify_scene_fns=modify_scene_fns,
                save_path=str(video_path)
            )
            
            print(f"\n✅ Video generation complete!")
            print(f"   - Video saved: {video_path}")
            print(f"   - Frames: {len(frames) if frames is not None else 0}")
            print(f"   - Duration: {len(frames)/video_renderer.fps:.1f}s @ {video_renderer.fps}fps" if frames is not None else "N/A")
            
            return str(video_path)
            
        except Exception as e:
            print(f"\n❌ Video rendering failed: {e}")
            print("   This is likely due to missing display/OpenGL context")
            
            # Save episode data instead
            import pickle
            data_path = video_dir / f"locomotion_episode_{timestamp}.pkl"
            
            episode_data = {
                'rollout': rollout,
                'command': command,
                'env_name': env_name,
                'checkpoint': checkpoint_path,
                'modify_scene_fns': len(modify_scene_fns)
            }
            
            with open(data_path, 'wb') as f:
                pickle.dump(episode_data, f)
            
            print(f"\n💾 Episode data saved: {data_path}")
            print("   Can be rendered later on a machine with display")
            
            return None
    else:
        print("❌ No valid states for rendering")
        return None


if __name__ == "__main__":
    video_path = main()
    
    if video_path:
        print(f"\n🎉 Success! Locomotion video created at: {video_path}")
    else:
        print(f"\n⚠️  Video rendering requires display/OpenGL context")
        print(f"   Episode data has been saved for later rendering")