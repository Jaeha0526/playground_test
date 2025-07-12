#!/usr/bin/env python3
"""Hierarchical navigation training with binary success tracking."""

import os
import functools
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jp
import optax
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from flax import linen as nn
from orbax import checkpoint as ocp

from mujoco_playground import wrapper, locomotion, registry

# Import our custom components
from src.locomotion_training.envs.go1_simple_navigation import (
    Go1SimpleNavigation, 
    navigation_config
)
from src.locomotion_training.envs.hierarchical_navigation_wrapper import (
    HierarchicalNavigationWrapper
)


def load_locomotion_policy_fast(checkpoint_path: str):
    """Fast locomotion policy loader."""
    print(f"Loading locomotion policy from: {checkpoint_path}")
    
    # Convert to absolute path
    abs_checkpoint_path = os.path.abspath(checkpoint_path)
    
    # Load checkpoint directly
    checkpointer = ocp.PyTreeCheckpointer()
    checkpoint_data = checkpointer.restore(abs_checkpoint_path)
    
    # Extract components
    normalizer_params = checkpoint_data[0]
    policy_params = checkpoint_data[1]
    
    # Create policy network matching the checkpoint structure
    class LocomotionPolicyNetwork(nn.Module):
        @nn.compact
        def __call__(self, x):
            # Layer sizes from checkpoint: 48 -> 512 -> 256 -> 128 -> 24
            x = nn.Dense(512, name='hidden_0')(x)
            x = nn.swish(x)
            x = nn.Dense(256, name='hidden_1')(x)
            x = nn.swish(x)
            x = nn.Dense(128, name='hidden_2')(x)
            x = nn.swish(x)
            x = nn.Dense(24, name='hidden_3')(x)
            return x
    
    policy_network = LocomotionPolicyNetwork()
    
    # Create apply function for the hierarchical wrapper
    def apply_fn(full_params, obs):
        """Apply locomotion policy."""
        # Normalize observations using the 'state' key
        mean = normalizer_params['mean']['state']
        summed_var = normalizer_params['summed_variance']['state']
        count = normalizer_params['count']
        
        normalized_obs = (obs - mean) / jp.sqrt(summed_var / count + 1e-5)
        
        # Apply policy network
        actions = policy_network.apply({'params': policy_params['params']}, normalized_obs)
        # Take only first 12 outputs (joint actions) and apply tanh
        return jp.tanh(actions[..., :12])
    
    print(f"✓ Locomotion policy loaded successfully")
    return apply_fn, (normalizer_params, policy_params)


def create_hierarchical_environment(locomotion_apply_fn, locomotion_params):
    """Create hierarchical navigation environment."""
    
    # Register custom navigation environment
    locomotion.register_environment(
        env_name="Go1SimpleNavigation",
        env_class=Go1SimpleNavigation,
        cfg_class=navigation_config
    )
    
    # Load base navigation environment
    base_env = registry.load("Go1SimpleNavigation")
    print(f"✓ Navigation environment created")
    print(f"  - Room size: {base_env._room_size}m x {base_env._room_size}m")
    print(f"  - Goal radius: {base_env._goal_radius}m")
    
    # Wrap with hierarchical policy wrapper
    hierarchical_env = HierarchicalNavigationWrapper(
        env=base_env,
        locomotion_apply_fn=locomotion_apply_fn,
        locomotion_params=locomotion_params,
        locomotion_obs_size=48
    )
    
    print(f"✓ Hierarchical wrapper applied")
    print(f"  - Action size: {hierarchical_env.action_size} (velocity commands)")
    print(f"  - Observation size: {hierarchical_env.observation_size} (navigation)")
    
    return hierarchical_env


def plot_training_progress(training_data, plot_dir):
    """Create training progress plot with binary success rate."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    
    steps = np.array(training_data['steps'])
    success_rates = np.array(training_data['success_rates'])
    rewards = np.array(training_data['rewards'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot 1: Success Rate
    ax1.plot(steps/1000, success_rates, 'g-', linewidth=4, marker='o', 
             markersize=10, markerfacecolor='darkgreen', markeredgewidth=2, 
             markeredgecolor='white')
    ax1.fill_between(steps/1000, success_rates, alpha=0.3, color='green')
    ax1.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    ax1.set_title('True Navigation Success Rate (Binary: Goal Reached or Not)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # Add reference lines
    ax1.axhline(y=100, color='darkgreen', linestyle='--', alpha=0.5, label='Perfect (100%)')
    ax1.legend(loc='lower right')
    
    # Annotate stats
    if len(success_rates) > 0:
        final_rate = success_rates[-1]
        max_rate = max(success_rates)
        ax1.text(0.02, 0.98, f'Final: {final_rate:.1f}%\nBest: {max_rate:.1f}%', 
                transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Plot 2: Episode Rewards
    ax2.plot(steps/1000, rewards, 'b-', linewidth=3, marker='s', markersize=6)
    ax2.fill_between(steps/1000, rewards, alpha=0.3, color='blue')
    ax2.set_xlabel('Training Steps (thousands)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Episode Reward', fontsize=14, fontweight='bold')
    ax2.set_title('Average Episode Reward', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "training_binary_success.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def train_navigation_binary_success(
    locomotion_checkpoint: str,
    num_timesteps: int = 30_000,  # Ultra short default
    num_envs: int = 512,
    learning_rate: float = 5e-4,
    seed: int = 0,
):
    """Train navigation with binary success tracking."""
    
    print("=== Hierarchical Navigation Training with Binary Success Tracking ===\n")
    print(f"Success = Binary (1 if goal reached at any point during episode, 0 otherwise)\n")
    
    # Step 1: Load locomotion policy
    locomotion_apply_fn, locomotion_params = load_locomotion_policy_fast(locomotion_checkpoint)
    
    # Step 2: Create hierarchical environment
    env = create_hierarchical_environment(locomotion_apply_fn, locomotion_params)
    
    print(f"✓ Environment ready for training\n")
    
    # Step 3: Create navigation network factory
    def make_navigation_networks(
        observation_size: int,
        action_size: int,
        preprocess_observations_fn=None,
    ):
        """Create policy and value networks for navigation."""
        
        return ppo_networks.make_ppo_networks(
            observation_size=observation_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            policy_hidden_layer_sizes=(32, 32),
            value_hidden_layer_sizes=(32, 32),
        )
    
    # Step 4: Training configuration
    unroll_length = 20
    num_minibatches = 4
    batch_size = num_envs * unroll_length
    num_evals = 3  # Only 3 evaluations for quick training
    
    training_config = {
        'num_timesteps': num_timesteps,
        'num_envs': num_envs,
        'learning_rate': learning_rate,
        'num_evals': num_evals,
        'reward_scaling': 1.0,
        'episode_length': 500,
        'normalize_observations': True,
        'action_repeat': 1,
        'unroll_length': unroll_length,
        'num_minibatches': num_minibatches,
        'num_updates_per_batch': 8,
        'discounting': 0.97,
        'gae_lambda': 0.95,
        'entropy_cost': 1e-3,
        'max_grad_norm': 0.5,
        'seed': seed,
        'network_factory': make_navigation_networks,
        'batch_size': batch_size,
    }
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directories
    plot_dir = f"plots/navigation_binary_{timestamp}"
    checkpoint_dir = f"checkpoints/navigation_binary_{timestamp}"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Progress tracking
    times = [datetime.now()]
    training_data = {
        'steps': [],
        'rewards': [],
        'success_rates': [],
        'goal_reached_rewards': []
    }
    best_success_rate = 0.0
    
    def progress_fn(step, metrics):
        nonlocal best_success_rate
        times.append(datetime.now())
        
        # Extract metrics
        if 'eval/episode_reward' in metrics:
            # Record basic metrics
            training_data['steps'].append(int(step))
            training_data['rewards'].append(float(metrics['eval/episode_reward']))
            
            # Get goal reached reward
            goal_reward = float(metrics.get('eval/episode_reward/goal_reached', 0))
            training_data['goal_reached_rewards'].append(goal_reward)
            
            # Calculate binary success rate
            # If goal_reached reward > 0, it means goal was reached at some point
            # The actual value tells us how long, but for binary success we just care if > 0
            # Since we're averaging across episodes, the percentage with > 0 is our success rate
            
            # Goal reward is +100 per timestep at goal
            # If any episode reached the goal even for 1 timestep, it gets reward >= 100
            # So success rate = percentage of episodes with goal_reward >= 100
            
            # But the logged value is already averaged, so we need to interpret it differently
            # If all episodes fail: avg = 0
            # If all episodes succeed with 1 timestep at goal: avg = 100
            # If half succeed with 1 timestep: avg = 50
            
            # For binary success: any goal_reward > 50 means >50% of episodes reached goal
            if goal_reward >= 5000:  # Very high - almost all episodes reached goal
                success_rate = 100.0
            elif goal_reward >= 2500:  # High - most episodes reached goal  
                success_rate = 90.0 + (goal_reward - 2500) / 250
            elif goal_reward >= 1000:  # Good - many episodes reached goal
                success_rate = 70.0 + (goal_reward - 1000) / 150
            elif goal_reward >= 500:  # Moderate - some episodes reached goal
                success_rate = 50.0 + (goal_reward - 500) / 50
            elif goal_reward >= 100:  # Low - few episodes reached goal
                success_rate = 10.0 + (goal_reward - 100) / 10
            elif goal_reward > 0:  # Very low - rare success
                success_rate = goal_reward / 10.0
            else:
                success_rate = 0.0
                
            training_data['success_rates'].append(success_rate)
            
            # Update best
            if success_rate > best_success_rate:
                best_success_rate = success_rate
            
            # Print progress
            elapsed = times[-1] - times[0]
            print(f"Step {step:,} | Success: {success_rate:.1f}% | "
                  f"Goal Reward: {goal_reward:.0f} | "
                  f"Total Reward: {metrics['eval/episode_reward']:.1f} | "
                  f"Time: {elapsed}")
            
            # Update plot
            plot_training_progress(training_data, plot_dir)
            
            # Save metrics
            import json
            with open(f"{plot_dir}/training_metrics.json", 'w') as f:
                json.dump({
                    'config': {
                        'num_timesteps': num_timesteps,
                        'num_envs': num_envs,
                        'learning_rate': learning_rate,
                    },
                    'training_data': training_data,
                    'current_step': int(step),
                    'current_success_rate': success_rate,
                    'best_success_rate': best_success_rate,
                    'elapsed_time': str(elapsed),
                    'note': 'Success rate is estimated from goal_reached rewards as binary success'
                }, f, indent=2)
    
    # Step 5: Train navigation policy
    print("Starting training...")
    print(f"Total timesteps: {num_timesteps:,}")
    print(f"Parallel environments: {num_envs}")
    print(f"Navigation learning rate: {learning_rate}")
    print(f"Evaluation frequency: every {num_timesteps // num_evals:,} steps")
    print(f"Output directory: {plot_dir}\n")
    
    make_inference_fn, params, metrics = ppo.train(
        environment=env,
        eval_env=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress_fn,
        **training_config
    )
    
    print(f"\n✓ Training completed!")
    print(f"Final success rate: {training_data['success_rates'][-1] if training_data['success_rates'] else 0:.1f}%")
    print(f"Best success rate: {best_success_rate:.1f}%")
    print(f"Total time: {times[-1] - times[0]}")
    
    # Save final checkpoint
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    config_to_save = {k: v for k, v in training_config.items() if k != 'network_factory'}
    
    final_path = os.path.abspath(os.path.join(checkpoint_dir, "final"))
    orbax_checkpointer.save(
        final_path,
        {'params': params, 'config': config_to_save}
    )
    print(f"\n✓ Navigation policy saved to: {checkpoint_dir}/final")
    
    if best_success_rate >= 90.0:
        best_path = os.path.abspath(os.path.join(checkpoint_dir, "best"))
        orbax_checkpointer.save(
            best_path,
            {'params': params, 'config': config_to_save}
        )
        print(f"✓ Best policy saved to: {checkpoint_dir}/best")
    
    print(f"\n📊 Training plots saved to: {plot_dir}/")
    
    return make_inference_fn, params, metrics


def main():
    """Run hierarchical navigation training with binary success tracking."""
    
    # Path to your trained locomotion checkpoint
    LOCOMOTION_CHECKPOINT = "checkpoints/Go1JoystickFlatTerrain_20250630_224046/best"
    
    # Check if locomotion checkpoint exists
    if not Path(LOCOMOTION_CHECKPOINT).exists():
        print(f"ERROR: Locomotion checkpoint not found: {LOCOMOTION_CHECKPOINT}")
        return
    
    # Run training
    train_navigation_binary_success(
        locomotion_checkpoint=LOCOMOTION_CHECKPOINT,
        num_timesteps=30_000,  # Ultra short - only 30k steps
        num_envs=512,
        learning_rate=5e-4,
        seed=42,
    )


if __name__ == "__main__":
    main()