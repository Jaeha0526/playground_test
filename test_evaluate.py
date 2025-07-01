#!/usr/bin/env python3
"""Simple test script to evaluate the best checkpoint using notebook pattern."""

import jax
import functools
from pathlib import Path

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import wrapper

def test_evaluate_best():
    """Test evaluation using exact notebook pattern."""
    
    # Configuration
    env_name = "Go1JoystickFlatTerrain"
    checkpoint_path = "/workspace/playground_test/checkpoints/Go1JoystickFlatTerrain_20250630_224046/best"
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load environment and config
    env = registry.load(env_name)
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    
    # Setup network factory exactly as in training
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )
    
    print("Loading checkpoint...")
    
    # Use ppo.train with num_timesteps=0 to load checkpoint
    train_params = dict(ppo_params)
    train_params.update({
        'num_timesteps': 0,  # Skip training, just load
        'num_evals': 1,
        'network_factory': network_factory,
    })
    
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        eval_env=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        restore_checkpoint_path=checkpoint_path,
        **train_params
    )
    
    print("✅ Checkpoint loaded successfully!")
    print(f"Inference function created: {type(make_inference_fn)}")
    
    # Test creating inference function
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)
    
    print("✅ Inference function created successfully!")
    
    # Test single step
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    
    rng = jax.random.PRNGKey(0)
    rng, reset_rng = jax.random.split(rng)
    
    state = reset_fn(reset_rng)
    
    print(f"Environment reset successful. Observation keys: {list(state.obs.keys()) if isinstance(state.obs, dict) else 'not dict'}")
    
    # Test inference
    rng, act_rng = jax.random.split(rng)
    action, _ = jit_inference_fn(state.obs, act_rng)
    
    print(f"✅ Inference successful! Action shape: {action.shape}")
    print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")
    
    # Test environment step
    next_state = step_fn(state, action)
    print(f"✅ Environment step successful! Reward: {next_state.reward:.3f}")
    
    print("\n🎉 All tests passed! The checkpoint and evaluation system is working correctly.")
    
    return make_inference_fn, params

if __name__ == "__main__":
    test_evaluate_best()