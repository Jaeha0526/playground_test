"""Configuration management for locomotion training."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
from pathlib import Path


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    env_name: str = 'Go1JoystickFlatTerrain'
    num_timesteps: int = 100_000_000
    num_evals: int = 20
    episode_length: int = 1000
    eval_every: int = 1_000_000
    deterministic_eval: bool = True
    normalize_observations: bool = True
    reward_scaling: float = 1e-3
    seed: int = 1
    
    # PPO specific parameters
    num_envs: int = 2048
    batch_size: int = 1024
    num_minibatches: int = 32
    num_updates_per_batch: int = 4
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-2
    discounting: float = 0.97
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.3
    normalize_advantage: bool = True
    
    # Checkpointing
    checkpoint_logdir: Optional[str] = None
    save_interval: int = 10_000_000


@dataclass
class EvalConfig:
    """Evaluation configuration parameters."""
    env_name: str = 'Go1JoystickFlatTerrain'
    num_episodes: int = 10
    episode_length: int = 1000
    render_every: int = 2
    camera: str = "track"
    width: int = 640
    height: int = 480
    enable_perturbations: bool = True
    
    # Command parameters
    x_vel: float = 0.0
    y_vel: float = 0.0
    yaw_vel: float = 3.14
    
    # Visualization
    save_video: bool = True
    video_path: str = "videos"
    show_contact_points: bool = True
    show_perturbation_forces: bool = True


@dataclass
class HandstandConfig:
    """Handstand-specific configuration."""
    env_name: str = 'Go1Handstand'
    energy_termination_threshold: float = 400
    energy_reward_weight: float = -0.003
    dof_acc_reward_weight: float = -2.5e-7


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config: dict, config_path: str) -> None:
    """Save configuration to JSON file."""
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def get_default_training_config(env_name: str) -> TrainingConfig:
    """Get default training configuration for environment."""
    config = TrainingConfig()
    config.env_name = env_name
    
    # Environment-specific adjustments
    if 'Humanoid' in env_name:
        config.num_timesteps = 200_000_000
        config.learning_rate = 1e-4
    elif 'Handstand' in env_name:
        config.num_timesteps = 150_000_000
        config.reward_scaling = 1e-2
    
    return config


def get_default_eval_config(env_name: str) -> EvalConfig:
    """Get default evaluation configuration for environment."""
    config = EvalConfig()
    config.env_name = env_name
    
    # Environment-specific adjustments
    if 'Humanoid' in env_name:
        config.camera = "track"
        config.width = 1280
    elif 'Handstand' in env_name:
        config.camera = "side"
        config.enable_perturbations = False
    
    return config