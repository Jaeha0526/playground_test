"""Training module for locomotion policies."""

import functools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Tuple

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from flax.training import orbax_utils
from orbax import checkpoint as ocp
from ml_collections import config_dict
import matplotlib.pyplot as plt
from IPython.display import clear_output

from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params
from ..config.config import TrainingConfig
from ..utils.plotting import plot_training_progress


class LocomotionTrainer:
    """Trainer for locomotion policies using PPO."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.training_data = {'x': [], 'y': [], 'y_err': []}
        self.times = [datetime.now()]
        self.best_reward = float('-inf')
        self.best_step = 0
        
        # Setup environment and training parameters
        self.env = registry.load(config.env_name)
        self.env_cfg = registry.get_default_config(config.env_name)
        
        # Try to get PPO params, fall back to Go1 params for custom environments
        try:
            self.ppo_params = locomotion_params.brax_ppo_config(config.env_name)
        except ValueError:
            # For custom environments, use Go1 params as template
            self.ppo_params = locomotion_params.brax_ppo_config('Go1JoystickFlatTerrain')
            print(f"Using Go1JoystickFlatTerrain PPO params for custom env: {config.env_name}")
        
        # Override PPO params with config values
        self._update_ppo_params()
        
        # Use shared timestamp for consistent naming between reward graphs and checkpoints
        timestamp = config.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.reward_graph_dir = f"reward_graphs/{config.env_name}_{timestamp}"
        
        # Setup checkpointing if specified - use same timestamp as reward graphs
        self.checkpoint_path = None
        if config.checkpoint_logdir:
            # If checkpoint_logdir doesn't already have timestamp, use the same as reward graphs
            if f"{config.env_name}_{timestamp}" in config.checkpoint_logdir:
                self.checkpoint_path = epath.Path(config.checkpoint_logdir).resolve()
            else:
                # Legacy support: if user provides custom path, use it as-is
                self.checkpoint_path = epath.Path(config.checkpoint_logdir).resolve() / config.env_name
            self.checkpoint_path.mkdir(parents=True, exist_ok=True)
            
    def _update_ppo_params(self):
        """Update PPO parameters with config values."""
        self.ppo_params.update({
            'num_timesteps': self.config.num_timesteps,
            'num_evals': self.config.num_evals,
            'episode_length': self.config.episode_length,
            'deterministic_eval': self.config.deterministic_eval,
            'normalize_observations': self.config.normalize_observations,
            'reward_scaling': self.config.reward_scaling,
            'seed': self.config.seed,
            'num_envs': self.config.num_envs,
            'batch_size': self.config.batch_size,
            'num_minibatches': self.config.num_minibatches,
            'num_updates_per_batch': self.config.num_updates_per_batch,
            'unroll_length': self.config.unroll_length,
            'learning_rate': self.config.learning_rate,
            'entropy_cost': self.config.entropy_cost,
            'discounting': self.config.discounting,
            'gae_lambda': self.config.gae_lambda,
            'clipping_epsilon': self.config.clipping_epsilon,
            'normalize_advantage': self.config.normalize_advantage,
            'action_repeat': self.config.action_repeat,
            'max_grad_norm': self.config.max_grad_norm,
        })
    
    def _progress_callback(self, num_steps: int, metrics: Dict[str, Any]):
        """Progress callback for training visualization."""
        self.times.append(datetime.now())
        self.training_data['x'].append(num_steps)
        self.training_data['y'].append(metrics["eval/episode_reward"])
        self.training_data['y_err'].append(metrics["eval/episode_reward_std"])
        
        # Update plot and save to file
        plot_training_progress(
            self.training_data['x'],
            self.training_data['y'],
            self.training_data['y_err'],
            self.config.num_timesteps,
            title=f"Training {self.config.env_name}",
            save_dir=self.reward_graph_dir
        )
        
        # Save training data to JSON file
        self._save_training_data_json(num_steps, metrics)
        
        # Save checkpoint at same time as reward plot (evaluation intervals)
        checkpoint_msg = ""
        if self.checkpoint_path:
            current_reward = metrics["eval/episode_reward"]
            is_best = self._save_checkpoint_at_evaluation(num_steps, current_reward)
            checkpoint_msg = f" | Checkpoint saved"
            if is_best:
                checkpoint_msg += " (NEW BEST!)"
        
        # Print progress
        elapsed = self.times[-1] - self.times[1] if len(self.times) > 1 else 0
        print(f"Step: {num_steps:,} | Reward: {metrics['eval/episode_reward']:.3f} ± "
              f"{metrics['eval/episode_reward_std']:.3f} | Elapsed: {elapsed}")
        print(f"Progress saved to: {self.reward_graph_dir}/{checkpoint_msg}")
    
    def _save_training_data_json(self, num_steps: int, metrics: Dict[str, Any]):
        """Save training data to JSON file."""
        from pathlib import Path
        
        # Create directory if it doesn't exist
        save_path = Path(self.reward_graph_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare training progress data
        current_time = datetime.now()
        elapsed_total = current_time - self.times[0]
        elapsed_train = current_time - self.times[1] if len(self.times) > 1 else elapsed_total
        
        training_progress = {
            "config": {
                "env_name": self.config.env_name,
                "num_timesteps": self.config.num_timesteps,
                "num_envs": self.config.num_envs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
            },
            "training_data": {
                "steps": self.training_data['x'],
                "rewards": self.training_data['y'],
                "reward_stds": self.training_data['y_err'],
            },
            "current_metrics": {
                "step": num_steps,
                "reward_mean": metrics["eval/episode_reward"],
                "reward_std": metrics["eval/episode_reward_std"],
                "progress_percent": (num_steps / self.config.num_timesteps) * 100,
            },
            "timing": {
                "started_at": self.times[0].isoformat(),
                "current_time": current_time.isoformat(),
                "elapsed_total_seconds": elapsed_total.total_seconds(),
                "elapsed_training_seconds": elapsed_train.total_seconds(),
            },
            "full_metrics": dict(metrics)  # Save all available metrics
        }
        
        # Save to JSON file
        json_path = save_path / "training_progress.json"
        with open(json_path, 'w') as f:
            json.dump(training_progress, f, indent=2, default=str)
    
    def _save_checkpoint_at_evaluation(self, num_steps: int, current_reward: float) -> bool:
        """Save checkpoint at evaluation intervals (synced with reward plots)."""
        if not self.checkpoint_path or not hasattr(self, '_current_params'):
            return False
        
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(self._current_params)
        
        # Always save regular checkpoint
        path = self.checkpoint_path / f"{num_steps}"
        orbax_checkpointer.save(path, self._current_params, force=True, save_args=save_args)
        
        # Check if this is the best reward so far
        is_best = current_reward > self.best_reward
        if is_best:
            self.best_reward = current_reward
            self.best_step = num_steps
            
            # Save best checkpoint
            best_path = self.checkpoint_path / "best"
            orbax_checkpointer.save(best_path, self._current_params, force=True, save_args=save_args)
            
            # Update best info in JSON
            best_info = {
                "best_reward": float(current_reward),
                "best_step": num_steps,
                "saved_at": datetime.now().isoformat()
            }
            best_json_path = self.checkpoint_path / "best_info.json"
            with open(best_json_path, 'w') as f:
                json.dump(best_info, f, indent=2)
        
        return is_best
    
    def _policy_params_callback(self, current_step: int, make_policy: Callable, params: Any):
        """Store current policy parameters for checkpoint saving."""
        # Store params for use in progress callback
        self._current_params = params
    
    def train(self, 
              restore_checkpoint_path: Optional[str] = None,
              custom_env_config: Optional[Dict] = None) -> Tuple[Callable, Any, Dict]:
        """Train the locomotion policy.
        
        Args:
            restore_checkpoint_path: Path to restore training from checkpoint
            custom_env_config: Custom environment configuration
            
        Returns:
            Tuple of (make_inference_fn, params, metrics)
        """
        print(f"Starting training for {self.config.env_name}")
        print(f"Total timesteps: {self.config.num_timesteps:,}")
        
        # Setup environment with custom config if provided
        env_cfg = custom_env_config if custom_env_config else self.env_cfg
        if custom_env_config:
            for key, value in custom_env_config.items():
                if hasattr(env_cfg, key):
                    setattr(env_cfg, key, value)
        
        # Save environment config if checkpointing
        if self.checkpoint_path:
            with open(self.checkpoint_path / "config.json", "w") as fp:
                json.dump(env_cfg.to_dict(), fp, indent=4)
        
        # Get domain randomizer
        randomizer = registry.get_domain_randomizer(self.config.env_name)
        
        # Setup network factory
        ppo_training_params = dict(self.ppo_params)
        network_factory = ppo_networks.make_ppo_networks
        if "network_factory" in self.ppo_params:
            del ppo_training_params["network_factory"]
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks,
                **self.ppo_params.network_factory
            )
        
        # Setup training function
        train_kwargs = {
            **ppo_training_params,
            'network_factory': network_factory,
            'randomization_fn': randomizer,
            'progress_fn': self._progress_callback,
        }
        
        if self.checkpoint_path:
            train_kwargs['policy_params_fn'] = self._policy_params_callback
        
        if restore_checkpoint_path:
            train_kwargs['restore_checkpoint_path'] = restore_checkpoint_path
            print(f"Restoring from checkpoint: {restore_checkpoint_path}")
        
        train_fn = functools.partial(ppo.train, **train_kwargs)
        
        # Reset training data
        self.training_data = {'x': [], 'y': [], 'y_err': []}
        self.times = [datetime.now()]
        
        # Train the policy
        make_inference_fn, params, metrics = train_fn(
            environment=registry.load(self.config.env_name, config=env_cfg),
            eval_env=registry.load(self.config.env_name, config=env_cfg),
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
        
        print(f"Training completed!")
        print(f"Time to JIT: {self.times[1] - self.times[0]}")
        print(f"Time to train: {self.times[-1] - self.times[1]}")
        
        return make_inference_fn, params, metrics
    
    def finetune(self, 
                 checkpoint_path: str,
                 custom_env_config: Dict,
                 num_timesteps: Optional[int] = None) -> Tuple[Callable, Any, Dict]:
        """Finetune a policy from checkpoint with new environment config.
        
        Args:
            checkpoint_path: Path to checkpoint to restore from
            custom_env_config: New environment configuration for finetuning
            num_timesteps: Override number of timesteps for finetuning
            
        Returns:
            Tuple of (make_inference_fn, params, metrics)
        """
        # Temporarily adjust timesteps for finetuning if specified
        original_timesteps = self.config.num_timesteps
        if num_timesteps:
            self.config.num_timesteps = num_timesteps
            self._update_ppo_params()
        
        try:
            result = self.train(
                restore_checkpoint_path=checkpoint_path,
                custom_env_config=custom_env_config
            )
        finally:
            # Restore original timesteps
            self.config.num_timesteps = original_timesteps
            self._update_ppo_params()
        
        return result
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the latest checkpoint path."""
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            return None
        
        checkpoints = [p for p in self.checkpoint_path.glob("*") if p.is_dir() and p.name.isdigit()]
        if not checkpoints:
            return None
        
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.name))
        return str(checkpoints[-1])
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get the best checkpoint path."""
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            return None
        
        best_path = self.checkpoint_path / "best"
        if best_path.exists():
            return str(best_path)
        return None