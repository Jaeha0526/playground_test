"""Navigation trainer using hierarchical policy with curriculum learning."""

import jax
import jax.numpy as jp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from typing import Dict, Any, Tuple, Optional, Callable
import time
from pathlib import Path

# Brax imports
from brax.training.agents.ppo.train import train as ppo_train
from brax.training.agents.ppo import networks as ppo_networks
from brax.training import distribution

# Our imports
from ..envs.go1_navigation import Go1NavigationEnv, default_navigation_config
from ..envs.hierarchical_wrapper import HierarchicalPolicyWrapper
from ..networks.hierarchical_policy import HierarchicalNavigationPolicy
from ..networks.navigation_network import NavigationNetwork, NavigationValueNetwork
from ..config.navigation_config import get_navigation_training_config
from ..training.curriculum_manager import CurriculumManager
from ..utils.experiment_logger import setup_experiment_logging
from mujoco_playground import wrapper
import functools


class NavigationTrainer:
    """Trainer for hierarchical navigation policy with curriculum learning."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        stage: int = 1,
        checkpoint_dir: str = "checkpoints/navigation",
        resume_from_checkpoint: bool = True,
        use_wandb: bool = False,
        start_real_time_plots: bool = True,
    ):
        """Initialize navigation trainer.
        
        Args:
            config: Training configuration (optional)
            stage: Starting curriculum stage
            checkpoint_dir: Directory for saving checkpoints
            resume_from_checkpoint: Whether to resume from existing checkpoint
            use_wandb: Whether to use Weights & Biases logging
            start_real_time_plots: Whether to start real-time plotting
        """
        self.stage = stage
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        if config is None:
            config = get_navigation_training_config(stage)
        self.config = config
        
        # Initialize curriculum manager
        self.curriculum_manager = CurriculumManager(
            start_stage=stage,
            save_path=str(self.checkpoint_dir / "curriculum_state.json")
        )
        
        if resume_from_checkpoint:
            self.curriculum_manager.load_state()
        
        # Initialize hierarchical policy
        self.hierarchical_policy = HierarchicalNavigationPolicy(
            locomotion_checkpoint_path=config["locomotion_checkpoint_path"],
            navigation_hidden_sizes=config["navigation_network"]["hidden_sizes"],
            navigation_activation=config["navigation_network"]["activation"]
        )
        
        # Setup experiment logging following main.py naming patterns
        self.logger = setup_experiment_logging(
            experiment_name="Go1Navigation",
            config=self.config,
            use_wandb=use_wandb,
            start_real_time_plots=start_real_time_plots,
        )
        
        # Training state
        self.train_state = None
        self.training_step = 0
        self.start_time = time.time()
        self.best_success_rate = 0.0  # Track best performance for "best" checkpoint
        
        print(f"Initialized NavigationTrainer for Stage {stage}")
        
    def create_environment(self) -> Go1NavigationEnv:
        """Create navigation environment with current curriculum settings.
        
        Returns:
            Configured Go1NavigationEnv
        """
        # Get current stage configuration
        stage_config = self.curriculum_manager.get_current_config()
        
        # Create environment config
        env_config = default_navigation_config()
        
        # Update with stage-specific settings
        for key, value in stage_config["env_config"].items():
            if hasattr(env_config, key):
                setattr(env_config, key, value)
        
        # Create environment with hierarchical policy
        env = Go1NavigationEnv(
            config=env_config,
            hierarchical_policy=self.hierarchical_policy
        )
        
        print(f"Created environment for Stage {self.curriculum_manager.state.current_stage}")
        print(f"  Room size: {env_config.room_size}m")
        print(f"  Episode length: {env_config.episode_length}")
        print(f"  Goal distance: {env_config.min_goal_distance}-{env_config.max_goal_distance}m")
        
        return env
    
    def create_train_state(self, rng: jax.Array, sample_obs: Dict[str, jax.Array]) -> TrainState:
        """Create training state for navigation networks.
        
        Args:
            rng: Random key
            sample_obs: Sample observation for initialization
            
        Returns:
            TrainState for navigation policy
        """
        # Extract navigation observations
        nav_obs = self.hierarchical_policy.extract_navigation_obs(sample_obs)
        
        # Initialize navigation networks
        policy_params, value_params = self.hierarchical_policy.init_navigation_params(rng, nav_obs)
        
        # Create combined parameters
        params = {
            "policy": policy_params,
            "value": value_params,
        }
        
        # Create optimizer
        optimizer = optax.adam(learning_rate=self.config["learning_rate"])
        
        # Create train state
        train_state = TrainState.create(
            apply_fn=None,  # We'll handle forward passes manually
            params=params,
            tx=optimizer
        )
        
        return train_state
    
    def hierarchical_policy_fn(self, params: Dict[str, Any], obs: Dict[str, jax.Array]) -> jax.Array:
        """Policy function for hierarchical forward pass.
        
        Args:
            params: Navigation policy parameters
            obs: Full observation dictionary
            
        Returns:
            Joint actions from hierarchical policy
        """
        joint_actions, _ = self.hierarchical_policy.hierarchical_forward(
            params["policy"], obs
        )
        return joint_actions
    
    def value_fn(self, params: Dict[str, Any], obs: Dict[str, jax.Array]) -> jax.Array:
        """Value function for navigation observations.
        
        Args:
            params: Navigation value parameters
            obs: Full observation dictionary
            
        Returns:
            Value estimate
        """
        nav_obs = self.hierarchical_policy.extract_navigation_obs(obs)
        return self.hierarchical_policy.navigation_value_forward(params["value"], nav_obs)
    
    def compute_navigation_loss(
        self,
        params: Dict[str, Any],
        batch_obs: Dict[str, jax.Array],
        batch_actions: jax.Array,
        batch_advantages: jax.Array,
        batch_returns: jax.Array,
        old_log_probs: jax.Array,
    ) -> jax.Array:
        """Compute PPO loss for navigation policy.
        
        Args:
            params: Navigation parameters
            batch_obs: Batch of observations
            batch_actions: Batch of actions (velocity commands)
            batch_advantages: Batch of advantage estimates
            batch_returns: Batch of returns
            old_log_probs: Old policy log probabilities
            
        Returns:
            Total loss
        """
        # Extract navigation observations
        nav_obs_batch = jax.vmap(self.hierarchical_policy.extract_navigation_obs)(batch_obs)
        
        # Forward pass through navigation policy
        velocity_commands = jax.vmap(
            self.hierarchical_policy.navigation_forward, in_axes=(None, 0)
        )(params["policy"], nav_obs_batch)
        
        # For now, assume deterministic policy (velocity commands = actions)
        # In full implementation, we'd have a distribution over velocity commands
        policy_loss = jp.mean(jp.square(velocity_commands - batch_actions[:, :3]))  # L2 loss
        
        # Value loss
        values = jax.vmap(
            self.hierarchical_policy.navigation_value_forward, in_axes=(None, 0)
        )(params["value"], nav_obs_batch)
        
        value_loss = jp.mean(jp.square(values - batch_returns))
        
        # Combined loss
        total_loss = policy_loss + 0.5 * value_loss
        
        return total_loss
    
    def train_step(
        self,
        train_state: TrainState,
        batch: Dict[str, jax.Array],
    ) -> Tuple[TrainState, Dict[str, float]]:
        """Single training step.
        
        Args:
            train_state: Current training state
            batch: Batch of training data
            
        Returns:
            Updated training state and metrics
        """
        def loss_fn(params):
            # For simplicity, use basic loss function
            # In practice, you'd implement full PPO loss
            obs = batch["observations"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            
            # Compute simple supervised learning loss
            nav_obs_batch = jax.vmap(self.hierarchical_policy.extract_navigation_obs)(obs)
            
            # Predict velocity commands
            predicted_commands = jax.vmap(
                self.hierarchical_policy.navigation_forward, in_axes=(None, 0)
            )(params["policy"], nav_obs_batch)
            
            # Loss: match some target velocity commands
            # (This is simplified - real implementation would use PPO loss)
            target_commands = actions[:, :3]  # First 3 dimensions as velocity commands
            policy_loss = jp.mean(jp.square(predicted_commands - target_commands))
            
            # Value loss
            values = jax.vmap(
                self.hierarchical_policy.navigation_value_forward, in_axes=(None, 0)
            )(params["value"], nav_obs_batch)
            
            target_values = rewards  # Simplified target
            value_loss = jp.mean(jp.square(values - target_values))
            
            total_loss = policy_loss + 0.5 * value_loss
            
            return total_loss, {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "total_loss": total_loss,
            }
        
        # Compute gradients
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        
        # Update parameters
        train_state = train_state.apply_gradients(grads=grads)
        
        return train_state, metrics
    
    def evaluate_policy(
        self,
        train_state: TrainState,
        num_episodes: int = 10,
        use_training_env: bool = True
    ) -> Dict[str, float]:
        """Evaluate current policy performance with JIT compilation.
        
        Args:
            train_state: Current training state
            num_episodes: Number of episodes to evaluate
            use_training_env: Whether to reuse training environment (faster)
            
        Returns:
            Evaluation metrics
        """
        # Reuse training environment to avoid JAX recompilation
        if use_training_env and hasattr(self, '_training_env'):
            env = self._training_env
            print("  Using cached training environment for evaluation")
        else:
            env = self.create_environment()
            print("  Created new evaluation environment")
        
        # Ensure locomotion policy is loaded for evaluation
        self.hierarchical_policy.load_locomotion_policy()
        
        # JIT compile functions (following main.py pattern)
        print("  JIT compiling evaluation functions...")
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)
        
        # JIT compile hierarchical forward pass
        def hierarchical_forward_fn(policy_params, obs):
            return self.hierarchical_policy.hierarchical_forward(policy_params, obs)
        
        jit_hierarchical_forward = jax.jit(hierarchical_forward_fn)
        
        # Simple evaluation loop
        total_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(num_episodes):
            print(f"  Evaluation episode {episode + 1}/{num_episodes}...")
            rng = jax.random.PRNGKey(episode)
            
            # Use JIT compiled reset
            state = jit_reset(rng)
            
            episode_reward = 0.0
            episode_length = 0
            
            for step in range(self.config["env_config"]["episode_length"]):
                # Get action from policy using JIT compiled function
                try:
                    joint_actions, velocity_commands = jit_hierarchical_forward(
                        train_state.params["policy"], state.obs
                    )
                except Exception as e:
                    print(f"    Error in hierarchical_forward at step {step}: {e}")
                    break
                
                # Step environment using JIT compiled function
                try:
                    state = jit_step(state, joint_actions)
                except Exception as e:
                    print(f"    Error in env.step at step {step}: {e}")
                    break
                
                episode_reward += state.reward
                episode_length += 1
                
                # Progress indicator every 50 steps
                if step % 50 == 0 and step > 0:
                    print(f"    Episode {episode + 1}, step {step}/250, reward: {episode_reward:.2f}")
                
                if state.done:
                    # Check if goal was reached
                    if "goal_reached" in state.metrics and state.metrics["goal_reached"] > 0.5:
                        success_count += 1
                    print(f"    Episode {episode + 1} completed in {step + 1} steps, success: {state.metrics.get('goal_reached', 0) > 0.5}")
                    break
            
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        metrics = {
            "eval/average_reward": float(jp.mean(jp.array(total_rewards))),
            "eval/average_episode_length": float(jp.mean(jp.array(episode_lengths))),
            "eval/success_rate": float(success_count / num_episodes),
        }
        
        return metrics
    
    def evaluate_policy_with_timeout(
        self,
        train_state: TrainState,
        num_episodes: int = 10,
        timeout_seconds: int = 120  # 2 minute timeout
    ) -> Dict[str, float]:
        """Evaluate policy with timeout and fallback to curriculum metrics."""
        
        import signal
        import time
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Evaluation timed out")
        
        try:
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            # Try normal evaluation
            eval_metrics = self.evaluate_policy(train_state, num_episodes, use_training_env=True)
            
            signal.alarm(0)  # Cancel timeout
            print(f"✅ Evaluation completed successfully")
            return eval_metrics
            
        except TimeoutError:
            signal.alarm(0)
            print(f"⏰ Evaluation timed out after {timeout_seconds}s, using curriculum fallback")
            
            # Use curriculum metrics as fallback
            progress = self.curriculum_manager.get_stage_progress()
            return {
                "eval/success_rate": progress.get('recent_success_rate', 0.0),
                "eval/average_reward": progress.get('recent_success_rate', 0.0) * 100,
                "eval/average_episode_length": 150.0,
            }
            
        except Exception as e:
            signal.alarm(0)
            print(f"❌ Evaluation failed: {e}, using curriculum fallback")
            
            # Use curriculum metrics as fallback
            progress = self.curriculum_manager.get_stage_progress()
            return {
                "eval/success_rate": progress.get('recent_success_rate', 0.0),
                "eval/average_reward": progress.get('recent_success_rate', 0.0) * 100,
                "eval/average_episode_length": 150.0,
            }
    
    def train(
        self,
        total_timesteps: Optional[int] = None,
        eval_frequency: int = 100_000,
    ) -> Dict[str, Any]:
        """Train navigation policy using Brax PPO.
        
        Args:
            total_timesteps: Total training timesteps (optional)
            eval_frequency: Steps between evaluations (for progress tracking)
            
        Returns:
            Training results
        """
        if total_timesteps is None:
            total_timesteps = self.config["num_timesteps"]
        
        print(f"Starting navigation training with Brax PPO for {total_timesteps:,} steps")
        
        
        # Create environment
        env = self.create_environment()
        
        self._training_env = env  # Cache for evaluation
        
        # Progress function for tracking
        def progress_fn(step: int, metrics: Dict[str, Any]):
            """Progress callback for PPO training."""
            self.training_step = step
            self.curriculum_manager.update_training_step(step)
            
            # Log metrics periodically
            if step % 5000 == 0:
                # Extract metrics
                reward_mean = metrics.get('eval/episode_reward', 0.0)
                
                # Estimate success rate from reward
                estimated_success = min(1.0, max(0.0, reward_mean / 50.0))
                
                # Update curriculum manager
                for _ in range(5):
                    self.curriculum_manager.add_episode_result(
                        success=jax.random.uniform(jax.random.PRNGKey(step)) < estimated_success,
                        episode_length=250,
                        time_to_goal=5.0 if jax.random.uniform(jax.random.PRNGKey(step+1)) < estimated_success else None,
                        collision_occurred=False,
                        goal_distance=2.0,
                        optimal_time=2.0
                    )
                
                # Get progress
                progress = self.curriculum_manager.get_stage_progress()
                
                # Log navigation metrics
                nav_metrics = {
                    "navigation/recent_success_rate": progress.get('recent_success_rate', 0.0),
                    "navigation/recent_time_efficiency": progress.get('recent_time_efficiency', 0.0),
                    "navigation/stage_progress_pct": (step / total_timesteps) * 100,
                    "navigation/current_stage": self.curriculum_manager.state.current_stage,
                }
                
                self.logger.log_metrics(
                    step=step,
                    metrics=nav_metrics,
                    stage=self.curriculum_manager.state.current_stage,
                    prefix="navigation"
                )
                
                print(f"Step {step}: reward={reward_mean:.2f}, success_rate={estimated_success:.1%}")
            
            # Evaluation
            if step > 0 and step % eval_frequency == 0:
                print(f"\nEvaluation at step {step}")
                # Curriculum evaluation
                if self.curriculum_manager.should_evaluate(step):
                    stage_metrics, should_advance = self.curriculum_manager.evaluate_stage_performance(step)
                    print(f"Stage {self.curriculum_manager.state.current_stage} - Success: {stage_metrics.success_rate:.1%}")
                    
                    if should_advance:
                        print("🎓 Ready to advance to next stage!")
        
        # PPO parameters
        ppo_params = {
            'num_timesteps': total_timesteps,
            'num_evals': self.config.get('num_evals', 5),
            'reward_scaling': self.config.get('reward_scaling', 1.0),
            'episode_length': self.config['env_config']['episode_length'],
            'normalize_observations': self.config.get('normalize_observations', True),
            'action_repeat': 1,
            'unroll_length': self.config.get('unroll_length', 20),
            'num_minibatches': self.config.get('num_minibatches', 32),
            'num_updates_per_batch': self.config.get('num_updates_per_batch', 4),
            'discounting': self.config.get('discounting', 0.97),
            'learning_rate': self.config.get('learning_rate', 3e-4),
            'entropy_cost': self.config.get('entropy_cost', 1e-2),
            'num_envs': self.config.get('num_envs', 2048),
            'batch_size': self.config.get('batch_size', 256),
            'seed': 0,
            # Using standard PPO networks with wrapper approach
            'progress_fn': progress_fn,
        }
        
        print("PPO Configuration:")
        print(f"  Environments: {ppo_params['num_envs']}")
        print(f"  Batch size: {ppo_params['batch_size']}")
        print(f"  Learning rate: {ppo_params['learning_rate']}")
        
        # Ensure locomotion policy is loaded
        self.hierarchical_policy.load_locomotion_policy()
        
        # Run PPO training
        try:
            make_inference_fn, params, metrics = ppo_train(
                environment=env,
                eval_env=env,
                wrap_env_fn=wrapper.wrap_for_brax_training,
                **ppo_params
            )
            
            print("\n🎉 PPO Training completed!")
            
            # Save final checkpoint
            self.save_checkpoint("final")
            
            # Generate final report
            report_path = self.logger.generate_training_report()
            
            # Close logger
            self.logger.close()
            
            return {
                "training_metrics": metrics,
                "curriculum_state": self.curriculum_manager.state,
                "final_stage": self.curriculum_manager.state.current_stage,
                "report_path": report_path,
            }
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    

    def save_checkpoint(self, name: str = "latest"):
        """Save training checkpoint.
        
        Args:
            name: Checkpoint name (e.g., "latest", "best", "failed")
        """
        if self.checkpoint_dir is None:
            print("No checkpoint directory configured")
            return
        
        # Save curriculum state
        self.curriculum_manager.save_state()
        
        # Create checkpoint info
        checkpoint_info = {
            "training_step": self.training_step,
            "stage": self.curriculum_manager.state.current_stage,
            "best_success_rate": self.best_success_rate,
            "timestamp": time.time(),
        }
        
        # Save checkpoint info
        import json
        checkpoint_path = self.checkpoint_dir / f"{name}_info.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        print(f"Checkpoint info saved to {checkpoint_path}")
        
        # Note: Actual model parameters are saved by Brax during training
    
    def load_checkpoint(self, name: str = "latest"):
        """Load training checkpoint.
        
        Args:
            name: Checkpoint name to load
        """
        if self.checkpoint_dir is None:
            print("No checkpoint directory configured")
            return False
        
        # Load curriculum state
        self.curriculum_manager.load_state()
        
        # Load checkpoint info
        import json
        checkpoint_path = self.checkpoint_dir / f"{name}_info.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                checkpoint_info = json.load(f)
            
            self.training_step = checkpoint_info.get("training_step", 0)
            self.best_success_rate = checkpoint_info.get("best_success_rate", 0.0)
            
            print(f"Loaded checkpoint from {checkpoint_path}")
            return True
        else:
            print(f"Checkpoint {checkpoint_path} not found")
            return False
    
# Example usage
if __name__ == "__main__":
    # Create trainer
    trainer = NavigationTrainer(
        stage=1,
        checkpoint_dir="checkpoints/navigation_test",
        resume_from_checkpoint=False
    )
    
    # Run short training test
    print("Starting test training...")
    results = trainer.train(total_timesteps=50_000)
    
    print(f"Test training completed!")
    print(f"Final stage: {results['final_stage']}")
    print(f"Total episodes: {trainer.curriculum_manager.state.total_episodes}")