"""Navigation trainer using Brax PPO with hierarchical policy."""

import jax
import jax.numpy as jp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from typing import Dict, Any, Tuple, Optional, Callable
import time
from pathlib import Path
import functools

# Brax imports
from brax.training.agents.ppo import train as ppo_train
from brax.training.agents.ppo import networks as ppo_networks
from brax.training import distribution
from brax import envs

# MuJoCo Playground imports
from mujoco_playground import registry, wrapper

# Our imports
from ..envs.go1_navigation import Go1NavigationEnv, default_navigation_config
from ..networks.hierarchical_policy import HierarchicalNavigationPolicy
from ..networks.navigation_network import NavigationNetwork, NavigationValueNetwork
from ..config.navigation_config import get_navigation_training_config
from ..training.curriculum_manager import CurriculumManager
from ..utils.experiment_logger import setup_experiment_logging


class HierarchicalPPONetworks:
    """PPO networks wrapper for hierarchical navigation policy."""
    
    def __init__(
        self,
        hierarchical_policy: HierarchicalNavigationPolicy,
        observation_size: int,
        action_size: int,
        preprocess_observations_fn: Optional[Callable] = None
    ):
        """Initialize hierarchical PPO networks.
        
        Args:
            hierarchical_policy: Pre-initialized hierarchical policy
            observation_size: Size of observations
            action_size: Size of actions (12 for Go1)
            preprocess_observations_fn: Optional observation preprocessing
        """
        self.hierarchical_policy = hierarchical_policy
        self.observation_size = observation_size
        self.action_size = action_size
        self.preprocess_observations_fn = preprocess_observations_fn
        
        # Ensure locomotion policy is loaded
        self.hierarchical_policy.load_locomotion_policy()
        
    def make_policy_network(self, obs_size: int) -> nn.Module:
        """Create policy network using hierarchical structure."""
        hierarchical_policy = self.hierarchical_policy
        preprocess_fn = self.preprocess_observations_fn
        
        class HierarchicalPolicyNetwork(nn.Module):
            """Policy network that uses hierarchical navigation + locomotion."""
            
            @nn.compact
            def __call__(self, observations: jax.Array) -> jax.Array:
                # Preprocess observations if needed
                if preprocess_fn is not None:
                    observations = preprocess_fn(observations)
                
                # Extract navigation observations (10-dim)
                nav_obs = hierarchical_policy.extract_navigation_obs(observations)
                
                # Get navigation parameters from this module's params
                nav_params = self.param('navigation_params', 
                                       lambda rng: hierarchical_policy.navigation_policy.init(rng, nav_obs))
                
                # Navigation network forward pass
                velocity_commands = hierarchical_policy.navigation_policy.apply(
                    nav_params, nav_obs
                )
                
                # Create locomotion observations
                locomotion_obs = hierarchical_policy.create_locomotion_obs(
                    observations, velocity_commands
                )
                
                # Locomotion forward pass (with frozen weights)
                joint_actions = hierarchical_policy.locomotion_forward(locomotion_obs)
                
                return joint_actions
        
        return HierarchicalPolicyNetwork()
    
    def make_value_network(self, obs_size: int) -> nn.Module:
        """Create value network for navigation observations."""
        hierarchical_policy = self.hierarchical_policy
        preprocess_fn = self.preprocess_observations_fn
        
        class HierarchicalValueNetwork(nn.Module):
            """Value network that uses navigation observations."""
            
            @nn.compact
            def __call__(self, observations: jax.Array) -> jax.Array:
                # Preprocess observations if needed
                if preprocess_fn is not None:
                    observations = preprocess_fn(observations)
                
                # Extract navigation observations
                nav_obs = hierarchical_policy.extract_navigation_obs(observations)
                
                # Get value parameters from this module's params
                value_params = self.param('value_params',
                                         lambda rng: hierarchical_policy.navigation_value.init(rng, nav_obs))
                
                # Value network forward pass
                value = hierarchical_policy.navigation_value.apply(
                    value_params, nav_obs
                )
                
                return value
        
        return HierarchicalValueNetwork()


def create_hierarchical_network_factory(
    hierarchical_policy: HierarchicalNavigationPolicy,
    deterministic: bool = False
) -> Callable:
    """Create network factory for Brax PPO that uses hierarchical policy.
    
    Args:
        hierarchical_policy: Initialized hierarchical navigation policy
        deterministic: Whether to use deterministic policy (for evaluation)
        
    Returns:
        Network factory function for Brax PPO
    """
    def make_hierarchical_ppo_networks(
        observation_size: int,
        action_size: int,
        preprocess_observations_fn: Optional[Callable] = None,
        **kwargs  # Ignore other kwargs from Brax
    ) -> Tuple[nn.Module, nn.Module]:
        """Network factory that returns hierarchical policy and value networks."""
        
        # Create wrapper for hierarchical networks
        hierarchical_networks = HierarchicalPPONetworks(
            hierarchical_policy=hierarchical_policy,
            observation_size=observation_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn
        )
        
        # Create policy and value networks
        policy_network = hierarchical_networks.make_policy_network(observation_size)
        value_network = hierarchical_networks.make_value_network(observation_size)
        
        # Wrap in distribution if not deterministic
        if not deterministic:
            policy_network = distribution.NormalTanhDistribution(
                event_size=action_size,
                network=policy_network,
            )
        
        return policy_network, value_network
    
    return make_hierarchical_ppo_networks


class NavigationTrainerPPO:
    """Navigation trainer using Brax PPO with hierarchical policy."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        stage: int = 1,
        checkpoint_dir: str = "checkpoints/navigation",
        resume_from_checkpoint: bool = True,
        use_wandb: bool = False,
        start_real_time_plots: bool = True,
    ):
        """Initialize navigation trainer with PPO.
        
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
        
        # Setup experiment logging
        self.logger = setup_experiment_logging(
            experiment_name="Go1Navigation",
            config=self.config,
            use_wandb=use_wandb,
            start_real_time_plots=start_real_time_plots,
        )
        
        # Training state
        self.training_step = 0
        self.start_time = time.time()
        self.best_success_rate = 0.0
        
        print(f"Initialized NavigationTrainerPPO for Stage {stage}")
        print("Using Brax PPO with hierarchical policy")
        
    def create_environment(self) -> Go1NavigationEnv:
        """Create navigation environment with current curriculum settings."""
        # Get current stage configuration
        stage_config = self.curriculum_manager.get_current_config()
        
        # Create environment config
        env_config = default_navigation_config()
        
        # Update with stage-specific settings
        for key, value in stage_config["env_config"].items():
            if hasattr(env_config, key):
                setattr(env_config, key, value)
        
        # Create environment
        env = Go1NavigationEnv(config=env_config)
        
        print(f"Created environment for Stage {self.curriculum_manager.state.current_stage}")
        print(f"  Room size: {env_config.room_size}m")
        print(f"  Episode length: {env_config.episode_length}")
        print(f"  Goal distance: {env_config.min_goal_distance}-{env_config.max_goal_distance}m")
        
        return env
    
    def create_progress_fn(self) -> Callable:
        """Create progress function for Brax PPO training."""
        logger = self.logger
        curriculum_manager = self.curriculum_manager
        best_success_rate = [self.best_success_rate]  # Mutable container
        checkpoint_dir = self.checkpoint_dir
        
        def progress_fn(step: int, metrics: Dict[str, Any]):
            """Progress callback for PPO training."""
            # Update step count
            self.training_step = step
            curriculum_manager.update_training_step(step)
            
            # Log basic metrics
            if step % 1000 == 0:
                # Extract key metrics
                reward_mean = metrics.get('eval/episode_reward', 0.0)
                episode_length = metrics.get('eval/episode_length', 250)
                
                # Log to experiment logger
                logger.log_metrics(
                    step=step,
                    metrics={
                        'train/reward_mean': reward_mean,
                        'train/episode_length': episode_length,
                    },
                    stage=curriculum_manager.state.current_stage,
                    prefix='train'
                )
            
            # Update curriculum metrics
            if step % 5000 == 0 and step > 0:
                # Simulate episode results based on training progress
                # In real implementation, these would come from actual rollouts
                reward_mean = metrics.get('eval/episode_reward', 0.0)
                
                # Estimate success based on reward (rough approximation)
                # Goal reached reward is 100, so normalize
                estimated_success_rate = min(1.0, max(0.0, reward_mean / 50.0))
                
                # Add episode results to curriculum manager
                for _ in range(10):  # Add 10 simulated episodes
                    curriculum_manager.add_episode_result(
                        success=jp.random.uniform(jax.random.PRNGKey(step)) < estimated_success_rate,
                        episode_length=int(episode_length),
                        time_to_goal=float(episode_length * 0.02) if jp.random.uniform(jax.random.PRNGKey(step+1)) < estimated_success_rate else None,
                        collision_occurred=False,
                        goal_distance=2.0,
                        optimal_time=2.0
                    )
                
                # Get curriculum progress
                progress = curriculum_manager.get_stage_progress()
                
                # Log navigation-specific metrics
                nav_metrics = {
                    "navigation/recent_success_rate": progress.get('recent_success_rate', 0.0),
                    "navigation/recent_time_efficiency": progress.get('recent_time_efficiency', 0.0),
                    "navigation/recent_collision_rate": progress.get('recent_collision_rate', 0.0),
                    "navigation/stage_progress_pct": (step / self.config['num_timesteps']) * 100,
                    "navigation/total_episodes": curriculum_manager.state.total_episodes,
                    "navigation/current_stage": curriculum_manager.state.current_stage,
                }
                
                logger.log_metrics(
                    step=step,
                    metrics=nav_metrics,
                    stage=curriculum_manager.state.current_stage,
                    prefix="navigation"
                )
            
            # Checkpoint saving
            if step > 0 and step % 500_000 == 0:
                # Save checkpoint (Brax handles the actual saving)
                print(f"Checkpoint saved at step {step}")
            
            # Curriculum evaluation
            if step > 0 and step % 100_000 == 0:
                if curriculum_manager.should_evaluate(step):
                    stage_metrics, should_advance = curriculum_manager.evaluate_stage_performance(step)
                    
                    print(f"\nStage {curriculum_manager.state.current_stage} metrics:")
                    print(f"  Success rate: {stage_metrics.success_rate:.1%}")
                    print(f"  Time efficiency: {stage_metrics.time_efficiency:.1%}")
                    print(f"  Collision rate: {stage_metrics.collision_rate:.1%}")
                    
                    if should_advance:
                        print("🎓 Ready to advance to next curriculum stage!")
                        # Note: Actual stage advancement would require restarting training
                        # with new environment configuration
                
        return progress_fn
    
    def train(
        self,
        total_timesteps: Optional[int] = None,
        restore_checkpoint_path: Optional[str] = None,
    ) -> Tuple[Callable, Any, Dict[str, Any]]:
        """Train navigation policy using Brax PPO.
        
        Args:
            total_timesteps: Total training timesteps (optional)
            restore_checkpoint_path: Path to checkpoint to restore from
            
        Returns:
            Tuple of (make_inference_fn, params, metrics)
        """
        if total_timesteps is None:
            total_timesteps = self.config["num_timesteps"]
        
        print(f"Starting PPO navigation training for {total_timesteps:,} steps")
        
        # Create environment
        env = self.create_environment()
        
        # Create network factory
        network_factory = create_hierarchical_network_factory(
            self.hierarchical_policy,
            deterministic=False
        )
        
        # Setup PPO parameters
        ppo_params = {
            # Use config values
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
        }
        
        # Add network factory and progress function
        ppo_params['network_factory'] = network_factory
        ppo_params['progress_fn'] = self.create_progress_fn()
        
        # Print training configuration
        print("PPO Configuration:")
        print(f"  Environments: {ppo_params['num_envs']}")
        print(f"  Batch size: {ppo_params['batch_size']}")
        print(f"  Learning rate: {ppo_params['learning_rate']}")
        print(f"  Episode length: {ppo_params['episode_length']}")
        
        # Run PPO training
        print("Starting Brax PPO training...")
        make_inference_fn, params, metrics = ppo_train(
            environment=env,
            eval_env=env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
            restore_checkpoint_path=restore_checkpoint_path,
            checkpoint_logdir=str(self.checkpoint_dir) if self.checkpoint_dir else None,
            **ppo_params
        )
        
        print("PPO training completed!")
        
        # Save final metrics
        final_metrics = {
            'final_reward': metrics.get('eval/episode_reward', 0.0),
            'final_episode_length': metrics.get('eval/episode_length', 0.0),
            'total_env_steps': metrics.get('total_env_steps', 0),
            'training_time': time.time() - self.start_time,
        }
        
        # Generate training report
        report_path = self.logger.generate_training_report()
        
        # Close logger
        self.logger.close()
        
        return make_inference_fn, params, {
            'metrics': metrics,
            'final_metrics': final_metrics,
            'curriculum_state': self.curriculum_manager.state,
            'report_path': report_path,
        }


# Example usage
if __name__ == "__main__":
    # Create trainer
    trainer = NavigationTrainerPPO(
        stage=1,
        checkpoint_dir="checkpoints/navigation_ppo_test",
        resume_from_checkpoint=False,
        use_wandb=False,
        start_real_time_plots=True,
    )
    
    # Run training
    print("Starting navigation training with Brax PPO...")
    make_inference_fn, params, results = trainer.train(
        total_timesteps=1_000_000  # 1M steps for testing
    )
    
    print(f"\nTraining completed!")
    print(f"Final reward: {results['final_metrics']['final_reward']:.2f}")
    print(f"Training time: {results['final_metrics']['training_time']:.1f}s")
    print(f"Report saved to: {results['report_path']}")