"""Configuration for navigation training with curriculum learning."""

from ml_collections import config_dict
from typing import Dict, Any


def get_navigation_training_config(stage: int = 1) -> config_dict.ConfigDict:
    """Get navigation training configuration for a specific curriculum stage.
    
    Args:
        stage: Curriculum stage (1-4)
    
    Returns:
        Training configuration
    """
    # Base configuration
    base_config = config_dict.create(
        # Training parameters
        algorithm="ppo",
        num_timesteps=10_000_000,  # Will be adjusted per stage
        num_evals=5,
        reward_scaling=1.0,
        normalize_observations=True,
        
        # PPO-specific parameters
        learning_rate=3e-4,  # Higher LR for smaller navigation network
        num_envs=2048,       # Start with fewer envs for Stage 1
        unroll_length=25,    # Start with shorter rollouts
        batch_size=256,       # Smaller batches for navigation
        num_minibatches=8,  # 2048 / 64 = 32 minibatches
        num_updates_per_batch=4,
        discounting=0.97,
        entropy_cost=1e-2,
        max_grad_norm=1.0,
        
        # Network architecture
        navigation_network=config_dict.create(
            hidden_sizes=(64, 32),
            activation="relu",
            input_dim=10,   # Navigation observations
            output_dim=3,   # Velocity commands
        ),
        
        # Locomotion policy (frozen)
        locomotion_checkpoint_path="checkpoints/Go1JoystickFlatTerrain_20250630_224046/best/",
        
        # Environment configuration (will be overridden per stage)
        env_config=config_dict.create(
            episode_length=250,
            room_size=6.0,
            wall_buffer=1.0,
            goal_radius=0.5,
            min_goal_distance=1.0,
            max_goal_distance=3.0,
        ),
        
        # Evaluation configuration
        eval_config=config_dict.create(
            num_episodes=10,
            save_video=True,
            video_path="videos/navigation",
            eval_frequency=100_000,  # Evaluate every 100k steps
        ),
        
        # Logging and saving
        checkpoint_frequency=500_000,  # Save every 500k steps
        log_frequency=10_000,          # Log every 10k steps
        experiment_name="go1_navigation",
        run_name=f"stage_{stage}",
    )
    
    # Stage-specific modifications
    stage_configs = {
        1: {
            "num_timesteps": 10_000_000,  # 10M steps
            "num_envs": 2048,
            "unroll_length": 25,
            "env_config": {
                "episode_length": 250,
                "room_size": 6.0,
                "max_goal_distance": 3.0,
                "min_goal_distance": 1.0,
            }
        },
        2: {
            "num_timesteps": 15_000_000,  # 15M steps
            "num_envs": 4096,
            "unroll_length": 40,
            "env_config": {
                "episode_length": 400,
                "room_size": 8.0,
                "max_goal_distance": 5.0,
                "min_goal_distance": 2.0,
            }
        },
        3: {
            "num_timesteps": 15_000_000,  # 15M steps
            "num_envs": 4096,
            "unroll_length": 50,
            "env_config": {
                "episode_length": 600,
                "room_size": 10.0,
                "max_goal_distance": 8.0,
                "min_goal_distance": 3.0,
                "add_obstacles": True,
            }
        },
        4: {
            "num_timesteps": 10_000_000,  # 10M steps
            "num_envs": 4096,
            "unroll_length": 50,
            "env_config": {
                "episode_length": 1000,
                "room_size": 10.0,
                "max_goal_distance": 12.0,
                "min_goal_distance": 5.0,
                "add_obstacles": True,
                "num_obstacles": (3, 5),
            }
        }
    }
    
    # Apply stage-specific configuration
    if stage in stage_configs:
        stage_config = stage_configs[stage]
        
        # Update top-level configs
        for key, value in stage_config.items():
            if key != "env_config":
                base_config[key] = value
        
        # Update environment config
        if "env_config" in stage_config:
            for key, value in stage_config["env_config"].items():
                base_config.env_config[key] = value
        
        # Update batch size calculation
        base_config.num_minibatches = base_config.num_envs // base_config.batch_size
    
    return base_config


def get_curriculum_stages() -> Dict[int, Dict[str, Any]]:
    """Get curriculum stage definitions.
    
    Returns:
        Dictionary mapping stage number to stage configuration
    """
    return {
        1: {
            "name": "Close Goals",
            "description": "Learn basic navigation to nearby goals",
            "goal_distance_range": (1.0, 3.0),
            "success_threshold": 0.8,
            "time_efficiency_threshold": 0.7,
            "episodes_to_evaluate": 100,
        },
        2: {
            "name": "Medium Goals", 
            "description": "Learn path planning for medium distances",
            "goal_distance_range": (2.0, 5.0),
            "success_threshold": 0.7,
            "time_efficiency_threshold": 0.6,
            "episodes_to_evaluate": 100,
        },
        3: {
            "name": "Full Room Navigation",
            "description": "Navigate full room with simple obstacles",
            "goal_distance_range": (3.0, 8.0),
            "success_threshold": 0.6,
            "time_efficiency_threshold": 0.5,
            "episodes_to_evaluate": 100,
        },
        4: {
            "name": "Complex Navigation",
            "description": "Navigate complex environments with multiple obstacles",
            "goal_distance_range": (5.0, 12.0),
            "success_threshold": 0.5,
            "time_efficiency_threshold": 0.4,
            "episodes_to_evaluate": 200,
        }
    }


def get_reward_config() -> config_dict.ConfigDict:
    """Get navigation reward configuration.
    
    Returns:
        Reward configuration
    """
    return config_dict.create(
        scales=config_dict.create(
            # Main navigation objectives
            goal_reached=100.0,        # Sparse success reward
            goal_progress=2.0,         # Dense progress reward
            goal_distance=-0.1,        # Distance penalty
            
            # Safety constraints
            wall_collision=-10.0,      # Wall collision penalty
            boundary_penalty=-0.5,     # Stay away from walls
            
            # Efficiency incentives
            time_penalty=-0.1,         # Encourage speed
            heading_alignment=0.2,     # Face toward goal
            
            # Basic stability (reduced since locomotion frozen)
            orientation=-1.0,          # Stay upright
            termination=-1.0,          # Don't fall
        ),
        
        # Reward-specific parameters
        goal_radius=0.5,
        wall_collision_threshold=0.3,
        boundary_warning_distance=1.0,
        progress_smoothing=0.1,
    )


def get_evaluation_config() -> config_dict.ConfigDict:
    """Get evaluation configuration.
    
    Returns:
        Evaluation configuration
    """
    return config_dict.create(
        # Evaluation episodes
        num_episodes=20,
        max_episode_steps=1000,
        
        # Video recording
        save_video=True,
        video_fps=50,
        video_width=640,
        video_height=480,
        camera_view="room_overview",
        
        # Metrics to track
        metrics=config_dict.create(
            success_rate=True,
            time_efficiency=True,
            path_efficiency=True,
            collision_rate=True,
            goal_distance_distribution=True,
        ),
        
        # Test scenarios
        test_scenarios=config_dict.create(
            # Standard evaluation
            standard=config_dict.create(
                num_episodes=20,
                goal_distance_range=(1.0, 8.0),
                room_size=10.0,
            ),
            
            # Generalization tests
            close_goals=config_dict.create(
                num_episodes=10,
                goal_distance_range=(0.5, 2.0),
                room_size=10.0,
            ),
            
            long_distance=config_dict.create(
                num_episodes=10,
                goal_distance_range=(8.0, 12.0),
                room_size=10.0,
            ),
            
            large_room=config_dict.create(
                num_episodes=10,
                goal_distance_range=(5.0, 15.0),
                room_size=15.0,
            ),
        )
    )


# Training schedule for all stages
def get_training_schedule() -> Dict[str, Any]:
    """Get complete training schedule for curriculum learning.
    
    Returns:
        Training schedule dictionary
    """
    return {
        "total_stages": 4,
        "total_timesteps": 50_000_000,  # 50M total
        "stage_timesteps": {
            1: 10_000_000,  # 10M for stage 1
            2: 15_000_000,  # 15M for stage 2
            3: 15_000_000,  # 15M for stage 3
            4: 10_000_000,  # 10M for stage 4
        },
        "estimated_training_time": {
            "stage_1": "1-2 hours",
            "stage_2": "2-3 hours", 
            "stage_3": "2-3 hours",
            "stage_4": "1-2 hours",
            "total": "6-10 hours"
        },
        "advancement_criteria": {
            "min_episodes": 100,
            "success_rate_threshold": {1: 0.8, 2: 0.7, 3: 0.6, 4: 0.5},
            "time_efficiency_threshold": {1: 0.7, 2: 0.6, 3: 0.5, 4: 0.4},
            "stability_episodes": 50,  # Consistent performance over N episodes
        }
    }


# Example usage
if __name__ == "__main__":
    # Test configuration for each stage
    for stage in range(1, 5):
        config = get_navigation_training_config(stage)
        print(f"\nStage {stage} Configuration:")
        print(f"  Timesteps: {config.num_timesteps:,}")
        print(f"  Environments: {config.num_envs}")
        print(f"  Episode length: {config.env_config.episode_length}")
        print(f"  Room size: {config.env_config.room_size}m")
        print(f"  Goal distance: {config.env_config.min_goal_distance}-{config.env_config.max_goal_distance}m")
        print(f"  Minibatches: {config.num_minibatches}")
    
    # Test curriculum stages
    print("\nCurriculum Stages:")
    stages = get_curriculum_stages()
    for stage_id, stage_info in stages.items():
        print(f"  Stage {stage_id}: {stage_info['name']} - {stage_info['description']}")
    
    # Test training schedule
    print("\nTraining Schedule:")
    schedule = get_training_schedule()
    for stage, timesteps in schedule["stage_timesteps"].items():
        time_est = schedule["estimated_training_time"][f"stage_{stage}"]
        print(f"  Stage {stage}: {timesteps:,} steps ({time_est})")