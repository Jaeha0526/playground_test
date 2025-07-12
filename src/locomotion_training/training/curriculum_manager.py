"""Curriculum learning manager for progressive navigation training."""

import numpy as np
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import time
from dataclasses import dataclass

from ..config.navigation_config import (
    get_navigation_training_config,
    get_curriculum_stages,
    get_training_schedule
)


@dataclass
class StageMetrics:
    """Metrics for a curriculum stage."""
    stage: int
    episodes_completed: int
    success_rate: float
    avg_episode_length: float
    avg_time_to_goal: float
    collision_rate: float
    time_efficiency: float
    path_efficiency: float
    last_evaluation_time: float
    

@dataclass 
class CurriculumState:
    """Current state of curriculum learning."""
    current_stage: int
    training_step: int
    total_episodes: int
    stage_start_time: float
    stage_start_step: int
    metrics_history: List[StageMetrics]
    ready_for_advancement: bool


class CurriculumManager:
    """Manages curriculum learning progression for navigation training."""
    
    def __init__(
        self,
        start_stage: int = 1,
        save_path: Optional[str] = None,
        advancement_buffer: int = 50,  # Episodes to wait after meeting criteria
    ):
        """Initialize curriculum manager.
        
        Args:
            start_stage: Starting curriculum stage (1-4)
            save_path: Path to save curriculum state
            advancement_buffer: Number of episodes to confirm advancement
        """
        self.start_stage = start_stage
        self.advancement_buffer = advancement_buffer
        self.save_path = save_path or "curriculum_state.json"
        
        # Load curriculum configuration
        self.curriculum_stages = get_curriculum_stages()
        self.training_schedule = get_training_schedule()
        self.advancement_criteria = self.training_schedule["advancement_criteria"]
        
        # Initialize state
        self.state = CurriculumState(
            current_stage=start_stage,
            training_step=0,
            total_episodes=0,
            stage_start_time=time.time(),
            stage_start_step=0,
            metrics_history=[],
            ready_for_advancement=False
        )
        
        # Tracking variables
        self.recent_episodes = []  # Buffer for recent episode results
        self.evaluation_frequency = 1000  # Steps between evaluations
        self.last_evaluation_step = 0
        
        print(f"Initialized curriculum manager starting at Stage {start_stage}")
        print(f"Stage: {self.curriculum_stages[start_stage]['name']}")
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get training configuration for current stage.
        
        Returns:
            Training configuration for current stage
        """
        return get_navigation_training_config(self.state.current_stage)
    
    def should_evaluate(self, training_step: int) -> bool:
        """Check if it's time to evaluate progress.
        
        Args:
            training_step: Current training step
        
        Returns:
            True if evaluation should be performed
        """
        return (training_step - self.last_evaluation_step) >= self.evaluation_frequency
    
    def add_episode_result(
        self,
        success: bool,
        episode_length: int,
        time_to_goal: Optional[float],
        collision_occurred: bool,
        goal_distance: float,
        optimal_time: Optional[float] = None,
        optimal_path_length: Optional[float] = None,
        actual_path_length: Optional[float] = None,
    ):
        """Add result from a completed episode.
        
        Args:
            success: Whether goal was reached
            episode_length: Number of steps in episode
            time_to_goal: Time taken to reach goal (if successful)
            collision_occurred: Whether robot collided with walls
            goal_distance: Distance from spawn to goal
            optimal_time: Optimal time for this goal distance
            optimal_path_length: Optimal path length
            actual_path_length: Actual path taken by robot
        """
        episode_result = {
            "success": success,
            "episode_length": episode_length,
            "time_to_goal": time_to_goal,
            "collision": collision_occurred,
            "goal_distance": goal_distance,
            "timestamp": time.time(),
        }
        
        # Calculate efficiency metrics if optimal values provided
        if success and optimal_time and time_to_goal:
            episode_result["time_efficiency"] = optimal_time / time_to_goal
        else:
            episode_result["time_efficiency"] = 0.0
            
        if success and optimal_path_length and actual_path_length:
            episode_result["path_efficiency"] = optimal_path_length / actual_path_length
        else:
            episode_result["path_efficiency"] = 0.0
        
        # Add to recent episodes buffer
        self.recent_episodes.append(episode_result)
        
        # Keep only recent episodes for evaluation
        max_buffer_size = max(200, self.advancement_criteria["min_episodes"] * 2)
        if len(self.recent_episodes) > max_buffer_size:
            self.recent_episodes = self.recent_episodes[-max_buffer_size:]
        
        self.state.total_episodes += 1
    
    def evaluate_stage_performance(self, training_step: int) -> Tuple[StageMetrics, bool]:
        """Evaluate current stage performance and check advancement criteria.
        
        Args:
            training_step: Current training step
            
        Returns:
            Tuple of (stage_metrics, should_advance)
        """
        self.last_evaluation_step = training_step
        
        if len(self.recent_episodes) < self.advancement_criteria["min_episodes"]:
            # Not enough episodes for evaluation
            metrics = StageMetrics(
                stage=self.state.current_stage,
                episodes_completed=len(self.recent_episodes),
                success_rate=0.0,
                avg_episode_length=0.0,
                avg_time_to_goal=0.0,
                collision_rate=0.0,
                time_efficiency=0.0,
                path_efficiency=0.0,
                last_evaluation_time=time.time(),
            )
            return metrics, False
        
        # Calculate metrics from recent episodes
        recent = self.recent_episodes[-self.advancement_criteria["min_episodes"]:]
        
        successes = [ep for ep in recent if ep["success"]]
        collisions = [ep for ep in recent if ep["collision"]]
        
        success_rate = len(successes) / len(recent)
        collision_rate = len(collisions) / len(recent)
        avg_episode_length = np.mean([ep["episode_length"] for ep in recent])
        
        if successes:
            avg_time_to_goal = np.mean([ep["time_to_goal"] for ep in successes if ep["time_to_goal"]])
            time_efficiency = np.mean([ep["time_efficiency"] for ep in successes])
            path_efficiency = np.mean([ep["path_efficiency"] for ep in successes])
        else:
            avg_time_to_goal = 0.0
            time_efficiency = 0.0 
            path_efficiency = 0.0
        
        metrics = StageMetrics(
            stage=self.state.current_stage,
            episodes_completed=len(recent),
            success_rate=success_rate,
            avg_episode_length=avg_episode_length,
            avg_time_to_goal=avg_time_to_goal,
            collision_rate=collision_rate,
            time_efficiency=time_efficiency,
            path_efficiency=path_efficiency,
            last_evaluation_time=time.time(),
        )
        
        # Check advancement criteria
        should_advance = self._check_advancement_criteria(metrics)
        
        # Store metrics
        self.state.metrics_history.append(metrics)
        
        return metrics, should_advance
    
    def _check_advancement_criteria(self, metrics: StageMetrics) -> bool:
        """Check if current stage meets advancement criteria.
        
        Args:
            metrics: Current stage metrics
            
        Returns:
            True if stage is ready for advancement
        """
        stage = self.state.current_stage
        
        # Get thresholds for current stage
        success_threshold = self.advancement_criteria["success_rate_threshold"][stage]
        efficiency_threshold = self.advancement_criteria["time_efficiency_threshold"][stage]
        min_episodes = self.advancement_criteria["min_episodes"]
        stability_episodes = self.advancement_criteria["stability_episodes"]
        
        # Basic criteria
        meets_success_rate = metrics.success_rate >= success_threshold
        meets_efficiency = metrics.time_efficiency >= efficiency_threshold
        has_enough_episodes = metrics.episodes_completed >= min_episodes
        
        if not (meets_success_rate and meets_efficiency and has_enough_episodes):
            self.state.ready_for_advancement = False
            return False
        
        # Check stability over recent episodes
        if len(self.state.metrics_history) < 3:
            # Need at least 3 evaluations for stability
            self.state.ready_for_advancement = False
            return False
        
        # Check last few evaluations for consistent performance
        recent_metrics = self.state.metrics_history[-3:]
        stable_performance = all(
            m.success_rate >= success_threshold * 0.9 and  # Allow 10% tolerance
            m.time_efficiency >= efficiency_threshold * 0.9
            for m in recent_metrics
        )
        
        if stable_performance:
            if not self.state.ready_for_advancement:
                print(f"Stage {stage} meets advancement criteria! Confirming over {self.advancement_buffer} episodes...")
                self.state.ready_for_advancement = True
                self._advancement_confirmation_start = self.state.total_episodes
            
            # Check if we've had stable performance for enough episodes
            episodes_since_ready = self.state.total_episodes - self._advancement_confirmation_start
            if episodes_since_ready >= self.advancement_buffer:
                return True
        else:
            self.state.ready_for_advancement = False
        
        return False
    
    def advance_to_next_stage(self) -> bool:
        """Advance to the next curriculum stage.
        
        Returns:
            True if advancement was successful, False if already at final stage
        """
        if self.state.current_stage >= 4:
            print("Already at final stage (4), cannot advance further")
            return False
        
        old_stage = self.state.current_stage
        self.state.current_stage += 1
        self.state.stage_start_time = time.time()
        self.state.stage_start_step = self.state.training_step
        self.state.ready_for_advancement = False
        
        # Reset episode buffer for new stage
        self.recent_episodes = []
        
        # Log advancement
        old_stage_info = self.curriculum_stages[old_stage]
        new_stage_info = self.curriculum_stages[self.state.current_stage]
        
        print(f"\n🎓 ADVANCING CURRICULUM!")
        print(f"   From Stage {old_stage}: {old_stage_info['name']}")
        print(f"   To Stage {self.state.current_stage}: {new_stage_info['name']}")
        print(f"   New goal distance range: {new_stage_info['goal_distance_range']}")
        print(f"   Episodes completed in Stage {old_stage}: {self.state.total_episodes}")
        
        # Save state
        self.save_state()
        
        return True
    
    def get_stage_progress(self) -> Dict[str, Any]:
        """Get progress information for current stage.
        
        Returns:
            Dictionary with stage progress information
        """
        stage_info = self.curriculum_stages[self.state.current_stage]
        
        # Calculate time and step progress
        time_in_stage = time.time() - self.state.stage_start_time
        steps_in_stage = self.state.training_step - self.state.stage_start_step
        target_steps = self.training_schedule["stage_timesteps"][self.state.current_stage]
        
        progress = {
            "current_stage": self.state.current_stage,
            "stage_name": stage_info["name"],
            "stage_description": stage_info["description"],
            "total_episodes": self.state.total_episodes,
            "training_step": self.state.training_step,
            "steps_in_stage": steps_in_stage,
            "target_steps": target_steps,
            "step_progress": min(1.0, steps_in_stage / target_steps),
            "time_in_stage_hours": time_in_stage / 3600,
            "ready_for_advancement": self.state.ready_for_advancement,
            "recent_episodes_count": len(self.recent_episodes),
        }
        
        # Add latest metrics if available
        if self.state.metrics_history:
            latest_metrics = self.state.metrics_history[-1]
            progress.update({
                "success_rate": latest_metrics.success_rate,
                "time_efficiency": latest_metrics.time_efficiency,
                "collision_rate": latest_metrics.collision_rate,
                "avg_episode_length": latest_metrics.avg_episode_length,
            })
        
        return progress
    
    def update_training_step(self, step: int):
        """Update current training step.
        
        Args:
            step: Current training step
        """
        self.state.training_step = step
    
    def save_state(self):
        """Save curriculum state to file."""
        state_dict = {
            "current_stage": self.state.current_stage,
            "training_step": self.state.training_step,
            "total_episodes": self.state.total_episodes,
            "stage_start_time": self.state.stage_start_time,
            "stage_start_step": self.state.stage_start_step,
            "ready_for_advancement": self.state.ready_for_advancement,
            "metrics_history": [
                {
                    "stage": m.stage,
                    "episodes_completed": m.episodes_completed,
                    "success_rate": m.success_rate,
                    "avg_episode_length": m.avg_episode_length,
                    "avg_time_to_goal": m.avg_time_to_goal,
                    "collision_rate": m.collision_rate,
                    "time_efficiency": m.time_efficiency,
                    "path_efficiency": m.path_efficiency,
                    "last_evaluation_time": m.last_evaluation_time,
                }
                for m in self.state.metrics_history
            ],
            "recent_episodes": self.recent_episodes,
        }
        
        with open(self.save_path, 'w') as f:
            json.dump(state_dict, f, indent=2)
    
    def load_state(self, path: Optional[str] = None):
        """Load curriculum state from file.
        
        Args:
            path: Path to state file (optional)
        """
        load_path = path or self.save_path
        
        if not Path(load_path).exists():
            print(f"No curriculum state found at {load_path}, starting fresh")
            return
        
        with open(load_path, 'r') as f:
            state_dict = json.load(f)
        
        self.state.current_stage = state_dict["current_stage"]
        self.state.training_step = state_dict["training_step"]
        self.state.total_episodes = state_dict["total_episodes"]
        self.state.stage_start_time = state_dict["stage_start_time"]
        self.state.stage_start_step = state_dict["stage_start_step"]
        self.state.ready_for_advancement = state_dict["ready_for_advancement"]
        
        # Reconstruct metrics history
        self.state.metrics_history = [
            StageMetrics(**m) for m in state_dict["metrics_history"]
        ]
        
        self.recent_episodes = state_dict.get("recent_episodes", [])
        
        print(f"Loaded curriculum state from {load_path}")
        print(f"Resuming at Stage {self.state.current_stage} with {self.state.total_episodes} episodes")
    
    def print_stage_summary(self):
        """Print summary of current stage progress."""
        progress = self.get_stage_progress()
        
        print(f"\n📊 Stage {progress['current_stage']} Progress: {progress['stage_name']}")
        print(f"   Episodes: {progress['total_episodes']}")
        print(f"   Training Steps: {progress['training_step']:,} / {progress['target_steps']:,}")
        print(f"   Step Progress: {progress['step_progress']:.1%}")
        print(f"   Time in Stage: {progress['time_in_stage_hours']:.1f} hours")
        
        if "success_rate" in progress:
            print(f"   Success Rate: {progress['success_rate']:.1%}")
            print(f"   Time Efficiency: {progress['time_efficiency']:.1%}")
            print(f"   Collision Rate: {progress['collision_rate']:.1%}")
        
        if progress["ready_for_advancement"]:
            print(f"   🎯 READY FOR ADVANCEMENT!")


# Example usage
if __name__ == "__main__":
    # Test curriculum manager
    manager = CurriculumManager(start_stage=1)
    
    # Print initial configuration
    config = manager.get_current_config()
    print(f"Stage 1 configuration: {config.env_config}")
    
    # Simulate some episodes
    import random
    
    for episode in range(150):
        # Simulate episode results
        success = random.random() < 0.7  # 70% success rate
        episode_length = random.randint(50, 200)
        time_to_goal = episode_length * 0.02 if success else None  # 50Hz
        collision = random.random() < 0.1  # 10% collision rate
        goal_distance = random.uniform(1.0, 3.0)
        
        manager.add_episode_result(
            success=success,
            episode_length=episode_length,
            time_to_goal=time_to_goal,
            collision_occurred=collision,
            goal_distance=goal_distance,
            optimal_time=goal_distance / 1.0,  # Assume 1 m/s optimal speed
        )
        
        # Evaluate every 25 episodes
        if episode % 25 == 0:
            manager.update_training_step(episode * 1000)  # Fake training steps
            
            if manager.should_evaluate(episode * 1000):
                metrics, should_advance = manager.evaluate_stage_performance(episode * 1000)
                print(f"Episode {episode}: Success rate = {metrics.success_rate:.1%}")
                
                if should_advance:
                    print("Should advance to next stage!")
                    break
    
    manager.print_stage_summary()