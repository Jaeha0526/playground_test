"""Experiment logging and visualization for navigation training."""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import threading
import queue

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not available. Using local logging only.")


class ExperimentLogger:
    """Comprehensive logging and visualization for navigation experiments."""
    
    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        log_dir: str = "reward_graphs",
        use_wandb: bool = False,
        wandb_project: str = "go1-navigation",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            run_name: Unique run identifier
            log_dir: Local logging directory (follows main.py pattern)
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            config: Experiment configuration to log
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.start_time = time.time()
        
        # Setup directories following main.py pattern: reward_graphs/{env_name}_{timestamp}/
        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging buffers
        self.metrics_history = []
        self.config_history = []
        self.stage_history = []
        
        # Real-time plotting
        self.plot_queue = queue.Queue()
        self.plotting_thread = None
        self.stop_plotting = threading.Event()
        
        # W&B setup
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=run_name,
                config=config,
                dir=str(self.log_dir),
            )
        
        # Save initial config
        if config:
            self.log_config(config)
        
        print(f"🔬 Experiment Logger initialized")
        print(f"   Experiment: {experiment_name}")
        print(f"   Run: {run_name}")
        print(f"   Log dir: {self.log_dir}")
        print(f"   W&B: {'✅' if self.use_wandb else '❌'}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        config_entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "config": config,
        }
        
        self.config_history.append(config_entry)
        
        # Save to file
        config_file = self.log_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_entry, f, indent=2, default=str)
        
        print(f"📝 Configuration logged to {config_file}")
    
    def log_metrics(
        self,
        step: int,
        metrics: Dict[str, float],
        stage: Optional[int] = None,
        prefix: str = "",
    ):
        """Log training metrics.
        
        Args:
            step: Training step
            metrics: Dictionary of metrics
            stage: Current curriculum stage
            prefix: Prefix for metric names
        """
        timestamp = time.time()
        
        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Create metrics entry
        metrics_entry = {
            "step": step,
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
            "elapsed_time": timestamp - self.start_time,
            "stage": stage,
            **metrics
        }
        
        self.metrics_history.append(metrics_entry)
        
        # Log to W&B
        if self.use_wandb:
            wandb_metrics = {"step": step, **metrics}
            if stage is not None:
                wandb_metrics["stage"] = stage
            wandb.log(wandb_metrics, step=step)
        
        # Queue for real-time plotting
        self.plot_queue.put(metrics_entry)
        
        # Save to file periodically (following main.py pattern)
        if len(self.metrics_history) % 100 == 0:
            self._save_training_progress()
    
    def log_stage_transition(
        self,
        step: int,
        old_stage: int,
        new_stage: int,
        stage_metrics: Dict[str, float],
    ):
        """Log curriculum stage transition.
        
        Args:
            step: Training step
            old_stage: Previous stage
            new_stage: New stage
            stage_metrics: Final metrics from old stage
        """
        transition_entry = {
            "step": step,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "old_stage": old_stage,
            "new_stage": new_stage,
            "final_metrics": stage_metrics,
        }
        
        self.stage_history.append(transition_entry)
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({
                "stage_transition": new_stage,
                "stage_old": old_stage,
                **{f"final_{k}": v for k, v in stage_metrics.items()}
            }, step=step)
        
        # Save stage history
        stage_file = self.log_dir / "stage_transitions.json"
        with open(stage_file, 'w') as f:
            json.dump(self.stage_history, f, indent=2, default=str)
        
        print(f"🎓 Stage transition logged: {old_stage} → {new_stage}")
    
    def start_real_time_plotting(self, metrics_to_plot: List[str] = None):
        """Start real-time plotting in separate thread.
        
        Args:
            metrics_to_plot: List of metric names to plot
        """
        if metrics_to_plot is None:
            metrics_to_plot = [
                "eval/success_rate",
                "eval/average_reward", 
                "train/policy_loss",
                "navigation/goal_distance"
            ]
        
        self.metrics_to_plot = metrics_to_plot
        self.stop_plotting.clear()
        
        self.plotting_thread = threading.Thread(
            target=self._real_time_plotting_worker,
            daemon=True
        )
        self.plotting_thread.start()
        
        print(f"📊 Real-time plotting started for: {metrics_to_plot}")
    
    def stop_real_time_plotting(self):
        """Stop real-time plotting."""
        if self.plotting_thread:
            self.stop_plotting.set()
            self.plotting_thread.join(timeout=1.0)
            print("📊 Real-time plotting stopped")
    
    def _real_time_plotting_worker(self):
        """Worker function for real-time plotting."""
        # Setup matplotlib for real-time plotting
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Navigation Training: {self.run_name}")
        axes = axes.flatten()
        
        # Initialize plot data
        plot_data = {metric: {"steps": [], "values": []} for metric in self.metrics_to_plot}
        lines = {}
        
        # Setup axes
        for i, metric in enumerate(self.metrics_to_plot[:4]):  # Limit to 4 plots
            if i < len(axes):
                axes[i].set_title(metric)
                axes[i].set_xlabel("Training Step")
                axes[i].set_ylabel("Value")
                line, = axes[i].plot([], [], 'b-', alpha=0.7)
                lines[metric] = line
        
        plt.tight_layout()
        
        # Real-time update loop
        while not self.stop_plotting.is_set():
            try:
                # Get new metrics (non-blocking)
                while not self.plot_queue.empty():
                    metrics_entry = self.plot_queue.get_nowait()
                    step = metrics_entry["step"]
                    
                    # Update plot data
                    for metric in self.metrics_to_plot:
                        if metric in metrics_entry:
                            plot_data[metric]["steps"].append(step)
                            plot_data[metric]["values"].append(metrics_entry[metric])
                            
                            # Keep only last 1000 points for performance
                            if len(plot_data[metric]["steps"]) > 1000:
                                plot_data[metric]["steps"] = plot_data[metric]["steps"][-1000:]
                                plot_data[metric]["values"] = plot_data[metric]["values"][-1000:]
                
                # Update plots
                for i, metric in enumerate(self.metrics_to_plot[:4]):
                    if metric in lines and plot_data[metric]["steps"]:
                        line = lines[metric]
                        steps = plot_data[metric]["steps"]
                        values = plot_data[metric]["values"]
                        
                        line.set_data(steps, values)
                        
                        # Update axis limits
                        if len(steps) > 1:
                            axes[i].set_xlim(min(steps), max(steps))
                            axes[i].set_ylim(min(values) * 0.9, max(values) * 1.1)
                
                # Redraw
                fig.canvas.draw()
                fig.canvas.flush_events()
                
                # Save plot to disk every few updates
                if hasattr(self, '_plot_save_counter'):
                    self._plot_save_counter += 1
                else:
                    self._plot_save_counter = 1
                
                # Save every 10 updates (every ~10 seconds)
                if self._plot_save_counter % 10 == 0:
                    try:
                        plot_path = self.log_dir / "training_progress_current.png"
                        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                        print(f"📊 Plot saved to {plot_path}")
                    except Exception as e:
                        print(f"Failed to save plot: {e}")
                
                # Sleep briefly
                time.sleep(1.0)
                
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"Real-time plotting error: {e}")
                break
        
        plt.ioff()
        plt.close(fig)
    
    def _save_training_progress(self):
        """Save training progress following main.py pattern."""
        # Create training_progress.json matching main.py format
        if not self.metrics_history:
            return
            
        # Extract key metrics for main format
        steps = [m["step"] for m in self.metrics_history if "eval/" in str(m)]
        rewards = [m.get("eval/average_reward", 0) for m in self.metrics_history if "eval/" in str(m)]
        reward_stds = [m.get("eval/success_rate", 0) for m in self.metrics_history if "eval/" in str(m)]
        
        if not steps:
            return
            
        # Get current metrics
        latest_metrics = self.metrics_history[-1]
        current_step = latest_metrics.get("step", 0)
        
        # Calculate progress (assuming total timesteps from config)
        total_timesteps = self.config_history[0]["config"].get("num_timesteps", 50_000_000) if self.config_history else 50_000_000
        progress_percent = (current_step / total_timesteps) * 100
        
        # Create training progress data
        training_progress = {
            "config": {
                "env_name": "Go1Navigation",
                "num_timesteps": total_timesteps,
                "num_envs": self.config_history[0]["config"].get("num_envs", 4096) if self.config_history else 4096,
                "batch_size": self.config_history[0]["config"].get("batch_size", 256) if self.config_history else 256,
                "learning_rate": self.config_history[0]["config"].get("learning_rate", 0.001) if self.config_history else 0.001
            },
            "training_data": {
                "steps": steps,
                "rewards": [str(r) for r in rewards],
                "reward_stds": [str(r) for r in reward_stds]
            },
            "current_metrics": {
                "step": current_step,
                "reward_mean": str(rewards[-1] if rewards else 0),
                "reward_std": str(reward_stds[-1] if reward_stds else 0),
                "progress_percent": progress_percent
            },
            "timing": {
                "started_at": datetime.fromtimestamp(self.start_time).isoformat(),
                "current_time": datetime.now().isoformat(),
                "elapsed_total_seconds": time.time() - self.start_time,
                "elapsed_training_seconds": time.time() - self.start_time
            },
            "full_metrics": latest_metrics
        }
        
        # Save training_progress.json
        progress_file = self.log_dir / "training_progress.json"
        with open(progress_file, 'w') as f:
            json.dump(training_progress, f, indent=2, default=str)
            
        # Also save detailed metrics.json
        metrics_file = self.log_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        
        # Create and save training progress plot
        self._create_progress_plot()
    
    def _create_progress_plot(self):
        """Create and save training progress plot."""
        if not self.metrics_history:
            return
            
        try:
            plt.figure(figsize=(12, 8))
            
            # Create subplots for key metrics
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Navigation Training Progress: {self.run_name}")
            
            # Extract data for plotting
            steps = []
            recent_success_rates = []
            time_efficiencies = []
            policy_losses = []
            stage_progress = []
            
            for m in self.metrics_history:
                if m.get("step"):
                    steps.append(m["step"])
                    recent_success_rates.append(m.get("navigation/recent_success_rate", 0))
                    time_efficiencies.append(m.get("navigation/recent_time_efficiency", 0))
                    policy_losses.append(m.get("train/policy_loss", 0))
                    stage_progress.append(m.get("navigation/stage_progress_pct", 0))
            
            if steps:
                # Plot 1: Recent Success Rate
                axes[0, 0].plot(steps, recent_success_rates, 'g-', alpha=0.8)
                axes[0, 0].set_title("Recent Success Rate")
                axes[0, 0].set_ylabel("Success Rate")
                axes[0, 0].grid(True, alpha=0.3)
                
                # Plot 2: Time Efficiency
                axes[0, 1].plot(steps, time_efficiencies, 'b-', alpha=0.8)
                axes[0, 1].set_title("Time Efficiency")
                axes[0, 1].set_ylabel("Efficiency")
                axes[0, 1].grid(True, alpha=0.3)
                
                # Plot 3: Policy Loss
                axes[1, 0].plot(steps, policy_losses, 'r-', alpha=0.8)
                axes[1, 0].set_title("Policy Loss")
                axes[1, 0].set_ylabel("Loss")
                axes[1, 0].set_xlabel("Training Steps")
                axes[1, 0].grid(True, alpha=0.3)
                
                # Plot 4: Stage Progress
                axes[1, 1].plot(steps, stage_progress, 'm-', alpha=0.8)
                axes[1, 1].set_title("Stage Progress")
                axes[1, 1].set_ylabel("Progress %")
                axes[1, 1].set_xlabel("Training Steps")
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.log_dir / "training_progress_current.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Failed to create progress plot: {e}")
    
    def generate_training_report(self) -> str:
        """Generate comprehensive training report.
        
        Returns:
            Path to generated report
        """
        if not self.metrics_history:
            print("No metrics to report")
            return ""
        
        report_file = self.log_dir / "training_report.html"
        
        # Convert metrics to DataFrame-like structure
        steps = [m["step"] for m in self.metrics_history]
        
        # Generate plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Training Report: {self.run_name}")
        
        # Plot 1: Success Rate
        if any("eval/success_rate" in m for m in self.metrics_history):
            success_rates = [m.get("eval/success_rate", 0) for m in self.metrics_history]
            axes[0, 0].plot(steps, success_rates, 'g-', alpha=0.7)
            axes[0, 0].set_title("Success Rate")
            axes[0, 0].set_ylabel("Success Rate (%)")
            axes[0, 0].grid(True)
        
        # Plot 2: Average Reward
        if any("eval/average_reward" in m for m in self.metrics_history):
            rewards = [m.get("eval/average_reward", 0) for m in self.metrics_history]
            axes[0, 1].plot(steps, rewards, 'b-', alpha=0.7)
            axes[0, 1].set_title("Average Reward")
            axes[0, 1].set_ylabel("Reward")
            axes[0, 1].grid(True)
        
        # Plot 3: Policy Loss
        if any("train/policy_loss" in m for m in self.metrics_history):
            losses = [m.get("train/policy_loss", 0) for m in self.metrics_history]
            axes[0, 2].plot(steps, losses, 'r-', alpha=0.7)
            axes[0, 2].set_title("Policy Loss")
            axes[0, 2].set_ylabel("Loss")
            axes[0, 2].grid(True)
        
        # Plot 4: Episode Length
        if any("eval/average_episode_length" in m for m in self.metrics_history):
            lengths = [m.get("eval/average_episode_length", 0) for m in self.metrics_history]
            axes[1, 0].plot(steps, lengths, 'm-', alpha=0.7)
            axes[1, 0].set_title("Episode Length")
            axes[1, 0].set_ylabel("Steps")
            axes[1, 0].grid(True)
        
        # Plot 5: Curriculum Stages
        stages = [m.get("stage", 1) for m in self.metrics_history]
        axes[1, 1].plot(steps, stages, 'c-', alpha=0.7, marker='o', markersize=2)
        axes[1, 1].set_title("Curriculum Stage")
        axes[1, 1].set_ylabel("Stage")
        axes[1, 1].set_yticks([1, 2, 3, 4])
        axes[1, 1].grid(True)
        
        # Plot 6: Training Summary
        axes[1, 2].axis('off')
        
        # Training summary text
        total_steps = max(steps) if steps else 0
        total_time = (time.time() - self.start_time) / 3600  # hours
        current_stage = stages[-1] if stages else 1
        final_success_rate = [m.get("eval/success_rate", 0) for m in self.metrics_history[-10:]]
        avg_success_rate = np.mean(final_success_rate) if final_success_rate else 0
        
        summary_text = f"""Training Summary:
        
Total Steps: {total_steps:,}
Training Time: {total_time:.1f} hours
Current Stage: {current_stage}
Final Success Rate: {avg_success_rate:.1%}

Stage Transitions: {len(self.stage_history)}
Total Episodes: {len(self.metrics_history)}

Experiment: {self.experiment_name}
Run: {self.run_name}
Start Time: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot following main.py pattern
        plot_file = self.log_dir / "training_progress_current.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Navigation Training Report: {self.run_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metrics {{ margin: 20px 0; }}
        .plot {{ text-align: center; margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Go1 Navigation Training Report</h1>
        <h2>{self.run_name}</h2>
        <p><strong>Experiment:</strong> {self.experiment_name}</p>
        <p><strong>Start Time:</strong> {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Duration:</strong> {total_time:.1f} hours</p>
    </div>
    
    <div class="plot">
        <h3>Training Progress</h3>
        <img src="training_progress_current.png" alt="Training Plots" style="max-width: 100%; height: auto;">
    </div>
    
    <div class="metrics">
        <h3>Final Metrics</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Training Steps</td><td>{total_steps:,}</td></tr>
            <tr><td>Training Time</td><td>{total_time:.1f} hours</td></tr>
            <tr><td>Current Stage</td><td>{current_stage}</td></tr>
            <tr><td>Final Success Rate</td><td>{avg_success_rate:.1%}</td></tr>
            <tr><td>Stage Transitions</td><td>{len(self.stage_history)}</td></tr>
            <tr><td>Total Episodes</td><td>{len(self.metrics_history)}</td></tr>
        </table>
    </div>
    
    <div class="metrics">
        <h3>Configuration</h3>
        <pre>{json.dumps(dict(self.config_history[-1]["config"]) if self.config_history else {}, indent=2, default=str)}</pre>
    </div>
</body>
</html>
"""
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        print(f"📋 Training report generated: {report_file}")
        return str(report_file)
    
    def close(self):
        """Close logger and save all data."""
        # Stop real-time plotting
        self.stop_real_time_plotting()
        
        # Save final training progress
        self._save_training_progress()
        
        # Generate final report
        self.generate_training_report()
        
        # Close W&B
        if self.use_wandb:
            wandb.finish()
        
        print(f"📊 Experiment logging completed. Files saved to {self.log_dir}")


# Convenience function for easy logging setup
def setup_experiment_logging(
    experiment_name: str,
    config: Dict[str, Any],
    use_wandb: bool = False,
    start_real_time_plots: bool = True,
) -> ExperimentLogger:
    """Setup experiment logging following main.py naming patterns.
    
    Args:
        experiment_name: Name of experiment (e.g. "Go1Navigation")
        config: Training configuration
        use_wandb: Whether to use Weights & Biases
        start_real_time_plots: Whether to start real-time plotting
    
    Returns:
        ExperimentLogger instance
    """
    # Generate run name following main.py pattern: {env_name}_{timestamp}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment_name}_{timestamp}"
    
    # Create logger (log_dir defaults to "reward_graphs")
    logger = ExperimentLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        use_wandb=use_wandb,
        config=config,
    )
    
    # Start real-time plotting
    if start_real_time_plots:
        logger.start_real_time_plotting([
            "navigation/recent_success_rate",  # Updates every 5k steps
            "navigation/recent_time_efficiency",  # Updates every 5k steps  
            "train/policy_loss",  # Updates every 1k steps
            "navigation/stage_progress_pct",  # Updates every 5k steps
        ])
    
    return logger


# Example usage
if __name__ == "__main__":
    # Test logging system
    config = {
        "stage": 1,
        "num_envs": 2048,
        "learning_rate": 1e-3,
        "episode_length": 250,
    }
    
    logger = setup_experiment_logging(
        experiment_name="go1_navigation_test",
        config=config,
        use_wandb=False,
        start_real_time_plots=True,
    )
    
    # Simulate training
    for step in range(0, 10000, 100):
        # Simulate metrics
        success_rate = min(0.8, step / 10000 * 0.8 + np.random.normal(0, 0.05))
        reward = success_rate * 100 + np.random.normal(0, 5)
        loss = max(0.1, 2.0 - step / 5000 + np.random.normal(0, 0.1))
        
        metrics = {
            "eval/success_rate": max(0, success_rate),
            "eval/average_reward": reward,
            "train/policy_loss": loss,
            "navigation/goal_distance": np.random.uniform(1, 3),
        }
        
        logger.log_metrics(step, metrics, stage=1)
        
        # Simulate stage transition
        if step == 5000:
            logger.log_stage_transition(
                step=step,
                old_stage=1,
                new_stage=2,
                stage_metrics={"final_success_rate": 0.85, "final_reward": 95.0}
            )
        
        time.sleep(0.1)  # Simulate training time
    
    # Cleanup
    logger.close()
    print("Test logging completed!")