#!/usr/bin/env python3
"""Main CLI script for locomotion training and evaluation."""

import typer
from typing import Optional, List
from pathlib import Path
import json

import jax.numpy as jp
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.locomotion_training.config.config import (
    TrainingConfig, 
    EvalConfig, 
    HandstandConfig,
    get_default_training_config,
    get_default_eval_config
)
from src.locomotion_training.training.trainer import LocomotionTrainer
from src.locomotion_training.evaluation.evaluator import LocomotionEvaluator
from mujoco_playground import registry

app = typer.Typer(help="Locomotion Training CLI - Train and evaluate locomotion policies")
console = Console()


@app.command()
def list_envs():
    """List all available locomotion environments."""
    envs = registry.locomotion.ALL_ENVS
    
    table = Table(title="Available Locomotion Environments")
    table.add_column("Environment Name", style="cyan")
    table.add_column("Type", style="green")
    
    for env_name in envs:
        env_type = "Quadruped" if any(x in env_name for x in ["Go1", "Spot", "Barkour"]) else "Biped"
        table.add_row(env_name, env_type)
    
    console.print(table)


@app.command()
def train(
    env_name: str = typer.Argument(..., help="Environment name to train"),
    timesteps: int = typer.Option(100_000_000, "--timesteps", "-t", help="Number of training timesteps"),
    checkpoint_off: bool = typer.Option(False, "--checkpoint-off", help="Disable checkpoint saving"),
    restore_from: Optional[str] = typer.Option(None, "--restore-from", "-r", help="Checkpoint path to restore from"),
    config_file: Optional[str] = typer.Option(None, "--config", help="JSON config file"),
    seed: int = typer.Option(1, "--seed", help="Random seed"),
):
    """Train a locomotion policy."""
    console.print(f"[bold blue]Training {env_name}[/bold blue]")
    
    # Load config
    if config_file:
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = get_default_training_config(env_name)
    
    # Generate timestamp once for consistent naming
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Override with CLI arguments
    config.env_name = env_name
    config.num_timesteps = timesteps
    config.seed = seed
    config.timestamp = timestamp
    
    # Enable checkpointing by default unless disabled
    if not checkpoint_off:
        config.checkpoint_logdir = f"checkpoints/{env_name}_{timestamp}"
    
    # Initialize trainer
    trainer = LocomotionTrainer(config)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description="Training policy...", total=None)
            
            make_inference_fn, params, metrics = trainer.train(
                restore_checkpoint_path=restore_from
            )
        
        console.print("[bold green]Training completed successfully![/bold green]")
        console.print(f"Final reward: {metrics.get('eval/episode_reward', 'N/A')}")
        
        if trainer.checkpoint_path:
            latest_checkpoint = trainer.get_latest_checkpoint()
            console.print(f"Latest checkpoint: {latest_checkpoint}")
            best_checkpoint = trainer.get_best_checkpoint()
            if best_checkpoint:
                console.print(f"Best checkpoint: {best_checkpoint} (reward: {trainer.best_reward:.3f})")
        else:
            console.print("[bold yellow]Checkpointing was disabled[/bold yellow]")
        
    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def evaluate(
    env_name: str = typer.Argument(..., help="Environment name to evaluate"),
    checkpoint_path: str = typer.Argument(..., help="Path to checkpoint to evaluate"),
    x_vel: float = typer.Option(0.0, "--x-vel", help="Forward velocity command"),
    y_vel: float = typer.Option(0.0, "--y-vel", help="Lateral velocity command"),
    yaw_vel: float = typer.Option(0.0, "--yaw-vel", help="Angular velocity command"),
    episodes: int = typer.Option(5, "--episodes", "-e", help="Number of episodes to evaluate"),
    save_video: bool = typer.Option(True, "--save-video/--no-video", help="Save evaluation video"),
    show_video: bool = typer.Option(False, "--show-video", help="Display video interactively"),
    video_path: str = typer.Option("videos", "--video-path", help="Directory to save videos"),
    show_plots: bool = typer.Option(True, "--plots/--no-plots", help="Show evaluation plots"),
    camera: str = typer.Option("track", "--camera", help="Camera view: track, side, front"),
    width: int = typer.Option(640, "--width", help="Video width"),
    height: int = typer.Option(480, "--height", help="Video height"),
):
    """Evaluate a trained locomotion policy."""
    console.print(f"[bold blue]Evaluating {env_name}[/bold blue]")
    console.print(f"Command: x={x_vel}, y={y_vel}, yaw={yaw_vel}")
    
    # Setup evaluation config
    config = get_default_eval_config(env_name)
    config.x_vel = x_vel
    config.y_vel = y_vel
    config.yaw_vel = yaw_vel
    config.num_episodes = episodes
    config.save_video = save_video
    config.video_path = video_path
    config.camera = camera
    config.width = width
    config.height = height
    
    # Create evaluator and run evaluation
    try:
        evaluator = LocomotionEvaluator(config)
        
        # Create command array from velocity parameters
        command = jp.array([x_vel, y_vel, yaw_vel]) if any([x_vel, y_vel, yaw_vel]) else None
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description="Running evaluation...", total=None)
            
            # Use evaluate_checkpoint which automatically includes model ID in video filenames
            evaluation_data = evaluator.evaluate_checkpoint(checkpoint_path, command)
        
        # Show results
        if show_plots:
            evaluator.visualize_results(evaluation_data)
        
        # Create video
        if save_video or show_video:
            frames = evaluator.create_video(evaluation_data)
            if save_video:
                console.print(f"[bold green]Video saved to {video_path}[/bold green]")
            if show_video:
                console.print("[bold yellow]Displaying video...[/bold yellow]")
                try:
                    import mediapy as media
                    fps = 1.0 / evaluator.eval_env.dt / config.render_every
                    media.show_video(frames, fps=fps, loop=False)
                    console.print("[bold green]Video displayed successfully![/bold green]")
                except ImportError:
                    console.print("[bold red]mediapy not available for video display[/bold red]")
                except Exception as e:
                    console.print(f"[bold red]Failed to display video: {e}[/bold red]")
        
        console.print("[bold green]Evaluation completed successfully![/bold green]")
        
        # Print metrics summary
        metrics_table = Table(title="Evaluation Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        
        for key, value in evaluation_data['metrics'].items():
            metrics_table.add_row(key, f"{value:.4f}")
        
        console.print(metrics_table)
        
    except Exception as e:
        console.print(f"[bold red]Evaluation failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def train_handstand(
    checkpoint_off: bool = typer.Option(False, "--checkpoint-off", help="Disable checkpoint saving"),
    finetune: bool = typer.Option(False, "--finetune", help="Finetune with energy penalties"),
    restore_from: Optional[str] = typer.Option(None, "--restore-from", "-r", help="Checkpoint to restore from"),
):
    """Train Go1 handstand policy with optional finetuning."""
    env_name = "Go1Handstand"
    console.print(f"[bold blue]Training {env_name}[/bold blue]")
    
    # Generate timestamp once for consistent naming
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup config
    config = get_default_training_config(env_name)
    config.timestamp = timestamp
    
    # Enable checkpointing by default unless disabled
    if not checkpoint_off:
        config.checkpoint_logdir = f"checkpoints/{env_name}_{timestamp}"
    
    trainer = LocomotionTrainer(config)
    
    if finetune:
        if not restore_from:
            # Find latest checkpoint
            restore_from = trainer.get_latest_checkpoint()
            if not restore_from:
                console.print("[bold red]No checkpoint found for finetuning. Train base policy first.[/bold red]")
                raise typer.Exit(1)
        
        console.print("[bold yellow]Finetuning with energy penalties...[/bold yellow]")
        
        # Setup finetuning config
        handstand_config = HandstandConfig()
        custom_env_config = {
            'energy_termination_threshold': handstand_config.energy_termination_threshold,
            'reward_config.energy': handstand_config.energy_reward_weight,
            'reward_config.dof_acc': handstand_config.dof_acc_reward_weight,
        }
        
        make_inference_fn, params, metrics = trainer.finetune(
            checkpoint_path=restore_from,
            custom_env_config=custom_env_config,
            num_timesteps=50_000_000  # Shorter finetuning
        )
    else:
        console.print("[bold yellow]Training base handstand policy...[/bold yellow]")
        make_inference_fn, params, metrics = trainer.train(restore_checkpoint_path=restore_from)
    
    console.print("[bold green]Handstand training completed![/bold green]")


@app.command()
def command_sequence(
    env_name: str = typer.Argument(..., help="Environment name"),
    checkpoint_path: str = typer.Argument(..., help="Path to checkpoint"),
    commands: str = typer.Option(
        "0,0,0;1,0,0;1,0,1;0,1,0;-1,0,0", 
        "--commands", 
        help="Semicolon-separated commands (x,y,yaw)"
    ),
    steps_per_command: int = typer.Option(200, "--steps", help="Steps per command"),
    save_video: bool = typer.Option(True, "--save-video/--no-video", help="Save video"),
    show_video: bool = typer.Option(False, "--show-video", help="Display video interactively"),
    video_path: str = typer.Option("videos", "--video-path", help="Directory to save videos"),
    camera: str = typer.Option("track", "--camera", help="Camera view: track, side, front"),
    width: int = typer.Option(640, "--width", help="Video width"),
    height: int = typer.Option(480, "--height", help="Video height"),
):
    """Evaluate policy with a sequence of commands."""
    console.print(f"[bold blue]Running command sequence evaluation for {env_name}[/bold blue]")
    
    # Parse commands
    command_list = []
    for cmd_str in commands.split(';'):
        cmd = [float(x.strip()) for x in cmd_str.split(',')]
        command_list.append(cmd)
    
    console.print(f"Commands: {command_list}")
    
    # Load checkpoint and setup inference
    import jax
    import functools
    from orbax import checkpoint as ocp
    from brax.training.agents.ppo import networks as ppo_networks, train as ppo
    from mujoco_playground.config import locomotion_params
    from mujoco_playground import wrapper
    
    try:
        # Load checkpoint using same pattern as notebook (via ppo.train with restore)
        env = registry.load(env_name)
        ppo_params = locomotion_params.brax_ppo_config(env_name)
        
        # Setup network factory exactly as in training
        network_factory = ppo_networks.make_ppo_networks
        if "network_factory" in ppo_params:
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks,
                **ppo_params.network_factory
            )
        
        # Use ppo.train with num_timesteps=0 to load checkpoint (matching test_evaluate.py)
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
        
        # Setup evaluation config
        config = get_default_eval_config(env_name)
        config.save_video = save_video
        config.video_path = video_path
        config.camera = camera
        config.width = width
        config.height = height
        
        # Run command sequence evaluation
        evaluator = LocomotionEvaluator(config)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description="Running command sequence evaluation...", total=None)
            
            evaluation_data = evaluator.run_command_sequence_evaluation(
                make_inference_fn, params, command_list, steps_per_command
            )
        
        # Create video
        if save_video or show_video:
            console.print("[bold yellow]Creating command sequence video...[/bold yellow]")
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_file = f"{video_path}/{env_name}_command_sequence_{timestamp}.mp4"
            
            # Use video renderer to create video
            frames = evaluator.video_renderer.render(
                evaluator.eval_env,
                evaluation_data['rollout'],
                evaluation_data['modify_scene_fns'],
                video_file if save_video else None
            )
            
            if save_video:
                console.print(f"[bold green]Command sequence video saved to {video_file}[/bold green]")
            
            if show_video:
                console.print("[bold yellow]Displaying command sequence video...[/bold yellow]")
                try:
                    import mediapy as media
                    fps = 1.0 / evaluator.eval_env.dt / config.render_every
                    media.show_video(frames, fps=fps, loop=False)
                    console.print("[bold green]Video displayed successfully![/bold green]")
                except ImportError:
                    console.print("[bold red]mediapy not available for video display[/bold red]")
                except Exception as e:
                    console.print(f"[bold red]Failed to display video: {e}[/bold red]")
        
        # Print summary
        console.print("[bold green]Command sequence evaluation completed![/bold green]")
        console.print(f"Evaluated {len(command_list)} commands with {steps_per_command} steps each")
        
    except Exception as e:
        console.print(f"[bold red]Command sequence evaluation failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def create_config(
    env_name: str = typer.Argument(..., help="Environment name"),
    output_file: str = typer.Argument(..., help="Output config file path"),
    config_type: str = typer.Option("training", "--type", help="Config type: training or eval"),
):
    """Create a default configuration file."""
    if config_type == "training":
        config = get_default_training_config(env_name)
    else:
        config = get_default_eval_config(env_name)
    
    # Convert to dict
    config_dict = {}
    for field in config.__dataclass_fields__:
        config_dict[field] = getattr(config, field)
    
    # Save to file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    console.print(f"[bold green]Config saved to {output_file}[/bold green]")


@app.command()
def test_custom_env(
    timesteps: int = typer.Option(100_000, "--timesteps", "-t", help="Number of training timesteps"),
    seed: int = typer.Option(1, "--seed", help="Random seed"),
):
    """Test custom environment registration and training.
    
    This command demonstrates:
    1. Creating a custom environment outside the library
    2. Registering it with MuJoCo Playground
    3. Training it exactly like built-in environments
    """
    console.print("[bold blue]Testing Custom Environment Registration[/bold blue]")
    
    # Import and register the custom environment
    from mujoco_playground import locomotion
    from src.locomotion_training.envs import Go1CustomIdentical, custom_identical_config
    
    # Register the custom environment
    console.print("1. Registering custom environment...")
    locomotion.register_environment(
        env_name="Go1CustomIdentical",
        env_class=Go1CustomIdentical,
        cfg_class=custom_identical_config
    )
    console.print("[green]✓[/green] Environment registered: Go1CustomIdentical")
    
    # Verify it's in the registry
    console.print("\n2. Verifying registration...")
    all_envs = registry.locomotion.ALL_ENVS
    if "Go1CustomIdentical" in all_envs:
        console.print("[green]✓[/green] Found in registry!")
    else:
        console.print("[red]✗[/red] Not found in registry!")
        raise typer.Exit(1)
    
    # Test loading the environment
    console.print("\n3. Loading custom environment...")
    try:
        env = registry.load("Go1CustomIdentical")
        console.print(f"[green]✓[/green] Environment loaded successfully!")
        console.print(f"   - Action size: {env.action_size}")
        console.print(f"   - Observation size: {env.observation_size}")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load: {e}")
        raise typer.Exit(1)
    
    # Train the custom environment
    console.print(f"\n4. Training custom environment for {timesteps:,} steps...")
    
    # Use the existing train command logic
    config = get_default_training_config("Go1CustomIdentical")
    config.num_timesteps = timesteps
    config.seed = seed
    config.checkpoint_logdir = None  # Disable checkpointing for test
    
    trainer = LocomotionTrainer(config)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description="Training custom environment...", total=None)
            
            make_inference_fn, params, metrics = trainer.train()
        
        console.print("\n[bold green]Success! Custom environment trained successfully![/bold green]")
        console.print(f"Final reward: {metrics.get('eval/episode_reward', 'N/A')}")
        
        # Compare with original
        console.print("\n[bold yellow]Comparison:[/bold yellow]")
        console.print("The custom environment (Go1CustomIdentical) should perform")
        console.print("identically to the original Go1JoystickFlatTerrain since")
        console.print("it inherits all behavior without modifications.")
        
    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()