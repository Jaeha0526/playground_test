#!/usr/bin/env python3
"""
Example script showing how to properly evaluate a trained policy from checkpoint.

This follows the same patterns as the original locomotion.ipynb notebook.
"""

import jax.numpy as jp
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.locomotion_training.config.config import EvalConfig
from src.locomotion_training.evaluation.evaluator import LocomotionEvaluator


def main():
    # Configure evaluation
    eval_config = EvalConfig(
        env_name="Go1JoystickFlatTerrain",
        num_episodes=1,
        episode_length=1000,
        x_vel=1.0,
        y_vel=0.0,
        yaw_vel=0.5,
        enable_perturbations=True,
        save_video=True,
        show_contact_points=True,
        camera="track"
    )
    
    # Create evaluator
    evaluator = LocomotionEvaluator(eval_config)
    
    # Find the best checkpoint (you can also specify a specific checkpoint path)
    checkpoint_dir = Path("checkpoints/Go1JoystickFlatTerrain_20250630_224046")
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        print("Available checkpoints:")
        checkpoints_root = Path("checkpoints")
        if checkpoints_root.exists():
            for checkpoint in checkpoints_root.iterdir():
                if checkpoint.is_dir():
                    print(f"  {checkpoint}")
        return
    
    # Use the best checkpoint if available
    best_checkpoint = checkpoint_dir / "best"
    if best_checkpoint.exists():
        checkpoint_path = str(best_checkpoint)
        print(f"Using best checkpoint: {checkpoint_path}")
    else:
        # Use latest checkpoint
        checkpoints = [p for p in checkpoint_dir.glob("*") if p.is_dir() and p.name.isdigit()]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.name))
            checkpoint_path = str(checkpoints[-1])
            print(f"Using latest checkpoint: {checkpoint_path}")
        else:
            print(f"No valid checkpoints found in {checkpoint_dir}")
            return
    
    print(f"\nEvaluating checkpoint: {checkpoint_path}")
    print(f"Environment: {eval_config.env_name}")
    print(f"Command: [{eval_config.x_vel}, {eval_config.y_vel}, {eval_config.yaw_vel}]")
    print(f"Model ID will be included in video filename")
    print("-" * 50)
    
    try:
        # Evaluate the checkpoint
        evaluation_data = evaluator.evaluate_checkpoint(checkpoint_path)
        
        # Show results
        evaluator.visualize_results(evaluation_data)
        
        # Create video if requested
        if eval_config.save_video:
            print("\nCreating evaluation video...")
            frames = evaluator.create_video(evaluation_data)
            print(f"Video created with {len(frames)} frames")
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("This error is likely due to network architecture mismatch.")
        print("Make sure the checkpoint was created with the same network configuration.")
        raise


def evaluate_command_sequence():
    """Example of evaluating multiple commands in sequence."""
    
    eval_config = EvalConfig(
        env_name="Go1JoystickFlatTerrain",
        num_episodes=1,
        episode_length=300,  # Shorter episodes for sequence
        save_video=True,
        camera="track"
    )
    
    evaluator = LocomotionEvaluator(eval_config)
    
    # Find checkpoint
    checkpoint_dir = Path("checkpoints/Go1JoystickFlatTerrain_20250630_224046")
    best_checkpoint = checkpoint_dir / "best"
    
    if not best_checkpoint.exists():
        print("Best checkpoint not found for command sequence evaluation")
        return
    
    # Load checkpoint once
    make_inference_fn, params = evaluator.load_checkpoint_for_evaluation(str(best_checkpoint))
    
    # Define command sequence: forward, backward, turn left, turn right
    command_sequence = [
        [1.0, 0.0, 0.0],    # Forward
        [-0.5, 0.0, 0.0],   # Backward
        [0.0, 0.0, 1.5],    # Turn left
        [0.0, 0.0, -1.5],   # Turn right
        [0.5, 0.5, 0.5],    # Diagonal + turn
    ]
    
    print("Evaluating command sequence...")
    sequence_data = evaluator.run_command_sequence_evaluation(
        make_inference_fn, params, command_sequence, steps_per_command=200
    )
    
    # Create combined video
    combined_frames = evaluator.create_video(sequence_data)
    print(f"Command sequence video created with {len(combined_frames)} frames")


if __name__ == "__main__":
    print("=== Single Command Evaluation ===")
    main()
    
    print("\n" + "="*60)
    print("=== Command Sequence Evaluation ===")
    evaluate_command_sequence()