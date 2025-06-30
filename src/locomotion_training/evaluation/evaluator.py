"""Evaluation module for trained locomotion policies."""

import functools
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable

import jax
import jax.numpy as jp
import numpy as np
from datetime import datetime

from mujoco_playground import registry
from ..config.config import EvalConfig
from ..utils.plotting import (
    plot_foot_trajectories, 
    plot_velocity_tracking, 
    plot_reward_components,
    plot_power_analysis,
    plot_contact_analysis
)
from ..utils.video import VideoRenderer, create_joystick_command_overlay


class LocomotionEvaluator:
    """Evaluator for trained locomotion policies."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.env = registry.load(config.env_name)
        self.env_cfg = registry.get_default_config(config.env_name)
        
        # Apply config modifications
        if config.enable_perturbations and hasattr(self.env_cfg, 'pert_config'):
            self.env_cfg.pert_config.enable = True
            self.env_cfg.pert_config.velocity_kick = [3.0, 6.0]
            self.env_cfg.pert_config.kick_wait_times = [5.0, 15.0]
        
        self.eval_env = registry.load(config.env_name, config=self.env_cfg)
        
        # Setup video renderer
        self.video_renderer = VideoRenderer(
            camera=config.camera,
            width=config.width,
            height=config.height,
            render_every=config.render_every,
            show_contacts=config.show_contact_points,
            show_perturbations=config.show_perturbation_forces
        )
    
    def evaluate_policy(self,
                       make_inference_fn: Callable,
                       params: Any,
                       command: Optional[jp.ndarray] = None,
                       custom_env_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Evaluate a trained policy with comprehensive metrics.
        
        Args:
            make_inference_fn: Function to create inference function
            params: Trained policy parameters
            command: Command to track [x_vel, y_vel, yaw_vel]
            custom_env_config: Custom environment configuration
            
        Returns:
            Dictionary containing evaluation metrics and trajectories
        """
        if command is None:
            command = jp.array([self.config.x_vel, self.config.y_vel, self.config.yaw_vel])
        
        # Setup environment with custom config if provided
        if custom_env_config:
            env_cfg = custom_env_config
            eval_env = registry.load(self.config.env_name, config=env_cfg)
        else:
            eval_env = self.eval_env
        
        # JIT compiled functions
        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)
        jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
        
        # Data collection
        rollout = []
        modify_scene_fns = []
        swing_peak = []
        rewards = []
        linvel = []
        angvel = []
        actions = []
        torques = []
        power = []
        contact_states = []
        
        print(f"Evaluating policy with command: {command}")
        
        rng = jax.random.PRNGKey(42)
        
        for episode in range(self.config.num_episodes):
            print(f"Episode {episode + 1}/{self.config.num_episodes}")
            
            episode_rollout = []
            episode_modify_fns = []
            
            rng, reset_rng = jax.random.split(rng)
            state = jit_reset(reset_rng)
            
            for step in range(self.config.episode_length):
                # Set command
                state.info["command"] = command
                
                # Sample perturbation if enabled
                if (hasattr(state.info, "steps_since_last_pert") and 
                    hasattr(state.info, "steps_until_next_pert")):
                    if state.info["steps_since_last_pert"] < state.info["steps_until_next_pert"]:
                        rng = self._sample_perturbation(rng, state)
                
                # Run inference and step
                act_rng, rng = jax.random.split(rng)
                ctrl, _ = jit_inference_fn(state.obs, act_rng)
                state = jit_step(state, ctrl)
                
                # Early termination check
                if state.done:
                    print(f"Episode {episode + 1} terminated early at step {step}")
                    break
                
                # Collect data
                episode_rollout.append(state)
                actions.append(ctrl)
                
                # Extract metrics
                if hasattr(state.info, 'swing_peak'):
                    swing_peak.append(state.info["swing_peak"])
                
                rewards.append({
                    k[7:]: v for k, v in state.metrics.items() 
                    if k.startswith("reward/")
                })
                
                linvel.append(self.env.get_global_linvel(state.data))
                angvel.append(self.env.get_gyro(state.data))
                torques.append(state.data.actuator_force)
                
                # Calculate power
                qvel = state.data.qvel[6:]  # Exclude base DOFs
                power_instant = jp.sum(jp.abs(qvel * state.data.actuator_force))
                power.append(power_instant)
                
                # Contact states
                if hasattr(state.info, 'last_contact'):
                    contact_states.append(state.info["last_contact"])
                
                # Create command overlay for visualization
                if self.config.save_video:
                    overlay_fn = create_joystick_command_overlay(
                        state, self.env, command,
                        scale_factor=abs(command[0]) / 2.0 if abs(command[0]) > 0 else 1.0
                    )
                    episode_modify_fns.append(overlay_fn)
            
            rollout.extend(episode_rollout)
            modify_scene_fns.extend(episode_modify_fns)
        
        # Convert to arrays
        swing_peak = jp.array(swing_peak) if swing_peak else None
        linvel = jp.array(linvel)
        angvel = jp.array(angvel)
        actions = jp.array(actions)
        torques = jp.array(torques)
        power = jp.array(power)
        contact_states = jp.array(contact_states) if contact_states else None
        
        # Calculate evaluation metrics
        metrics = self._calculate_metrics(
            linvel, angvel, command, rewards, power, torques, actions
        )
        
        # Store trajectories and data
        evaluation_data = {
            'rollout': rollout,
            'modify_scene_fns': modify_scene_fns,
            'swing_peak': swing_peak,
            'rewards': rewards,
            'linvel': linvel,
            'angvel': angvel,
            'actions': actions,
            'torques': torques,
            'power': power,
            'contact_states': contact_states,
            'command': command,
            'metrics': metrics,
            'config': self.config
        }
        
        return evaluation_data
    
    def _sample_perturbation(self, rng: jax.random.PRNGKey, state: Any) -> jax.random.PRNGKey:
        """Sample perturbation parameters."""
        rng, key1, key2 = jax.random.split(rng, 3)
        
        # These would be environment-specific perturbation ranges
        velocity_kick_range = [0.0, 6.0]
        kick_duration_range = [0.05, 0.2]
        
        pert_mag = jax.random.uniform(
            key1, minval=velocity_kick_range[0], maxval=velocity_kick_range[1]
        )
        duration_seconds = jax.random.uniform(
            key2, minval=kick_duration_range[0], maxval=kick_duration_range[1]
        )
        duration_steps = jp.round(duration_seconds / self.eval_env.dt).astype(jp.int32)
        
        # Update state info (note: this modifies the state in-place)
        state.info["pert_mag"] = pert_mag
        state.info["pert_duration"] = duration_steps
        state.info["pert_duration_seconds"] = duration_seconds
        
        return rng
    
    def _calculate_metrics(self,
                          linvel: jp.ndarray,
                          angvel: jp.ndarray,
                          command: jp.ndarray,
                          rewards: List[Dict],
                          power: jp.ndarray,
                          torques: jp.ndarray,
                          actions: jp.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Velocity tracking metrics
        linvel_error = jp.mean(jp.abs(linvel[:, :2] - command[:2]))
        angvel_error = jp.mean(jp.abs(angvel[:, 2] - command[2]))
        
        metrics['linvel_tracking_error'] = float(linvel_error)
        metrics['angvel_tracking_error'] = float(angvel_error)
        metrics['total_tracking_error'] = float(linvel_error + angvel_error)
        
        # Power and torque metrics
        metrics['mean_power'] = float(jp.mean(power))
        metrics['max_power'] = float(jp.max(power))
        metrics['mean_torque'] = float(jp.mean(jp.abs(torques)))
        metrics['max_torque'] = float(jp.max(jp.abs(torques)))
        
        # Action metrics
        metrics['mean_action'] = float(jp.mean(jp.abs(actions)))
        metrics['max_action'] = float(jp.max(jp.abs(actions)))
        metrics['action_smoothness'] = float(jp.mean(jp.abs(jp.diff(actions, axis=0))))
        
        # Reward metrics
        if rewards:
            total_reward = sum(sum(r.values()) for r in rewards)
            metrics['total_reward'] = float(total_reward / len(rewards))
            
            # Individual reward components
            for key in rewards[0].keys():
                values = [r[key] for r in rewards]
                metrics[f'reward_{key}'] = float(np.mean(values))
        
        return metrics
    
    def visualize_results(self, evaluation_data: Dict[str, Any]):
        """Create comprehensive visualization of evaluation results."""
        print("Creating evaluation visualizations...")
        
        # Plot foot trajectories if available
        if evaluation_data['swing_peak'] is not None:
            plot_foot_trajectories(
                evaluation_data['swing_peak'],
                self.env_cfg.reward_config.max_foot_height if hasattr(self.env_cfg, 'reward_config') else 0.15
            )
        
        # Plot velocity tracking
        env_limits = getattr(self.env_cfg.command_config, 'a', [2.0, 2.0, 2*jp.pi])
        plot_velocity_tracking(
            evaluation_data['linvel'],
            evaluation_data['angvel'],
            evaluation_data['command'],
            env_limits
        )
        
        # Plot reward components
        plot_reward_components(evaluation_data['rewards'])
        
        # Plot power analysis
        plot_power_analysis(evaluation_data['power'], evaluation_data['torques'])
        
        # Plot contact analysis if available
        if evaluation_data['contact_states'] is not None:
            plot_contact_analysis(evaluation_data['contact_states'])
        
        # Print metrics summary
        print("\n" + "="*50)
        print("EVALUATION METRICS SUMMARY")
        print("="*50)
        for key, value in evaluation_data['metrics'].items():
            print(f"{key:30s}: {value:8.4f}")
    
    def create_video(self, 
                    evaluation_data: Dict[str, Any],
                    save_path: Optional[str] = None) -> np.ndarray:
        """Create and optionally save evaluation video."""
        if save_path is None and self.config.save_video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{self.config.video_path}/{self.config.env_name}_{timestamp}.mp4"
        
        frames = self.video_renderer.render(
            self.eval_env,
            evaluation_data['rollout'],
            evaluation_data['modify_scene_fns'],
            save_path
        )
        
        return frames
    
    def run_command_sequence_evaluation(self,
                                      make_inference_fn: Callable,
                                      params: Any,
                                      command_sequence: List[List[float]],
                                      steps_per_command: int = 200) -> Dict[str, Any]:
        """Evaluate policy with a sequence of different commands."""
        print("Running command sequence evaluation...")
        
        all_data = []
        combined_rollout = []
        combined_modify_fns = []
        
        for i, cmd in enumerate(command_sequence):
            command = jp.array(cmd)
            print(f"\nEvaluating command {i+1}/{len(command_sequence)}: {command}")
            
            # Temporarily modify episode length
            original_length = self.config.episode_length
            self.config.episode_length = steps_per_command
            
            try:
                eval_data = self.evaluate_policy(make_inference_fn, params, command)
                all_data.append(eval_data)
                combined_rollout.extend(eval_data['rollout'])
                combined_modify_fns.extend(eval_data['modify_scene_fns'])
            finally:
                self.config.episode_length = original_length
        
        # Create combined video
        combined_data = {
            'rollout': combined_rollout,
            'modify_scene_fns': combined_modify_fns,
            'command_sequence': command_sequence,
            'individual_evaluations': all_data
        }
        
        return combined_data