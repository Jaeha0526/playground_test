"""Plotting utilities for training and evaluation visualization."""

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jp
from typing import List, Optional, Tuple
from IPython.display import clear_output, display


def plot_training_progress(x_data: List[int], 
                          y_data: List[float], 
                          y_err: List[float],
                          max_timesteps: int,
                          title: str = "Training Progress"):
    """Plot training progress with error bars."""
    clear_output(wait=True)
    
    plt.figure(figsize=(10, 6))
    plt.xlim([0, max_timesteps * 1.25])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"{title} - Current: {y_data[-1]:.3f}" if y_data else title)
    plt.errorbar(x_data, y_data, yerr=y_err, color="blue", alpha=0.7)
    plt.grid(True, alpha=0.3)
    
    display(plt.gcf())
    plt.close()


def plot_foot_trajectories(swing_peak: jp.ndarray, 
                          max_foot_height: float,
                          foot_names: List[str] = None):
    """Plot foot swing trajectories in a 2x2 grid."""
    if foot_names is None:
        foot_names = ["FR", "FL", "RR", "RL"]
    
    colors = ["r", "g", "b", "y"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    for i, ax in enumerate(axs.flat):
        if i < swing_peak.shape[1]:
            ax.plot(swing_peak[:, i], color=colors[i], linewidth=2)
            ax.set_ylim([0, max_foot_height * 1.25])
            ax.axhline(max_foot_height, color="k", linestyle="--", alpha=0.7)
            ax.set_title(foot_names[i], fontsize=14)
            ax.set_xlabel("time steps")
            ax.set_ylabel("height (m)")
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_velocity_tracking(linvel: jp.ndarray, 
                          angvel: jp.ndarray,
                          command: jp.ndarray,
                          env_limits: Tuple[float, float, float],
                          smooth_window: int = 10):
    """Plot velocity tracking performance."""
    # Smooth the velocities
    linvel_x = jp.convolve(linvel[:, 0], jp.ones(smooth_window) / smooth_window, mode="same")
    linvel_y = jp.convolve(linvel[:, 1], jp.ones(smooth_window) / smooth_window, mode="same")
    angvel_yaw = jp.convolve(angvel[:, 2], jp.ones(smooth_window) / smooth_window, mode="same")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot each velocity component
    axes[0].plot(linvel_x, label="Actual", linewidth=2)
    axes[1].plot(linvel_y, label="Actual", linewidth=2)
    axes[2].plot(angvel_yaw, label="Actual", linewidth=2)
    
    # Set limits and command lines
    axes[0].set_ylim(-env_limits[0], env_limits[0])
    axes[1].set_ylim(-env_limits[1], env_limits[1])
    axes[2].set_ylim(-env_limits[2], env_limits[2])
    
    for i, ax in enumerate(axes):
        ax.axhline(command[i], color="red", linestyle="--", label="Command", linewidth=2)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel("Linear X (m/s)")
    axes[1].set_ylabel("Linear Y (m/s)")
    axes[2].set_ylabel("Angular Z (rad/s)")
    axes[2].set_xlabel("Time steps")
    
    plt.tight_layout()
    plt.show()


def plot_reward_components(rewards: List[dict], 
                          title: str = "Reward Components"):
    """Plot individual reward components over time."""
    if not rewards:
        return
    
    # Extract all unique reward keys
    all_keys = set()
    for reward_dict in rewards:
        all_keys.update(reward_dict.keys())
    
    # Convert to arrays
    reward_data = {}
    for key in all_keys:
        reward_data[key] = [r.get(key, 0.0) for r in rewards]
    
    # Plot rewards
    num_rewards = len(all_keys)
    fig, axes = plt.subplots((num_rewards + 2) // 3, 3, figsize=(15, 4 * ((num_rewards + 2) // 3)))
    if num_rewards == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if num_rewards > 3 else axes
    
    for i, (key, values) in enumerate(reward_data.items()):
        if i < len(axes):
            axes[i].plot(values, linewidth=2)
            axes[i].set_title(key.replace('_', ' ').title())
            axes[i].set_xlabel("Time steps")
            axes[i].set_ylabel("Reward")
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(reward_data), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_power_analysis(power: jp.ndarray, 
                       torques: jp.ndarray,
                       title: str = "Power Analysis"):
    """Plot power consumption and torque analysis."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Power plot
    axes[0].plot(power, linewidth=2, color='red')
    axes[0].set_title("Power Consumption")
    axes[0].set_ylabel("Power (W)")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(jp.mean(power), color="orange", linestyle="--", 
                   label=f"Mean: {jp.mean(power):.2f} W")
    axes[0].legend()
    
    # Torque plot (show max torque per timestep)
    max_torques = jp.max(jp.abs(torques), axis=1)
    axes[1].plot(max_torques, linewidth=2, color='blue')
    axes[1].set_title("Maximum Joint Torque")
    axes[1].set_ylabel("Torque (Nm)")
    axes[1].set_xlabel("Time steps")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(jp.mean(max_torques), color="cyan", linestyle="--",
                   label=f"Mean: {jp.mean(max_torques):.2f} Nm")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Max power: {jp.max(power):.2f} W")
    print(f"Mean power: {jp.mean(power):.2f} W")
    print(f"Max torque: {jp.max(max_torques):.2f} Nm")


def plot_contact_analysis(contact_states: jp.ndarray,
                         foot_names: List[str] = None):
    """Plot contact states for each foot."""
    if foot_names is None:
        foot_names = ["FR", "FL", "RR", "RL"]
    
    fig, axes = plt.subplots(len(foot_names), 1, figsize=(12, 2 * len(foot_names)))
    if len(foot_names) == 1:
        axes = [axes]
    
    for i, (foot_name, ax) in enumerate(zip(foot_names, axes)):
        if i < contact_states.shape[1]:
            contact = contact_states[:, i]
            ax.fill_between(range(len(contact)), contact, alpha=0.7)
            ax.set_title(f"{foot_name} Contact State")
            ax.set_ylabel("Contact")
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time steps")
    plt.tight_layout()
    plt.show()


def save_plot(fig, filepath: str):
    """Save matplotlib figure to file."""
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filepath}")


def create_comparison_plot(data_dict: dict, 
                          title: str = "Comparison",
                          xlabel: str = "Steps",
                          ylabel: str = "Value"):
    """Create comparison plot for multiple data series."""
    plt.figure(figsize=(12, 6))
    
    for label, data in data_dict.items():
        plt.plot(data, label=label, linewidth=2)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()