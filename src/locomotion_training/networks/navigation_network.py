"""Small navigation network for goal-conditioned velocity command generation."""

import jax
import jax.numpy as jp
import flax.linen as nn
from typing import Sequence


class NavigationNetwork(nn.Module):
    """Small MLP network for navigation policy.
    
    Takes navigation observations (10-dim) and outputs velocity commands (3-dim).
    Architecture: 10 → 64 → 32 → 3
    """
    
    hidden_sizes: Sequence[int] = (64, 32)
    output_size: int = 3  # [vx, vy, vyaw]
    activation: str = "relu"
    
    @nn.compact
    def __call__(self, navigation_obs: jax.Array) -> jax.Array:
        """Forward pass through navigation network.
        
        Args:
            navigation_obs: Navigation observations (10-dim)
                [goal_direction(2), goal_distance(1), robot_pos(2), 
                 robot_heading(1), wall_distances(4)]
        
        Returns:
            velocity_commands: [vx, vy, vyaw] in robot frame
        """
        x = navigation_obs
        
        # Hidden layers
        for hidden_size in self.hidden_sizes:
            x = nn.Dense(hidden_size)(x)
            if self.activation == "relu":
                x = nn.relu(x)
            elif self.activation == "tanh":
                x = nn.tanh(x)
            else:
                raise ValueError(f"Unknown activation: {self.activation}")
        
        # Output layer (no activation for velocity commands)
        velocity_commands = nn.Dense(self.output_size)(x)
        
        # Scale outputs to reasonable velocity ranges
        # Based on Go1 max velocities: vx=1.5, vy=0.8, vyaw=1.2
        velocity_scaling = jp.array([1.5, 0.8, 1.2])
        scaled_commands = nn.tanh(velocity_commands) * velocity_scaling
        
        return scaled_commands


class NavigationValueNetwork(nn.Module):
    """Value network for navigation policy (critic).
    
    Takes navigation observations and estimates value function.
    """
    
    hidden_sizes: Sequence[int] = (64, 64, 32)
    activation: str = "relu"
    
    @nn.compact
    def __call__(self, navigation_obs: jax.Array) -> jax.Array:
        """Forward pass through value network.
        
        Args:
            navigation_obs: Navigation observations (10-dim)
        
        Returns:
            value: Estimated value function (scalar)
        """
        x = navigation_obs
        
        # Hidden layers
        for hidden_size in self.hidden_sizes:
            x = nn.Dense(hidden_size)(x)
            if self.activation == "relu":
                x = nn.relu(x)
            elif self.activation == "tanh":
                x = nn.tanh(x)
            else:
                raise ValueError(f"Unknown activation: {self.activation}")
        
        # Output scalar value
        value = nn.Dense(1)(x)
        return jp.squeeze(value, axis=-1)


def create_navigation_networks(hidden_sizes=(64, 32), activation="relu"):
    """Factory function to create navigation policy and value networks.
    
    Args:
        hidden_sizes: Hidden layer sizes for policy network
        activation: Activation function ("relu" or "tanh")
    
    Returns:
        Tuple of (policy_network, value_network)
    """
    policy_network = NavigationNetwork(
        hidden_sizes=hidden_sizes,
        activation=activation
    )
    
    value_network = NavigationValueNetwork(
        hidden_sizes=hidden_sizes + (32,),  # Slightly larger for value
        activation=activation
    )
    
    return policy_network, value_network


def count_navigation_parameters(network, input_shape=(10,)):
    """Count total parameters in navigation network.
    
    Args:
        network: NavigationNetwork instance
        input_shape: Shape of input observations
    
    Returns:
        Total number of parameters
    """
    rng = jax.random.PRNGKey(0)
    dummy_input = jp.ones(input_shape)
    params = network.init(rng, dummy_input)
    
    def count_params(param_tree):
        return sum(x.size for x in jax.tree_util.tree_leaves(param_tree))
    
    return count_params(params)


# Example usage and testing
if __name__ == "__main__":
    # Create networks
    policy_net, value_net = create_navigation_networks()
    
    # Test shapes
    rng = jax.random.PRNGKey(42)
    dummy_nav_obs = jp.ones((10,))  # 10-dim navigation observation
    
    # Initialize parameters
    policy_params = policy_net.init(rng, dummy_nav_obs)
    value_params = value_net.init(rng, dummy_nav_obs)
    
    # Forward pass
    velocity_commands = policy_net.apply(policy_params, dummy_nav_obs)
    value_estimate = value_net.apply(value_params, dummy_nav_obs)
    
    print(f"Navigation observation shape: {dummy_nav_obs.shape}")
    print(f"Velocity commands shape: {velocity_commands.shape}")
    print(f"Value estimate shape: {value_estimate.shape}")
    print(f"Velocity commands: {velocity_commands}")
    print(f"Value estimate: {value_estimate}")
    
    # Count parameters
    policy_param_count = count_navigation_parameters(policy_net)
    value_param_count = count_navigation_parameters(value_net)
    
    print(f"Policy network parameters: {policy_param_count}")
    print(f"Value network parameters: {value_param_count}")
    print(f"Total navigation parameters: {policy_param_count + value_param_count}")