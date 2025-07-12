"""Hierarchical policy combining navigation layer with frozen locomotion policy."""

import jax
import jax.numpy as jp
import flax.linen as nn
from typing import Dict, Any, Tuple
from pathlib import Path
import pickle

from brax.training.agents.ppo import networks as ppo_networks
from orbax import checkpoint as ocp
from .navigation_network import NavigationNetwork, NavigationValueNetwork


class HierarchicalNavigationPolicy:
    """Hierarchical policy with navigation layer on top of frozen locomotion policy.
    
    Architecture:
    Navigation observations → NavigationNetwork → Velocity commands
                                                        ↓
    Locomotion observations ← LocomotionNetwork ← Velocity commands + Proprioception
                                                        ↓
                                                  Joint actions
    """
    
    def __init__(
        self,
        locomotion_checkpoint_path: str,
        navigation_hidden_sizes: Tuple[int, ...] = (64, 32),
        navigation_activation: str = "relu",
    ):
        """Initialize hierarchical policy.
        
        Args:
            locomotion_checkpoint_path: Path to pre-trained Go1 locomotion checkpoint
            navigation_hidden_sizes: Hidden layer sizes for navigation network
            navigation_activation: Activation function for navigation network
        """
        self.locomotion_checkpoint_path = locomotion_checkpoint_path
        
        # Create navigation networks
        self.navigation_policy = NavigationNetwork(
            hidden_sizes=navigation_hidden_sizes,
            activation=navigation_activation
        )
        self.navigation_value = NavigationValueNetwork(
            hidden_sizes=navigation_hidden_sizes + (32,),
            activation=navigation_activation
        )
        
        # Locomotion policy will be loaded from checkpoint
        self.locomotion_policy = None
        self.locomotion_params = None
        self.locomotion_normalizer_params = None
        self._locomotion_loaded = False
    
    def load_locomotion_policy(self):
        """Load pre-trained locomotion policy from checkpoint."""
        if self._locomotion_loaded:
            return
        
        try:
            # Load checkpoint using orbax
            checkpointer = ocp.StandardCheckpointer()
            checkpoint_path = Path(self.locomotion_checkpoint_path)
            
            # Convert to absolute path if it's relative
            if not checkpoint_path.is_absolute():
                checkpoint_path = checkpoint_path.resolve()
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            # Load the full checkpoint
            checkpoint_data = checkpointer.restore(str(checkpoint_path))
            
            # Extract locomotion policy parameters
            # Brax/PPO checkpoint format is typically [normalizer_params, policy_params, value_params]
            if isinstance(checkpoint_data, list) and len(checkpoint_data) >= 2:
                self.locomotion_normalizer_params = checkpoint_data[0]  # Normalizer
                self.locomotion_params = checkpoint_data[1]  # Policy parameters
                print(f"Loaded Brax checkpoint: normalizer + policy + value")
            elif isinstance(checkpoint_data, dict):
                if 'params' in checkpoint_data:
                    self.locomotion_params = checkpoint_data['params']
                elif 'policy' in checkpoint_data:
                    self.locomotion_params = checkpoint_data['policy']
                else:
                    print(f"Checkpoint keys: {list(checkpoint_data.keys())}")
                    raise KeyError("Could not find policy parameters in checkpoint")
            else:
                print(f"Unexpected checkpoint format: {type(checkpoint_data)}")
                raise ValueError("Could not parse checkpoint format")
            
            # Create locomotion policy network factory
            # This should match the original Go1 training configuration
            self.locomotion_policy = self._create_locomotion_network()
            
            self._locomotion_loaded = True
            print(f"Successfully loaded locomotion policy from {checkpoint_path}")
            
        except Exception as e:
            print(f"Error loading locomotion policy: {e}")
            print("Will create dummy locomotion policy for testing")
            self._create_dummy_locomotion_policy()
    
    def _create_locomotion_network(self):
        """Create locomotion network matching original Go1 architecture."""
        try:
            # Try to create a standard MLP that matches Go1 architecture
            from brax.training.agents.ppo import networks as ppo_networks
            
            # Create MLP network matching the exact checkpoint structure
            class LocomotionMLP(nn.Module):
                @nn.compact
                def __call__(self, obs):
                    # Match exact architecture from checkpoint parameter inspection
                    x = nn.Dense(512, name='hidden_0')(obs)  # 48 -> 512
                    x = nn.relu(x) 
                    x = nn.Dense(256, name='hidden_1')(x)    # 512 -> 256
                    x = nn.relu(x)
                    x = nn.Dense(128, name='hidden_2')(x)    # 256 -> 128
                    x = nn.relu(x)
                    x = nn.Dense(24, name='hidden_3')(x)     # 128 -> 24 (policy + value outputs)
                    return x
            
            return LocomotionMLP()
            
        except Exception as e:
            print(f"Failed to create locomotion network: {e}")
            return None
    
    def _create_dummy_locomotion_policy(self):
        """Create dummy locomotion policy for testing when checkpoint unavailable."""
        print("Creating dummy locomotion policy - not for actual training!")
        
        class DummyLocomotionPolicy(nn.Module):
            @nn.compact
            def __call__(self, obs):
                # Simple dummy policy that outputs small joint movements
                x = nn.Dense(64)(obs)
                x = nn.relu(x)
                x = nn.Dense(12)(x)
                x = 0.1 * nn.tanh(x)  # Small joint movements
                return x
        
        self.locomotion_policy = DummyLocomotionPolicy()
        
        # Create dummy parameters
        rng = jax.random.PRNGKey(42)
        dummy_obs = jp.ones((48,))  # Typical Go1 observation size
        self.locomotion_params = self.locomotion_policy.init(rng, dummy_obs)
        
        self._locomotion_loaded = True
    
    def init_navigation_params(self, rng: jax.Array, sample_nav_obs: jax.Array):
        """Initialize navigation network parameters.
        
        Args:
            rng: Random key
            sample_nav_obs: Sample navigation observation for initialization
        
        Returns:
            Tuple of (policy_params, value_params)
        """
        rng1, rng2 = jax.random.split(rng)
        
        policy_params = self.navigation_policy.init(rng1, sample_nav_obs)
        value_params = self.navigation_value.init(rng2, sample_nav_obs)
        
        return policy_params, value_params
    
    def extract_navigation_obs(self, full_obs: Dict[str, jax.Array]) -> jax.Array:
        """Extract navigation observations from full observation dict.
        
        Args:
            full_obs: Full observation dictionary from environment
        
        Returns:
            Navigation observations (10-dim)
        """
        if "navigation" in full_obs:
            return full_obs["navigation"]
        else:
            # Fallback: create dummy navigation observations
            # This shouldn't happen in production
            print("Warning: No navigation observations found, using dummy values")
            return jp.zeros((10,))
    
    def extract_locomotion_obs(
        self, 
        full_obs: Dict[str, jax.Array], 
        velocity_commands: jax.Array
    ) -> jax.Array:
        """Extract and modify locomotion observations with velocity commands.
        
        Args:
            full_obs: Full observation dictionary from environment
            velocity_commands: Velocity commands from navigation layer [vx, vy, vyaw]
        
        Returns:
            Locomotion observations with embedded velocity commands
        """
        if "state" in full_obs:
            base_obs = full_obs["state"]
        else:
            # Fallback
            print("Warning: No state observations found")
            base_obs = jp.zeros((45,))  # 48 - 3 for commands
        
        # Original Go1 observation structure (48-dim):
        # [linvel(3), gyro(3), gravity(3), joint_pos(12), joint_vel(12), last_act(12), command(3)]
        
        # Replace the last 3 elements (command) with our navigation commands
        locomotion_obs = base_obs.at[-3:].set(velocity_commands)
        
        return locomotion_obs
    
    def navigation_forward(
        self, 
        navigation_params: Dict[str, Any],
        navigation_obs: jax.Array
    ) -> jax.Array:
        """Forward pass through navigation network.
        
        Args:
            navigation_params: Navigation policy parameters
            navigation_obs: Navigation observations (10-dim)
        
        Returns:
            Velocity commands [vx, vy, vyaw]
        """
        return self.navigation_policy.apply(navigation_params, navigation_obs)
    
    def navigation_value_forward(
        self,
        value_params: Dict[str, Any],
        navigation_obs: jax.Array
    ) -> jax.Array:
        """Forward pass through navigation value network.
        
        Args:
            value_params: Navigation value parameters
            navigation_obs: Navigation observations (10-dim)
        
        Returns:
            Value estimate (scalar)
        """
        return self.navigation_value.apply(value_params, navigation_obs)
    
    def locomotion_forward(
        self, 
        locomotion_obs: jax.Array
    ) -> jax.Array:
        """Forward pass through frozen locomotion network.
        
        Args:
            locomotion_obs: Locomotion observations (48-dim)
        
        Returns:
            Joint actions (12-dim)
        """
        if not self._locomotion_loaded:
            self.load_locomotion_policy()
        
        if self.locomotion_policy is None:
            # If we couldn't create a proper network, use dummy policy
            return self._dummy_locomotion_forward(locomotion_obs)
        
        # Apply frozen locomotion policy  
        frozen_params = jax.lax.stop_gradient(self.locomotion_params)
        network_output = self.locomotion_policy.apply(frozen_params, locomotion_obs)
        
        # Extract policy outputs (first 12 dims) from combined policy+value output (24 dims)
        joint_actions = network_output[..., :12]  # First 12 are joint actions
        
        return joint_actions
    
    def _dummy_locomotion_forward(self, locomotion_obs: jax.Array) -> jax.Array:
        """Dummy locomotion forward for when real policy can't be loaded."""
        # Return small random joint movements as placeholder
        batch_size = locomotion_obs.shape[0] if locomotion_obs.ndim > 1 else 1
        dummy_actions = 0.1 * jax.random.normal(
            jax.random.PRNGKey(0), 
            (batch_size, 12) if locomotion_obs.ndim > 1 else (12,)
        )
        return dummy_actions
    
    def hierarchical_forward(
        self,
        navigation_params: Dict[str, Any],
        full_obs: Dict[str, jax.Array]
    ) -> Tuple[jax.Array, jax.Array]:
        """Complete hierarchical forward pass.
        
        Args:
            navigation_params: Navigation policy parameters (trainable)
            full_obs: Full observation dictionary from environment
        
        Returns:
            Tuple of (joint_actions, velocity_commands)
        """
        # 1. Extract navigation observations
        nav_obs = self.extract_navigation_obs(full_obs)
        
        # 2. Navigation layer: observations → velocity commands
        velocity_commands = self.navigation_forward(navigation_params, nav_obs)
        
        # 3. Prepare locomotion observations
        loco_obs = self.extract_locomotion_obs(full_obs, velocity_commands)
        
        # 4. Locomotion layer: observations → joint actions
        joint_actions = self.locomotion_forward(loco_obs)
        
        return joint_actions, velocity_commands
    
    def get_parameter_count(self):
        """Get parameter counts for both networks.
        
        Returns:
            Dict with parameter counts
        """
        # Navigation parameters (trainable)
        rng = jax.random.PRNGKey(0)
        dummy_nav_obs = jp.ones((10,))
        
        nav_policy_params = self.navigation_policy.init(rng, dummy_nav_obs)
        nav_value_params = self.navigation_value.init(rng, dummy_nav_obs)
        
        def count_params(params):
            return sum(x.size for x in jax.tree_util.tree_leaves(params))
        
        nav_policy_count = count_params(nav_policy_params)
        nav_value_count = count_params(nav_value_params)
        
        # Locomotion parameters (frozen)
        if self._locomotion_loaded and self.locomotion_params:
            loco_count = count_params(self.locomotion_params)
        else:
            loco_count = "Not loaded"
        
        return {
            "navigation_policy": nav_policy_count,
            "navigation_value": nav_value_count,
            "navigation_total": nav_policy_count + nav_value_count,
            "locomotion_total": loco_count,
        }


# Factory function for easy creation
def create_hierarchical_policy(
    locomotion_checkpoint_path: str,
    navigation_hidden_sizes: Tuple[int, ...] = (64, 32),
    navigation_activation: str = "relu"
) -> HierarchicalNavigationPolicy:
    """Create hierarchical navigation policy.
    
    Args:
        locomotion_checkpoint_path: Path to pre-trained Go1 checkpoint
        navigation_hidden_sizes: Hidden layer sizes for navigation network
        navigation_activation: Activation function
    
    Returns:
        HierarchicalNavigationPolicy instance
    """
    return HierarchicalNavigationPolicy(
        locomotion_checkpoint_path=locomotion_checkpoint_path,
        navigation_hidden_sizes=navigation_hidden_sizes,
        navigation_activation=navigation_activation
    )


# Example usage
if __name__ == "__main__":
    # Create hierarchical policy
    checkpoint_path = "/workspace/playground_test/checkpoints/Go1JoystickFlatTerrain_20250630_224046/best/"
    
    policy = create_hierarchical_policy(
        locomotion_checkpoint_path=checkpoint_path,
        navigation_hidden_sizes=(64, 32)
    )
    
    # Test parameter counting
    param_counts = policy.get_parameter_count()
    print("Parameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count}")
    
    # Test forward pass with dummy data
    rng = jax.random.PRNGKey(42)
    dummy_nav_obs = jp.ones((10,))
    
    # Initialize navigation parameters
    nav_policy_params, nav_value_params = policy.init_navigation_params(rng, dummy_nav_obs)
    
    # Create dummy full observation
    dummy_full_obs = {
        "navigation": dummy_nav_obs,
        "state": jp.ones((48,))  # Typical Go1 state
    }
    
    # Test hierarchical forward pass
    joint_actions, velocity_commands = policy.hierarchical_forward(
        nav_policy_params, dummy_full_obs
    )
    
    print(f"Velocity commands: {velocity_commands}")
    print(f"Joint actions shape: {joint_actions.shape}")
    print(f"Joint actions range: [{jp.min(joint_actions):.3f}, {jp.max(joint_actions):.3f}]")