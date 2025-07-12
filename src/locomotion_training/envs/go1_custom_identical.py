"""Custom Go1 environment that is identical to Go1JoystickFlatTerrain.

This demonstrates how to create a custom environment outside the library
that exactly matches the original Go1JoystickFlatTerrain behavior.
"""

from typing import Optional
from ml_collections import config_dict

# Import from the MuJoCo Playground library
from mujoco_playground._src.locomotion.go1 import joystick


class Go1CustomIdentical(joystick.Joystick):
    """Exact copy of Go1JoystickFlatTerrain for testing custom environments.
    
    This environment is functionally identical to Go1JoystickFlatTerrain.
    It demonstrates how to create custom environments by inheriting from
    the library's base classes.
    """
    
    def __init__(self, 
                 task: str = "flat_terrain", 
                 config: Optional[config_dict.ConfigDict] = None,
                 config_overrides: Optional[dict] = None):
        """Initialize the custom environment.
        
        Args:
            task: Task type ("flat_terrain" or "rough_terrain")
            config: Environment configuration
            config_overrides: Additional config overrides
        """
        # Use the parent's default config if none provided
        if config is None:
            config = self.default_config()
        
        # Initialize parent class with all expected arguments
        super().__init__(task=task, config=config, config_overrides=config_overrides)
    
    @staticmethod
    def default_config() -> config_dict.ConfigDict:
        """Get default configuration (identical to original)."""
        # Start with the parent's default configuration
        return joystick.default_config()


# Optional: Create a custom config function for registration
def custom_identical_config() -> config_dict.ConfigDict:
    """Configuration function for environment registration."""
    return Go1CustomIdentical.default_config()