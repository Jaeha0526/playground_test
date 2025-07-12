"""Custom environments for locomotion training."""

# Navigation environments
from .go1_navigation import Go1NavigationEnv
from .hierarchical_wrapper import HierarchicalPolicyWrapper

# Custom environments
from .go1_custom_identical import Go1CustomIdentical, custom_identical_config

# Hierarchical navigation
from .go1_hierarchical_navigation import Go1HierarchicalNavigation, hierarchical_navigation_config
from .hierarchical_navigation_wrapper import HierarchicalNavigationWrapper

__all__ = [
    # Navigation
    "Go1NavigationEnv",
    "HierarchicalPolicyWrapper",
    # Custom
    "Go1CustomIdentical",
    "custom_identical_config",
    # Hierarchical
    "Go1HierarchicalNavigation",
    "hierarchical_navigation_config",
    "HierarchicalNavigationWrapper",
]