"""Utility functions for inferring environment properties."""

import minari
import gymnasium as gym


def infer_env_type(env_id: str, download: bool = True) -> str:
    """Infer environment type (continuous or discrete) from a Minari dataset.

    Args:
        env_id: The Minari dataset ID
        download: Whether to download the dataset when it is not cached

    Returns:
        "continuous" or "discrete" based on the action space
    """
    dataset = minari.load_dataset(env_id, download=download)
    action_space = dataset.spec.action_space
    if isinstance(action_space, gym.spaces.Box):
        return "continuous"
    if isinstance(action_space, gym.spaces.Discrete):
        return "discrete"
    raise ValueError(
        f"Unsupported action space for {env_id!r}: {type(action_space).__name__}; "
        "expected Box or Discrete."
    )
