import numpy as np
import minari
from d3rlpy.datasets import MDPDataset, Episode
from d3rlpy.dataset import create_fifo_replay_buffer


def get_minari_with_buffer(
    env_name,
    download=True,
    buffer_limit=None,
    max_episodes=None,
    goal_condition=False,
):
    """
    Load Minari dataset using ReplayBuffer for memory efficiency.

    Args:
        env_name: Minari dataset ID (e.g., "mujoco/halfcheetah/medium-v0")
        download: Whether to download if not found locally
        buffer_limit: Max number of transitions to store (None = use all)
        max_episodes: Max episodes to load (None = load all)
        goal_condition: Whether to concatenate goal observations

    Returns:
        tuple: (MDPDataset, None, environment)
    """
    if env_name.endswith("-GC"):
        goal_condition = True
        env_name = env_name[:-3]

    print(f"Loading Minari dataset: {env_name}")
    minari_data = minari.load_dataset(env_name, download=download)

    # Determine buffer size
    if buffer_limit is None:
        buffer_limit = minari_data.total_steps
        print(f"Using buffer for all {buffer_limit:,} transitions")
    else:
        print(f"Using buffer with limit: {buffer_limit:,} transitions")
        if buffer_limit < minari_data.total_steps:
            print(
                f"⚠️  Warning: Buffer can only hold {buffer_limit:,} out of {minari_data.total_steps:,} transitions"
            )
            print(
                f"   Oldest ~{minari_data.total_steps - buffer_limit:,} transitions will be dropped (FIFO)"
            )

    # Load first episode to determine shapes and initialize buffer
    print("Loading first episode to determine shapes...")
    episode_iterator = minari_data.iterate_episodes()
    first_ep_raw = next(episode_iterator)

    # Process first episode
    obs = first_ep_raw.observations
    if isinstance(obs, dict):
        if goal_condition:
            obs = np.concatenate(
                [
                    obs["observation"],
                    obs["achieved_goal"],
                    obs["desired_goal"],
                ],
                axis=-1,
            )
        else:
            obs = obs["observation"]

    # Reshape images if needed
    if len(obs.shape) > 3:
        obs = np.transpose(obs, (0, 3, 1, 2))  # H,W,C -> C,H,W

    terminals = (
        first_ep_raw.terminations | first_ep_raw.truncations
        if hasattr(first_ep_raw, "terminations")
        else np.zeros(len(first_ep_raw.actions), dtype=bool)
    )
    terminals[-1] = True

    # Create d3rlpy Episode object
    from d3rlpy.dataset import Episode as D3Episode

    first_episode = D3Episode(
        observations=obs.astype(np.float32),
        actions=first_ep_raw.actions.astype(np.float32),
        rewards=first_ep_raw.rewards.astype(np.float32),
        terminated=bool(np.any(terminals)),
    )

    print(f"  Observation shape: {obs[0].shape}")
    print(f"  Action shape: {first_ep_raw.actions[0].shape}")

    # Create replay buffer with first episode to infer shapes
    buffer = create_fifo_replay_buffer(
        limit=buffer_limit,
        episodes=[first_episode],  # Pass first episode to infer shapes
    )

    # Determine how many episodes to load
    total_to_load = max_episodes if max_episodes else minari_data.total_episodes
    print(f"Loading {total_to_load} episodes...")

    # Helper function to add episode to buffer
    def add_episode_to_buffer(ep, episode_idx):
        nonlocal transitions_loaded

        if episode_idx % 50 == 0 and episode_idx > 0:
            print(
                f"  Loaded {episode_idx}/{total_to_load} episodes ({transitions_loaded:,} transitions)"
            )

        # Extract observations
        obs = ep.observations

        # Handle dict observations (e.g., AntMaze with goals)
        if isinstance(obs, dict):
            if goal_condition:
                obs = np.concatenate(
                    [
                        obs["observation"],
                        obs["achieved_goal"],
                        obs["desired_goal"],
                    ],
                    axis=-1,
                )
            else:
                obs = obs["observation"]

        # Reshape images from (H, W, C) → (C, H, W) for PyTorch
        if len(obs.shape) > 3:
            obs = np.transpose(obs, (0, 3, 1, 2))

        # Get termination flags
        if hasattr(ep, "terminations"):
            terminals = ep.terminations | ep.truncations
        elif hasattr(ep, "dones"):
            terminals = ep.dones
        else:
            terminals = np.zeros(len(ep.actions), dtype=bool)
            terminals[-1] = True  # Mark last step as terminal

        # Create d3rlpy Episode object
        from d3rlpy.dataset import Episode as D3Episode

        d3_episode = D3Episode(
            observations=obs.astype(np.float32),
            actions=ep.actions.astype(np.float32),
            rewards=ep.rewards.astype(np.float32),
            terminated=bool(np.any(terminals)),
        )

        # Add episode to buffer (this is where memory efficiency happens!)
        buffer.append_episode(d3_episode)

        transitions_loaded += len(ep.actions)

    # Load episodes incrementally (memory efficient!)
    transitions_loaded = len(first_ep_raw.actions)  # First episode already in buffer
    print(f"  Episode 0: {transitions_loaded} transitions (already loaded)")

    # Add remaining episodes one at a time (continue from same iterator)
    episodes_loaded = 1
    for i, episode in enumerate(episode_iterator, start=1):
        if max_episodes and i >= max_episodes:
            break

        add_episode_to_buffer(episode, i)
        episodes_loaded += 1

    print(f"✓ Loaded {episodes_loaded} episodes with {transitions_loaded:,} transitions")

    # Convert buffer episodes to numpy arrays (same as d3rl_data.py does)
    print("Converting buffer to MDPDataset...")

    # Extract all episodes from buffer
    episodes = list(buffer.episodes)

    # Flatten episodes into transitions (like episode_list_to_array)
    _observations = []
    _actions = []
    _rewards = []
    _terminals = []

    for episode in episodes:
        # Each Episode object has: observations, actions, rewards, terminated
        obs = episode.observations
        acts = episode.actions
        rews = episode.rewards

        # Add all transitions from this episode
        for i in range(len(acts)):
            _observations.append(obs[i])
            _actions.append(acts[i])
            _rewards.append(rews[i])
            # Mark last transition as terminal if episode terminated
            _terminals.append(1.0 if (i == len(acts) - 1 and episode.terminated) else 0.0)

    # Convert to numpy arrays
    observations = np.array(_observations, dtype=np.float32)
    actions = np.array(_actions, dtype=np.float32)
    rewards = np.array(_rewards, dtype=np.float32)
    terminals = np.array(_terminals, dtype=np.float32)

    # Create MDPDataset (same way as d3rl_data.py)
    mdp_dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    print(f"✓ Created MDPDataset with {mdp_dataset.transition_count:,} transitions")

    # Try to recover environment
    try:
        env = minari_data.recover_environment()
    except Exception as e:
        print(f"Warning: Could not recover environment due to: {e}")
        env = None

    return mdp_dataset, None, env


# Keep original get_minari for backward compatibility
def episode_list_to_array(episodes, goal_condition: bool = False, train_percentile=0):
    _observations = []
    _actions = []
    _rewards = []
    _next_observations = []
    _terminals = []

    percentiles = [np.sum(ep.rewards) for ep in episodes]
    threshold = np.percentile(percentiles, train_percentile)

    for episode_data in episodes:
        # if the sum of rewards is below the threshold, skip this episode
        if np.sum(episode_data.rewards) < threshold:
            continue

        observations = episode_data.observations
        actions = episode_data.actions
        rewards = episode_data.rewards
        # special case: AntMaze
        # if observation is a dict, then take the "observation key", come back and fill in "achieved_goal" and "desired_goal" later
        # todo- add something like -GC to the end to see if you want to add the goals concatenated in
        if isinstance(observations, dict):
            if goal_condition:
                # concat achieved and desired goals to the observation
                observations = np.concatenate(
                    [
                        observations["observation"],
                        observations["achieved_goal"],
                        observations["desired_goal"],
                    ],
                    axis=-1,
                )
            else:
                observations = observations["observation"]

        if hasattr(episode_data, "terminations"):
            terminations = episode_data.terminations
            truncations = episode_data.truncations
        elif hasattr(episode_data, "dones"):
            terminations = episode_data.dones
            truncations = [False] * len(actions)
        else:
            terminations = [False] * len(actions)
            truncations = [False] * len(actions)

        # For each transition in the episode
        for i in range(len(actions)):
            _observations.append(observations[i])
            _actions.append(actions[i])
            _rewards.append(rewards[i])
            _terminals.append(terminations[i] or truncations[i])

            # Next observation
            if i + 1 < len(observations):
                _next_observations.append(observations[i + 1])
            else:
                # Last observation in episode, use same obs
                _next_observations.append(observations[i])

    observations = np.array(_observations)
    actions = np.array(_actions)
    next_observations = np.array(_next_observations)
    rewards = np.array(_rewards)
    terminals = np.array(_terminals)

    # minari seems to not save in pytorch format.
    # we should reshape the observations if there are 4 dims
    # current: N x H x W x C
    # new: N x C x H x W
    if len(observations.shape) > 3:
        observations = np.transpose(observations, (0, 3, 1, 2))
        next_observations = np.transpose(next_observations, (0, 3, 1, 2))

    return (
        observations,
        actions,
        rewards,
        next_observations,
        terminals,
    )


class DatasetWrapper:
    def __init__(
        self,
        minari_dataset: minari.MinariDataset,
        validation_split=0.1,
        goal_condition: bool = False,
    ):
        self.dataset = minari_dataset
        self.validation_split = validation_split
        self.goal_condition = goal_condition

        episodes = list(minari_dataset.iterate_episodes())
        n_val_episodes = int(len(minari_dataset) * validation_split)
        self.val_episodes = episodes[:n_val_episodes]
        self.train_episodes = episodes[n_val_episodes:]
        self.episodes_count = len(self.train_episodes)

        self.transition_count = sum(len(ep.actions) for ep in self.train_episodes)

    def to_mdp_dataset(self, train_percentile=0):
        """Convert to d3rlpy MDPDataset format"""
        (
            observations,
            actions,
            rewards,
            next_observations,
            terminals,
        ) = episode_list_to_array(self.train_episodes, self.goal_condition, train_percentile)

        return MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
        )

    def to_filtered_val_dataset(self, reward_percentile=90):
        val_episodes = self.val_episodes

        if len(val_episodes) == 0:
            return None

        rewards = [np.sum(ep.rewards) for ep in val_episodes]
        threshold = np.percentile(rewards, reward_percentile)
        filtered_episodes = [ep for ep, r in zip(val_episodes, rewards) if r >= threshold]

        if len(filtered_episodes) == 0:
            print("Warning: No episodes above the reward threshold, using all val episodes")
            filtered_episodes = val_episodes

        obs_maybe_reshaped = []
        for ep in filtered_episodes:
            # D4RL AntMaze special case: stores observations as dicts
            obs = ep.observations
            if isinstance(ep.observations, dict):
                obs = ep.observations["observation"]
                # todo- add something like -GC to the end to see if you want to add the goals concatenated in

            if len(obs.shape) > 3:
                obs_maybe_reshaped.append(np.transpose(obs, (0, 3, 1, 2)))
            else:
                obs_maybe_reshaped.append(obs)

        return [
            Episode(
                observations=np.expand_dims(obs, axis=1) if len(obs.shape) == 1 else obs,
                actions=np.expand_dims(ep.actions, axis=1)
                if len(ep.actions.shape) == 1
                else ep.actions,
                rewards=np.expand_dims(ep.rewards, axis=1)
                if len(ep.rewards.shape) == 1
                else ep.rewards,
                terminated=np.any(ep.terminations | ep.truncations),
            )
            for obs, ep in zip(obs_maybe_reshaped, filtered_episodes)
        ]


def get_minari(
    env_name,
    download=True,
    val_split=0.0,
    val_percentile=90,
    train_percentile=0.0,
):
    """Original get_minari - loads all data into memory at once."""
    goal_condition = False
    if env_name.endswith("-GC"):
        goal_condition = True
        env_name = env_name[:-3]

    minari_data = minari.load_dataset(env_name, download=download)
    wrapper = DatasetWrapper(
        minari_data,
        validation_split=val_split,
        goal_condition=goal_condition,
    )

    try:
        env = minari_data.recover_environment()
    except Exception as e:
        print(f"Warning: Could not recover environment due to: {e}")
        env = None

    return (
        wrapper.to_mdp_dataset(train_percentile),
        wrapper.to_filtered_val_dataset(val_percentile) if val_split > 0 else None,
        env,
    )
