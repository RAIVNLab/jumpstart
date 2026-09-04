import numpy as np
import gymnasium as gym
import minari
from d3rlpy.datasets import MDPDataset, Episode
import cv2
from collections import deque
from tqdm import tqdm


def preprocess_atari_observation(obs, frame_stack_buffer=None):
    """Apply Atari preprocessing: resize to 84x84, convert to grayscale.

    Args:
        obs: Raw observation from Atari environment (H, W, C) with values in [0, 255]
        frame_stack_buffer: Optional deque for frame stacking

    Returns:
        Preprocessed observation
    """
    # Resize to 84x84
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    # Convert to grayscale
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    # Add channel dimension (84, 84) -> (1, 84, 84)
    obs = obs[np.newaxis, :, :]

    if frame_stack_buffer is not None:
        frame_stack_buffer.append(obs)
        # Stack frames along channel dimension: (4, 84, 84)
        stacked = np.concatenate(list(frame_stack_buffer), axis=0)
        return stacked

    return obs


def is_atari_dataset(env_name: str) -> bool:
    """Check if the dataset is an Atari dataset."""
    return "atari" in env_name.lower()


def episode_list_to_array(
    episodes,
    goal_condition: bool = False,
    train_percentile=0,
    is_atari: bool = False,
):
    _observations = []
    _actions = []
    _rewards = []
    _next_observations = []
    _terminals = []

    percentiles = [np.sum(ep.rewards) for ep in episodes]
    threshold = np.percentile(percentiles, train_percentile)

    for episode_data in tqdm(episodes, leave=True):
        # if the sum of rewards is below the threshold, skip this episode
        if np.sum(episode_data.rewards) < threshold:
            continue

        observations = episode_data.observations
        actions = episode_data.actions
        rewards = episode_data.rewards

        # Apply Atari preprocessing if needed
        if is_atari and len(observations.shape) == 4:
            # Atari observations: preprocess each frame
            # Frame stack buffer (4 frames)
            frame_stack_buffer = deque(maxlen=4)
            preprocessed_obs = []

            for obs in tqdm(observations, leave=False):
                # Initialize buffer with first frame repeated 4 times
                if len(frame_stack_buffer) == 0:
                    first_frame = preprocess_atari_observation(obs, None)
                    for _ in range(4):
                        frame_stack_buffer.append(first_frame)

                preprocessed = preprocess_atari_observation(obs, frame_stack_buffer)
                preprocessed_obs.append(preprocessed)

            observations = np.array(preprocessed_obs)

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

    def to_mdp_dataset(self, train_percentile=0, is_atari=False):
        """Convert to d3rlpy MDPDataset format"""
        (
            observations,
            actions,
            rewards,
            next_observations,
            terminals,
        ) = episode_list_to_array(
            self.train_episodes,
            self.goal_condition,
            train_percentile,
            is_atari,
        )

        return MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
        )

    def to_filtered_val_dataset(self, reward_percentile=90, is_atari=False):
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

            # Apply Atari preprocessing if needed
            if is_atari and len(obs.shape) == 4:
                # Atari observations: preprocess each frame
                frame_stack_buffer = deque(maxlen=4)
                preprocessed_obs = []

                for frame in obs:
                    # Initialize buffer with first frame repeated 4 times
                    if len(frame_stack_buffer) == 0:
                        first_frame = preprocess_atari_observation(frame, None)
                        for _ in range(4):
                            frame_stack_buffer.append(first_frame)

                    preprocessed = preprocess_atari_observation(frame, frame_stack_buffer)
                    preprocessed_obs.append(preprocessed)

                obs = np.array(preprocessed_obs)

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


def get_minari_chunked(
    env_name,
    action_chunk_size=16,
    download=True,
    val_split=0.0,
    val_percentile=90,
    train_percentile=0.0,
):
    """Load a Minari dataset with action chunking for diffusion/VQ-BeT policies.

    For each transition at index i, collects actions[i:i+K] and flattens to (K*action_dim,).
    Zero-pads at episode boundaries.

    Returns:
        (MDPDataset, val_data, environment, original_action_dim)
    """
    goal_condition = "antmaze" in env_name.lower()
    atari = is_atari_dataset(env_name)
    minari_data = minari.load_dataset(env_name, download=download)

    wrapper = DatasetWrapper(
        minari_data,
        validation_split=val_split,
        goal_condition=goal_condition,
    )

    # Get base arrays from training episodes
    observations, actions, rewards, _, terminals = episode_list_to_array(
        wrapper.train_episodes,
        goal_condition,
        train_percentile,
        is_atari=atari,
    )

    original_action_dim = actions.shape[-1]

    # Compute per-dim action stats before chunking (for normalization)
    action_stat_min = actions.min(axis=0).astype(np.float32)
    action_stat_max = actions.max(axis=0).astype(np.float32)

    # Build chunked actions per episode
    # We need episode boundaries to avoid chunking across episodes
    episodes = wrapper.train_episodes
    chunked_actions_list = []
    idx = 0
    for ep in episodes:
        ep_len = len(ep.actions)
        if np.sum(ep.rewards) < np.percentile(
            [np.sum(e.rewards) for e in episodes], train_percentile
        ):
            idx += ep_len
            continue
        for i in range(ep_len):
            end = min(idx + i + action_chunk_size, idx + ep_len)
            chunk = actions[idx + i : end]
            if len(chunk) < action_chunk_size:
                # Zero-pad at episode boundary
                pad = np.zeros((action_chunk_size - len(chunk), original_action_dim))
                chunk = np.concatenate([chunk, pad], axis=0)
            chunked_actions_list.append(chunk.flatten())
        idx += ep_len

    chunked_actions = np.array(chunked_actions_list, dtype=np.float32)

    dataset = MDPDataset(
        observations=observations,
        actions=chunked_actions,
        rewards=rewards,
        terminals=terminals,
    )

    val_data = (
        wrapper.to_filtered_val_dataset(val_percentile, is_atari=atari) if val_split > 0 else None
    )

    # Recover environment
    try:
        env = minari_data.recover_environment()
        if atari and env is not None:
            import gymnasium as gym

            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayscaleObservation(env)
            env = gym.wrappers.FrameStackObservation(env, 4)
    except Exception as e:
        print(f"Warning: Could not recover environment due to: {e}")
        env = None

    return dataset, val_data, env, original_action_dim, (action_stat_min, action_stat_max)


def get_minari(
    env_name,
    download=True,
    val_split=0.0,
    val_percentile=90,
    train_percentile=0.0,
    env_only=False,
):
    goal_condition = False
    if "antmaze" in env_name.lower():
        goal_condition = True
        # env_name = env_name[:-3]

    # Detect if this is an Atari dataset
    atari = is_atari_dataset(env_name)

    minari_data = minari.load_dataset(env_name, download=download)

    wrapper = DatasetWrapper(
        minari_data,
        validation_split=val_split,
        goal_condition=goal_condition,
    )

    try:
        env = minari_data.recover_environment()

        # Apply Atari wrappers to the environment if needed
        if atari and env is not None:
            import gymnasium as gym

            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayscaleObservation(env)
            env = gym.wrappers.FrameStackObservation(env, 4)

    except Exception as e:
        print(f"Warning: Could not recover environment due to: {e}")
        env = None

    if env_only:
        return env

    return (
        wrapper.to_mdp_dataset(train_percentile, is_atari=atari),
        wrapper.to_filtered_val_dataset(val_percentile, is_atari=atari) if val_split > 0 else None,
        env,
    )
