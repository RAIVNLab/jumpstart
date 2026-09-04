import numpy as np
from dataclasses import dataclass


@dataclass
class DatasetStats:
    max_reward: float
    min_reward: float
    mean_reward: float
    std_reward: float
    num_episodes: int
    num_transitions: int
    mean_episode_length: float
    min_episode_length: int
    max_episode_length: int
    obs_dim: int
    action_dim: int

    @staticmethod
    def from_dataset(dataset) -> "DatasetStats":
        rewards = [i.compute_return() for i in dataset.episodes]
        num_episodes = len(dataset.episodes)
        num_transitions = dataset.transition_count
        episode_lengths = [len(ep.actions) for ep in dataset.episodes]
        mean_episode_length = np.mean(episode_lengths)
        min_episode_length = np.min(episode_lengths)
        max_episode_length = np.max(episode_lengths)
        obs_dim = dataset.episodes[0].observations.shape[1:]
        act_dim = dataset.episodes[0].actions.shape[1:]

        return DatasetStats(
            max_reward=float(np.max(rewards)),
            min_reward=float(np.min(rewards)),
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            num_episodes=num_episodes,
            num_transitions=num_transitions,
            mean_episode_length=float(mean_episode_length),
            min_episode_length=int(min_episode_length),
            max_episode_length=int(max_episode_length),
            obs_dim=obs_dim,
            action_dim=act_dim,
        )
