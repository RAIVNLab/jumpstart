# this is to replace out_metric = d3rlpy.metrics.evaluate_qlearning_with_environment and d3rlpy.metrics.evaluate_transformer_with_environment
import numpy as np
import torch
import d3rlpy
import sys
from tqdm import tqdm
from d3rlpy.dataset import ReplayBufferBase
from gymnasium.wrappers import TimeLimit

POINTMAZE_MAX_EPISODE_STEPS = 10000


def call_d3rl_model(d3rl_model, obs, reward=None):
    if reward is not None:
        return d3rl_model.predict(obs, reward)

    return d3rl_model.predict(obs)[0]


def take_obs_key(obs, key="observation"):
    if isinstance(obs, dict):
        return obs[key]
    return obs


def concat_all_keys(obs):
    if isinstance(obs, dict):
        return np.concatenate([v for v in obs.values()], axis=-1)
    return obs


def goal_condition(observations):
    observations = np.concatenate(
        [
            observations["observation"],
            observations["achieved_goal"],
            observations["desired_goal"],
        ],
        axis=-1,
    )
    return observations


def _set_action_horizon(algo, action_horizon: int | None):
    if action_horizon is None or not hasattr(algo, "set_action_horizon"):
        return None
    return algo.set_action_horizon(action_horizon)


def _resolve_action_horizon(
    action_horizon: int | None, diffusion_action_horizon: int | None
) -> int | None:
    return action_horizon if action_horizon is not None else diffusion_action_horizon


def _stack_obs(observations):
    first = observations[0]
    if isinstance(first, dict):
        return {key: _stack_obs([obs[key] for obs in observations]) for key in first}
    if isinstance(first, tuple):
        return tuple(_stack_obs([obs[i] for obs in observations]) for i in range(len(first)))
    return np.stack(observations)


class RecoveredVectorEnv:
    """Small sync vector env that owns episode accounting.

    Gym's SyncVectorEnv rebuilds from env ids; this one accepts thunks so callers
    can clone Minari recovered environments exactly.
    """

    def __init__(self, make_fns):
        self.envs = [make_fn() for make_fn in make_fns]
        self.returns = np.zeros(len(self.envs), dtype=np.float64)
        self.lengths = np.zeros(len(self.envs), dtype=np.int64)

    def reset(self):
        observations = []
        for env in self.envs:
            out = env.reset()
            observations.append(out[0] if isinstance(out, tuple) else out)
        self.returns.fill(0.0)
        self.lengths.fill(0)
        return _stack_obs(observations)

    def step(self, actions):
        observations, rewards, dones, infos = [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            out = env.step(action)
            if len(out) == 5:
                obs, reward, terminated, truncated, info = out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = out
                done = bool(done)

            reward = float(reward)
            self.returns[i] += reward
            self.lengths[i] += 1
            info = dict(info)

            if done:
                info["episode"] = {
                    "r": float(self.returns[i]),
                    "l": int(self.lengths[i]),
                }
                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                self.returns[i] = 0.0
                self.lengths[i] = 0

            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return _stack_obs(observations), np.asarray(rewards), np.asarray(dones), infos

    def close(self):
        for env in self.envs:
            env.close()


def _make_recovered_env_fn(dataset_name: str, download: bool = False):
    def thunk():
        from jumpstart.utils.d3rl_data import get_minari

        env = get_minari(
            env_name=dataset_name,
            download=download,
            val_split=0.0,
            env_only=True,
        )
        if env is None:
            raise RuntimeError(f"could not recover environment for {dataset_name}")
        return _wrap_pointmaze_time_limit(env, dataset_name)

    return thunk


def _wrap_pointmaze_time_limit(env, name: str | None = None):
    env_id = getattr(getattr(env, "spec", None), "id", "") or ""
    if "pointmaze" not in f"{name or ''} {env_id}".lower():
        return env
    if isinstance(env, TimeLimit):
        env._max_episode_steps = POINTMAZE_MAX_EPISODE_STEPS
        return env
    return TimeLimit(env, max_episode_steps=POINTMAZE_MAX_EPISODE_STEPS)


def evaluate_qlearning_vectorized(
    algo, make_fn, obs_processor, n_trials, n_envs, progress_bar=False
):
    """Evaluate using recovered env thunks for parallel episode collection."""
    envs = RecoveredVectorEnv([make_fn for _ in range(n_envs)])

    try:
        obs = envs.reset()
        episode_rewards = []
        pbar = tqdm(total=n_trials, desc="Evaluating (vec)") if progress_bar else None

        if hasattr(algo, "reset"):
            algo.reset()

        while len(episode_rewards) < n_trials:
            if obs_processor is not None:
                obs = obs_processor(obs)

            # Vectorized env already provides batch dim (n_envs, ...)
            actions = algo.predict(obs)

            obs, rewards, dones, infos = envs.step(actions)

            prev_count = len(episode_rewards)
            for done, info in zip(dones, infos):
                if done and "episode" in info:
                    episode_rewards.append(float(info["episode"]["r"]))
            if pbar is not None:
                pbar.update(len(episode_rewards) - prev_count)

            # Reset action cache for chunked policies (diffusion/VQBeT) per-env
            if hasattr(algo, "reset") and any(dones):
                done_mask = torch.tensor(dones, dtype=torch.bool)
                algo.reset(done_mask)

        if pbar is not None:
            pbar.close()
        return float(np.mean(episode_rewards[:n_trials]))
    finally:
        envs.close()


def evaluate_qlearning_with_environment(
    algo,
    env,
    obs_processor=concat_all_keys,
    n_trials: int = 10,
    progress_bar: bool = False,
    n_envs: int = 1,
    vector_env_factory=None,
) -> float:
    if n_envs > 1 and vector_env_factory is not None:
        return evaluate_qlearning_vectorized(
            algo, vector_env_factory, obs_processor, n_trials, n_envs, progress_bar
        )

    # Sequential fallback
    episode_rewards = []

    it = tqdm(range(n_trials), desc="Evaluating") if progress_bar else range(n_trials)

    for _ in it:
        if hasattr(algo, "reset"):
            algo.reset()
        observation, _ = env.reset()
        episode_reward = 0.0

        while True:
            if obs_processor is not None:
                observation = obs_processor(observation)

            # take action
            if isinstance(observation, np.ndarray):
                observation = np.expand_dims(observation, axis=0)
            elif isinstance(observation, (tuple, list)):
                observation = [np.expand_dims(o, axis=0) for o in observation]
            else:
                raise ValueError("Unsupported observation type")

            action = call_d3rl_model(algo, observation)

            observation, reward, done, truncated, _ = env.step(action)
            episode_reward += float(reward)

            if done or truncated:
                break
        episode_rewards.append(episode_reward)
    return float(np.mean(episode_rewards))


def evaluate_transformer_with_environment(
    algo,
    env,
    obs_processor=concat_all_keys,
    n_trials: int = 10,
    progress_bar: bool = False,
) -> float:
    episode_rewards = []

    it = range(n_trials) if not progress_bar else tqdm.trange(n_trials)

    for _ in it:
        algo.reset()
        observation, reward = env.reset()[0], 0.0
        episode_reward = 0.0

        while True:
            if obs_processor is not None:
                observation = obs_processor(observation)

            # if isinstance(observation, np.ndarray):
            #     observation = np.expand_dims(observation, axis=0)
            # elif isinstance(observation, (tuple, list)):
            #     observation = [np.expand_dims(o, axis=0) for o in observation]
            # else:
            #     raise ValueError("Unsupported observation type")

            # take action
            action = call_d3rl_model(algo, observation, reward)

            observation, _reward, done, truncated, _ = env.step(action)
            reward = float(_reward)
            episode_reward += reward

            if done or truncated:
                break
        episode_rewards.append(episode_reward)
    return float(np.mean(episode_rewards))


def evaluate_with_environment(
    algo,
    env,
    obs_processor=take_obs_key,
    reward=None,
    n_trials: int = 10,
    progress_bar: bool = False,
    n_envs: int = 1,
    action_horizon: int | None = None,
    diffusion_action_horizon: int | None = None,
    vector_env_dataset_name: str | None = None,
    vector_env_download: bool = False,
) -> float:
    obs_processor_name = obs_processor if isinstance(obs_processor, str) else None
    if obs_processor is None or obs_processor == take_obs_key:
        if obs_processor is None:
            obs_processor = take_obs_key

    if isinstance(obs_processor, str):
        if obs_processor == "concat_all_keys":
            obs_processor = concat_all_keys
        elif obs_processor == "take_obs_key":
            obs_processor = take_obs_key
        elif "antmaze" in obs_processor:
            obs_processor = goal_condition
        else:
            print(
                f"Unknown obs_processor string: {obs_processor}, falling back to take_obs_key",
                file=sys.stderr,
            )
            obs_processor = take_obs_key

    env = _wrap_pointmaze_time_limit(env, obs_processor_name)

    old_action_horizon = _set_action_horizon(
        algo, _resolve_action_horizon(action_horizon, diffusion_action_horizon)
    )
    try:
        if isinstance(algo, d3rlpy.algos.decision_transformer.TransformerAlgoBase):
            assert reward is not None, "Reward must be provided for transformer evaluation"
            return evaluate_transformer_with_environment(
                algo.as_stateful_wrapper(target_return=reward),
                env,
                obs_processor,
                n_trials,
                progress_bar,
            )
        else:
            return evaluate_qlearning_with_environment(
                algo,
                env,
                obs_processor,
                n_trials,
                progress_bar,
                n_envs=n_envs,
                vector_env_factory=_make_recovered_env_fn(
                    vector_env_dataset_name, vector_env_download
                )
                if vector_env_dataset_name is not None and n_envs > 1
                else None,
            )
    finally:
        # Close environment to prevent EGL cleanup errors
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass  # Ignore any errors during cleanup
        _set_action_horizon(algo, old_action_horizon)


class EnvironmentEvaluator:
    """Environment evaluator that properly handles obs_processor.

    This evaluator wraps evaluate_with_environment to provide a compatible
    interface with d3rlpy's evaluation system while supporting custom
    observation processors (like atari_obs_processor).

    Args:
        env: Gym environment.
        n_trials: Number of episodes to evaluate.
        obs_processor: Observation processor function, string, or None for auto-detection.
        reward: Target return for transformer models.
    """

    def __init__(
        self,
        env,
        n_trials: int = 10,
        obs_processor=None,
        reward=None,
        n_envs: int = 1,
        action_horizon: int | None = None,
        diffusion_action_horizon: int | None = None,
        vector_env_dataset_name: str | None = None,
        vector_env_download: bool = False,
    ):
        self._env = env
        self._n_trials = n_trials
        self._obs_processor = obs_processor
        self._reward = reward
        self._n_envs = n_envs
        self._action_horizon = _resolve_action_horizon(action_horizon, diffusion_action_horizon)
        self._vector_env_dataset_name = vector_env_dataset_name
        self._vector_env_download = vector_env_download

    def __call__(self, algo, dataset: "ReplayBufferBase") -> float:
        """Evaluate the algorithm on the environment.

        Args:
            algo: Algorithm to evaluate.
            dataset: Replay buffer (not used, for compatibility).

        Returns:
            Average episode reward.
        """
        return evaluate_with_environment(
            algo=algo,
            env=self._env,
            n_trials=self._n_trials,
            obs_processor=self._obs_processor,
            reward=self._reward,
            n_envs=self._n_envs,
            action_horizon=self._action_horizon,
            vector_env_dataset_name=self._vector_env_dataset_name,
            vector_env_download=self._vector_env_download,
        )
