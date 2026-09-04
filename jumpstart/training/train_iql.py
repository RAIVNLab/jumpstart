import d3rlpy
import numpy as np
from jumpstart.utils.d3rl_data import get_minari
from jumpstart.utils.data_stats import DatasetStats
from jumpstart.training.utils.factory import (
    get_encoder_factory,
    get_optimizer_factory,
    get_logger_factory,
    get_reward_scaler,
)
from jumpstart.training.utils.learning_curve import make_learning_curve
from jumpstart.utils.parse_loss import parse_loss
from jumpstart.utils.d3rl_evaluate import evaluate_with_environment
from typing import List
import tyro
from pathlib import Path


def iql_train(
    env: str,
    # --- Algorithm Hyperparameters ---
    # Best hyperparameters from search (avg score: 6225.51 on walker2d-medium-v0)
    actor_learning_rate: float = 3e-5,
    critic_learning_rate: float = 1e-4,
    batch_size: int = 256,
    gamma: float = 0.995,
    tau: float = 0.005,
    n_critics: int = 2,
    expectile: float = 0.7,
    weight_temp: float = 3.0,
    max_weight: float = 100.0,
    # --- Optimizer/Scheduler Hyperparameters ---
    n_steps: int | None = None,
    weight_decay: float = 0.0,
    clip_grad_norm: float = 1.0,
    use_scheduler: bool = False,
    warmup_steps: int = 2000,
    # --- Architecture Hyperparameters ---
    hidden_units: List[int] = [256, 256, 256],
    activation: str = "relu",
    dropout: float = 0.0,
    critic_ln: bool = False,
    # --- Preprocessing ---
    observation_scaler: str = "standard",
    action_scaler: str = "none",  # 'none', 'min_max', 'standard'
    scale_rewards: bool = True,
    reward_scale: float = 0.0,
    normalize_reward: bool = False,
    # --- General ---
    env_type=None,
    wandb: bool = False,
    tensorboard: bool = False,
    project_name: str = "iql-minari",
    compile_graph: bool = True,
    device: str = "cuda",
    enable_ddp: bool = False,
    eval_with_env: bool = True,
    eval_episodes: int = 100,
    eval_num_envs: int = 10,
    val_split: float = 0.0,
    val_percentile: int = 50,
    download: bool = True,
    eval_while_training: bool = False,
    curve_output: Path | None = None,
    curve_points: int = 10,
    seed: int | None = None,
):
    """
    Train Implicit Q-Learning (IQL) on continuous action space environments.

    Args:
        env: Environment name (e.g., 'hopper-medium-v2', 'halfcheetah-medium-v2')
        actor_learning_rate: Learning rate for policy function
        critic_learning_rate: Learning rate for Q functions
        batch_size: Mini-batch size
        gamma: Discount factor
        tau: Target network synchronization coefficient
        n_critics: Number of Q functions for ensemble
        expectile: Expectile value for value function training (0.7 = IQL paper default)
        weight_temp: Inverse temperature for advantage weighting (beta)
        max_weight: Maximum advantage weight value to clip
        epochs: Number of training epochs
        weight_decay: Weight decay for optimizer
        clip_grad_norm: Gradient clipping threshold
        use_scheduler: Whether to use learning rate scheduler
        warmup_steps: Number of warmup steps for scheduler
        hidden_units: Hidden layer sizes for all networks
        activation: Activation function
        dropout: Dropout rate
        observation_scaler: Observation preprocessing ('none', 'pixel', 'min_max', 'standard')
        action_scaler: Action preprocessing ('none', 'min_max', 'standard')
        scale_rewards: Scale rewards by std deviation
        reward_scale: Manual reward scaling factor
        wandb: Enable Weights & Biases logging
        tensorboard: Enable TensorBoard logging
        project_name: Project name for logging
        compile_graph: Enable JIT compilation and CUDAGraph
        device: Device to use ('cuda', 'cpu', or device index)
        enable_ddp: Enable Data Distributed Parallel training
        eval_with_env: Evaluate on environment after training (final eval)
        eval_episodes: Number of episodes for evaluation
        val_split: Validation split ratio
        val_percentile: Percentile for validation split
        download: Download dataset if not cached
        eval_while_training: Evaluate on environment during training

    Returns:
        Tuple of (trained IQL model, final metric)
    """
    if seed is not None:
        d3rlpy.seed(seed)
        np.random.seed(seed)

    if n_steps is None:
        n_steps = int((250_000 if "atari" in env else 500_000) / (batch_size / 256))

    if "d4rl" in env.lower() and eval_while_training:
        raise ValueError("D4RL is not currently compatible with evaluation during training")

    # --- Data Loading ---
    data, val_data, environment = get_minari(
        env_name=env,
        download=download,
        val_split=val_split,
        val_percentile=val_percentile,
    )
    stats = DatasetStats.from_dataset(data)
    print(stats)

    # --- Encoder Factory ---
    encoder_config, environment = get_encoder_factory(
        stats.obs_dim,
        environment,
        hidden_units,
        activation,
        dropout,
    )

    if critic_ln and len(stats.obs_dim) == 1:
        critic_encoder_config = d3rlpy.models.encoders.VectorEncoderFactory(
            hidden_units=hidden_units,
            activation=activation,
            dropout_rate=dropout,
            use_layer_norm=True,
        )
    else:
        critic_encoder_config = encoder_config

    # --- Optimizer and Scheduler Setup ---
    steps_per_epoch = 1000

    actor_optim_factory = get_optimizer_factory(
        n_steps=n_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm,
        use_scheduler=use_scheduler,
    )

    critic_optim_factory = get_optimizer_factory(
        n_steps=n_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm,
        use_scheduler=use_scheduler,
    )

    # --- Reward Scaler ---
    if normalize_reward:
        if "antmaze" not in env.lower():
            raise NotImplementedError("normalize_reward is only implemented for antmaze envs")
        reward_scaler_obj = d3rlpy.preprocessing.ConstantShiftRewardScaler(-1.0)
    else:
        reward_scaler_obj = get_reward_scaler(
            scale_rewards,
            reward_scale,
            stats.std_reward,
        )

    # --- Preprocessing Scalers ---
    obs_scaler_obj = None
    if observation_scaler == "pixel":
        obs_scaler_obj = d3rlpy.preprocessing.PixelObservationScaler()
    elif observation_scaler == "min_max":
        obs_scaler_obj = d3rlpy.preprocessing.MinMaxObservationScaler()
    elif observation_scaler == "standard":
        obs_scaler_obj = d3rlpy.preprocessing.StandardObservationScaler()

    action_scaler_obj = None
    if action_scaler == "min_max":
        action_scaler_obj = d3rlpy.preprocessing.MinMaxActionScaler()
    elif action_scaler == "standard":
        action_scaler_obj = d3rlpy.preprocessing.StandardActionScaler()

    # --- Algorithm Configuration ---
    iql_config = d3rlpy.algos.IQLConfig(
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        n_critics=n_critics,
        observation_scaler=obs_scaler_obj,
        action_scaler=action_scaler_obj,
        reward_scaler=reward_scaler_obj,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        actor_optim_factory=actor_optim_factory,
        critic_optim_factory=critic_optim_factory,
        actor_encoder_factory=encoder_config,
        critic_encoder_factory=critic_encoder_config,
        value_encoder_factory=critic_encoder_config,
        expectile=expectile,
        weight_temp=weight_temp,
        max_weight=max_weight,
        compile_graph=compile_graph,
    )

    # --- Logger and Evaluator Setup ---
    logger = get_logger_factory(
        project_name,
        wandb,
        tensorboard,
    )

    evaluators = None
    if eval_while_training:
        evaluators = {
            "environment": d3rlpy.metrics.EnvironmentEvaluator(environment, n_trials=eval_episodes)
        }

    curve = make_learning_curve(
        curve_output,
        env,
        n_steps,
        steps_per_epoch,
        curve_points,
        eval_episodes,
        download=download,
    )

    # --- Training ---
    iql = iql_config.create(device=device, enable_ddp=enable_ddp)
    logs = iql.fit(
        data,
        n_steps=n_steps,
        n_steps_per_epoch=steps_per_epoch,
        logger_adapter=logger,
        evaluators=evaluators,
        save_interval=1000,
        callback=curve,
    )

    if eval_with_env:
        out_metric = evaluate_with_environment(
            iql,
            environment,
            n_trials=eval_episodes,
            obs_processor=env,
            n_envs=eval_num_envs,
            vector_env_dataset_name=env,
            vector_env_download=download,
        )
    else:
        out_metric = parse_loss(logs[-1])

    return out_metric, iql


if __name__ == "__main__":
    # The last element of the last log entry will be the final metric
    print(tyro.cli(iql_train)[0])
