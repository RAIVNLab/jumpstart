import d3rlpy
import numpy as np
from jumpstart.utils.d3rl_data import get_minari
from jumpstart.utils.data_stats import DatasetStats
from jumpstart.utils.d3rl_scheduler import ChainedSchedulerFactory
from jumpstart.utils.d3rl_evaluate import evaluate_with_environment
from typing import List, Literal
import tyro
import gymnasium as gym


def get_encoder_factory(
    obs_dim,
    environment: gym.Env,
    hidden_units: List[int],
    activation: str,
    dropout: float,
):
    if len(obs_dim) > 1:
        encoder_config = d3rlpy.models.encoders.PixelEncoderFactory(
            feature_size=hidden_units[-1],
            dropout_rate=dropout,
            activation=activation,
        )

    else:
        encoder_config = d3rlpy.models.encoders.VectorEncoderFactory(
            hidden_units=hidden_units,
            activation=activation,
            dropout_rate=dropout,
        )

    return encoder_config, environment


def get_optimizer_factory(
    n_steps: int,
    warmup_steps: int,
    weight_decay: float,
    clip_grad_norm: float,
    use_scheduler: bool,
):
    # d3rlpy doesnt support this by default,
    # check import for implementation of chained
    if use_scheduler:
        # small datasets (e.g. d4rl/pen/human) can have n_steps <= warmup_steps,
        # which makes CosineAnnealingLR's T_max zero and crashes get_lr.
        warmup_steps = min(warmup_steps, max(n_steps - 1, 0))
        warmup_lr = d3rlpy.optimizers.WarmupSchedulerFactory(warmup_steps=warmup_steps)
        cosine_lr = d3rlpy.optimizers.CosineAnnealingLRFactory(
            T_max=max(n_steps - warmup_steps, 1), eta_min=0, last_epoch=-1
        )
        scheduler_factory = ChainedSchedulerFactory(warmup_lr, cosine_lr)
    else:
        scheduler_factory = None

    optimizer = d3rlpy.optimizers.AdamWFactory(
        lr_scheduler_factory=scheduler_factory,
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm,
    )
    return optimizer


def get_logger_factory(
    project_name: str,
    wandb: bool,
    tensorboard: bool,
):
    loggers = [d3rlpy.logging.FileAdapterFactory(project_name)]
    if wandb:
        wandb_logger = d3rlpy.logging.WanDBAdapterFactory(
            project=project_name,
        )
        loggers.append(wandb_logger)

    if tensorboard:
        tb_logger = d3rlpy.logging.TensorboardAdapterFactory(
            root_dir=project_name,
        )
        loggers.append(tb_logger)

    logger = d3rlpy.logging.CombineAdapterFactory(loggers)
    return logger


def get_reward_scaler(
    scale_rewards: bool,
    reward_scale: float,
    std_reward: float,
):
    reward_scaler = None
    if scale_rewards:
        if reward_scale == 0.0:
            reward_scale = 1 / (std_reward + 1e-8)
        else:
            reward_scale = 1 / reward_scale
        reward_scaler = d3rlpy.preprocessing.MultiplyRewardScaler(reward_scale)
    return reward_scaler
