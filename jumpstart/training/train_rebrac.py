import d3rlpy
import numpy as np
from jumpstart.utils.d3rl_data import get_minari
from jumpstart.utils.data_stats import DatasetStats
from jumpstart.training.utils.factory import get_optimizer_factory
from jumpstart.utils.d3rl_evaluate import evaluate_with_environment
from typing import List, Literal
from jumpstart.utils.parse_loss import parse_loss
from jumpstart.training.utils.learning_curve import make_learning_curve
import tyro
from pathlib import Path


def rebrac_train(
    env: str,
    # --- Algorithm Hyperparameters ---
    batch_size: int = 256,
    actor_learning_rate=0.001,
    critic_learning_rate=0.001,
    actor_beta=0.001,
    critic_beta=0.01,
    gamma: float = 0.99,
    tau=0.005,
    n_critics: int = 2,
    target_smoothing_sigma=0.2,
    target_smoothing_clip=0.5,
    # --- Discrete Action Specific ---
    target_update_interval: int = 8000,
    # --- Optimizer/Scheduler Hyperparameters ---
    n_steps: int | None = None,
    weight_decay: float = 0.01,
    clip_grad_norm: float = 1.0,
    use_scheduler: bool = False,
    # --- Architecture Hyperparameters ---
    hidden_units: List[int] = [256, 256, 256],
    activation: str = "relu",
    dropout: float = 0.0,
    critic_ln: bool = True,
    # --- General ---
    env_type: str | None = None,
    wandb: bool = False,
    tensorboard: bool = False,
    project_name: str = "rebrac-minari",
    compile_graph: bool = True,
    device: str = "cuda",
    enable_ddp: bool = False,
    eval_with_env: bool = True,
    eval_episodes: int = 100,
    eval_num_envs: int = 10,
    val_split: float = 0.0,
    val_percentile: int = 50,
    download: bool = True,
    scale_rewards: bool = False,
    reward_scale: float = 0.0,
    normalize_reward: bool = False,
    eval_while_training: bool = False,
    curve_output: Path | None = None,
    curve_points: int = 10,
    seed: int | None = None,
):
    if seed is not None:
        d3rlpy.seed(seed)
        np.random.seed(seed)

    if n_steps is None:
        n_steps = int((250_000 if "atari" in env else 500_000) / (batch_size / 256))

    data, val_data, environment = get_minari(
        env_name=env,
        download=download,
        val_split=val_split,
        val_percentile=val_percentile,
    )
    stats = DatasetStats.from_dataset(data)
    print(stats)

    # Auto-parse pixel inputs
    if len(stats.obs_dim) > 1:
        import gymnasium

        actor_encoder_config = critic_encoder_config = d3rlpy.models.encoders.PixelEncoderFactory(
            feature_size=hidden_units[-1],
            dropout_rate=dropout,
            activation=activation,
        )
        # Add a wrapper to change image format from NHWC to NCHW for PyTorch
        environment = gymnasium.wrappers.TransformObservation(
            environment,
            lambda obs: np.transpose(obs, (2, 0, 1)),
            observation_space=environment.observation_space,
        )
    else:
        actor_encoder_config = d3rlpy.models.encoders.VectorEncoderFactory(
            hidden_units=hidden_units,
            activation=activation,
            dropout_rate=dropout,
        )
        critic_encoder_config = d3rlpy.models.encoders.VectorEncoderFactory(
            hidden_units=hidden_units,
            activation=activation,
            dropout_rate=dropout,
            use_layer_norm=critic_ln,
        )

    # --- Optimizer and Scheduler Setup ---
    steps_per_epoch = 1000

    reward_scaler = None
    if normalize_reward:
        if "antmaze" not in env.lower():
            raise NotImplementedError("normalize_reward is only implemented for antmaze envs")
        reward_scaler = d3rlpy.preprocessing.MultiplyRewardScaler(100.0)
    elif scale_rewards:
        # if reward_scale > 0 use that, else use StandardScaler
        if reward_scale == 0.0:
            reward_scale = 1 / (stats.std_reward + 1e-8)
        else:
            reward_scale = 1 / reward_scale
        reward_scaler = d3rlpy.preprocessing.MultiplyRewardScaler(reward_scale)

    optim_factory = get_optimizer_factory(
        n_steps=n_steps,
        warmup_steps=2000,
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm,
        use_scheduler=use_scheduler,
    )

    # --- Algorithm Configuration ---
    rebrac_config = d3rlpy.algos.ReBRACConfig(
        batch_size=batch_size,
        gamma=gamma,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        tau=tau,
        actor_beta=actor_beta,
        critic_beta=critic_beta,
        target_smoothing_sigma=target_smoothing_sigma,
        target_smoothing_clip=target_smoothing_clip,
        reward_scaler=reward_scaler,
        actor_optim_factory=optim_factory,
        critic_optim_factory=optim_factory,
        actor_encoder_factory=actor_encoder_config,
        critic_encoder_factory=critic_encoder_config,
        n_critics=n_critics,
        compile_graph=compile_graph,
    )

    # --- Logger and Evaluator Setup ---
    loggers = [d3rlpy.logging.FileAdapterFactory(project_name)]
    if wandb:
        loggers.append(d3rlpy.logging.WanDBAdapterFactory(project=project_name))
    if tensorboard:
        loggers.append(d3rlpy.logging.TensorboardAdapterFactory(root_dir=project_name))
    logger = d3rlpy.logging.CombineAdapterFactory(loggers)

    evaluators = None
    if eval_while_training:
        from jumpstart.utils.d3rl_evaluate import EnvironmentEvaluator

        evaluators = {
            "environment": EnvironmentEvaluator(
                environment,
                n_trials=eval_episodes,
                obs_processor=env,
                n_envs=eval_num_envs,
                vector_env_dataset_name=env,
                vector_env_download=download,
            )
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
    rebrac = rebrac_config.create(device=device, enable_ddp=enable_ddp)
    logs = rebrac.fit(
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
            rebrac,
            environment,
            n_trials=eval_episodes,
            obs_processor=env,
            n_envs=eval_num_envs,
            vector_env_dataset_name=env,
            vector_env_download=download,
        )
    else:
        out_metric = parse_loss(logs[-1])

    return out_metric, rebrac


if __name__ == "__main__":
    # The last element of the last log entry will be the final environment reward
    print(tyro.cli(rebrac_train)[0])
