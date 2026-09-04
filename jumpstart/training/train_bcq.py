import d3rlpy
import numpy as np
from jumpstart.utils.d3rl_data import get_minari
from jumpstart.utils.data_stats import DatasetStats
from jumpstart.utils.json_enc import SafeJSONEncoder
from jumpstart.utils.d3rl_evaluate import evaluate_with_environment
from jumpstart.utils.parse_loss import parse_loss
from jumpstart.training.utils.factory import (
    get_encoder_factory,
    get_optimizer_factory,
    get_logger_factory,
    get_reward_scaler,
)
from jumpstart.training.utils.learning_curve import make_learning_curve
from typing import List, Literal
import tyro
from pathlib import Path
import json


def bcq_train(
    env: str,
    # --- Algorithm Hyperparameters ---
    batch_size: int = 256,
    actor_learning_rate: float = 1e-3,
    critic_learning_rate: float = 1e-3,
    imitator_learning_rate: float = 1e-3,
    gamma: float = 0.99,
    tau: float = 0.005,
    n_critics: int = 2,
    update_actor_interval: int = 1,
    lam: float = 0.75,
    n_action_samples: int = 100,
    action_flexibility: float = 0.05,
    rl_start_step: int = 0,
    beta: float = 0.5,
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
    # --- General ---
    env_type: Literal["continuous", "discrete"] = "continuous",
    wandb: bool = False,
    tensorboard: bool = False,
    project_name: str = "bcq-minari",
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
    eval_while_training: bool = False,
    curve_output: Path | None = None,
    curve_points: int = 10,
    save_model_path: str | None = None,
    seed: int | None = None,
):
    if seed is not None:
        d3rlpy.seed(seed)
        np.random.seed(seed)

    if n_steps is None:
        n_steps = int((250_000 if "atari" in env else 500_000) / (batch_size / 256))

    if "atari" in env.lower():
        env_type = "discrete"

    # if the save model path files exist, return quietly
    if save_model_path is not None:
        savep = Path(save_model_path)
        if (savep / "model.d3").exists() and (savep / "hyperparams.json").exists():
            print(f"Model and hyperparams already exist at {save_model_path}, exiting")
            exit()

    if "d4rl" in env.lower() and eval_while_training:
        raise ValueError("D4RL is not currently compatible with evaluation during training")

    data, val_data, environment = get_minari(
        env_name=env,
        download=download,
        val_split=val_split,
        val_percentile=val_percentile,
    )
    stats = DatasetStats.from_dataset(data)
    print(stats)

    # Auto-parse pixel inputs
    encoder_config, environment = get_encoder_factory(
        stats.obs_dim,
        environment,
        hidden_units,
        activation,
        dropout,
    )

    # --- Optimizer and Scheduler Setup ---
    steps_per_epoch = 1000

    optim_factory = get_optimizer_factory(
        n_steps=n_steps,
        warmup_steps=2000,
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm,
        use_scheduler=use_scheduler,
    )

    reward_scaler = get_reward_scaler(
        scale_rewards,
        reward_scale,
        stats.std_reward,
    )

    # --- Algorithm Configuration ---
    if env_type == "continuous":
        bcq_config = d3rlpy.algos.BCQConfig(
            batch_size=batch_size,
            gamma=gamma,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            imitator_learning_rate=imitator_learning_rate,
            actor_optim_factory=optim_factory,
            critic_optim_factory=optim_factory,
            imitator_optim_factory=optim_factory,
            tau=tau,
            n_critics=n_critics,
            update_actor_interval=update_actor_interval,
            lam=lam,
            n_action_samples=n_action_samples,
            action_flexibility=action_flexibility,
            rl_start_step=rl_start_step,
            beta=beta,
            reward_scaler=reward_scaler,
            actor_encoder_factory=encoder_config,
            critic_encoder_factory=encoder_config,
            imitator_encoder_factory=encoder_config,
            compile_graph=compile_graph,
        )
    else:  # env_type == "discrete"
        bcq_config = d3rlpy.algos.DiscreteBCQConfig(
            batch_size=batch_size,
            gamma=gamma,
            learning_rate=actor_learning_rate,
            reward_scaler=reward_scaler,
            optim_factory=optim_factory,
            encoder_factory=encoder_config,
            n_critics=n_critics,
            target_update_interval=target_update_interval,
            action_flexibility=action_flexibility,
            beta=beta,
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
    bcq = bcq_config.create(device=device, enable_ddp=enable_ddp)
    logs = bcq.fit(
        data,
        n_steps=n_steps,
        n_steps_per_epoch=steps_per_epoch,
        logger_adapter=logger,
        evaluators=evaluators,
        save_interval=1000,
        callback=curve,
    )

    out_metric = parse_loss(logs[-1])

    if eval_with_env:
        out_metric = evaluate_with_environment(
            bcq,
            environment,
            n_trials=eval_episodes,
            obs_processor=env,
            n_envs=eval_num_envs,
            vector_env_dataset_name=env,
            vector_env_download=download,
        )

    if save_model_path is not None:
        # save in a folder according to the optuna saver:
        # save_model_path/model.d3
        # save_model_path/hyperparams.json
        savep = Path(save_model_path)
        savep.mkdir(parents=True, exist_ok=True)
        bcq.save(savep / "model.d3")
        (savep / "hyperparams.json").write_text(
            json.dumps(bcq.config.serialize_to_dict(), indent=2, cls=SafeJSONEncoder)
        )

    return out_metric, bcq


if __name__ == "__main__":
    # The last element of the last log entry will be the final environment reward
    print(tyro.cli(bcq_train)[0])
