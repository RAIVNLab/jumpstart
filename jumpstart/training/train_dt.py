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
from jumpstart.utils.json_enc import SafeJSONEncoder
from jumpstart.utils.parse_loss import parse_loss
from jumpstart.utils.d3rl_evaluate import (
    evaluate_with_environment,
    take_obs_key,
    goal_condition,
)
from typing import List, Literal, Optional
import tyro
from pathlib import Path
import json


def dt_train(
    env: str,
    # --- Transformer Hyperparameters ---
    context_size: int = 30,
    hidden_units: List[int] = [256],  # The last element is the embedding dim
    num_heads: int = 1,
    num_layers: int = 3,
    attn_dropout: float = 0.1,
    resid_dropout: float = 0.1,
    embed_dropout: float = 0.1,
    all_dropout: float = 0.0,  # override all dropout rates if > 0
    gamma: float = 0.99,
    # --- Training Hyperparameters ---
    learning_rate: float = 1e-4,
    batch_size: int = 256,
    epochs: int | None = None,
    weight_decay: float = 0.01,
    clip_grad_norm: float = 0.25,
    warmup_steps: int = 1000,
    # --- Environment and Data ---
    env_type: Literal["continuous", "discrete"] = "continuous",
    val_split: float = 0.0,
    val_percentile: int = 50,
    download: bool = True,
    # --- Evaluation ---
    target_return: Optional[float] = None,
    # --- System and Logging ---
    wandb: bool = False,
    tensorboard: bool = False,
    project_name: str = "dt-minari",
    compile_graph: bool = True,
    device: str = "cuda",
    enable_ddp: bool = False,
    eval_with_env: bool = True,
    eval_episodes: int = 100,
    eval_num_envs: int = 10,
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

    if epochs is None:
        if "atari" not in env:
            epochs = 50
        else:
            # try to normalize based on the transition count since it varies widely
            epochs = int(250 / ((stats.num_transitions / batch_size) / 60))

    if "atari" in env.lower():
        env_type = "discrete"

    encoder_config, environment = get_encoder_factory(
        stats.obs_dim,
        environment,
        hidden_units,
        "relu",
        embed_dropout,
    )

    n_steps = int(np.ceil(stats.num_transitions / batch_size)) * epochs
    steps_per_epoch = int(np.ceil(stats.num_transitions / batch_size))

    if "atari" in env.lower():
        warmup_steps = max(1, warmup_steps // 10)

    optimizer = get_optimizer_factory(
        n_steps=n_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm,
        use_scheduler=True,
    )

    dt_cls = (
        d3rlpy.algos.DiscreteDecisionTransformerConfig
        if env_type == "discrete"
        else d3rlpy.algos.DecisionTransformerConfig
    )

    reward_scaler = get_reward_scaler(
        scale_rewards,
        reward_scale,
        stats.std_reward,
    )

    dt_config = dt_cls(
        batch_size=batch_size,
        learning_rate=learning_rate,
        optim_factory=optimizer,
        encoder_factory=encoder_config,
        context_size=context_size,
        num_heads=num_heads,
        num_layers=num_layers,
        attn_dropout=attn_dropout if all_dropout == 0.0 else all_dropout,
        resid_dropout=resid_dropout if all_dropout == 0.0 else all_dropout,
        embed_dropout=embed_dropout if all_dropout == 0.0 else all_dropout,
        max_timestep=stats.max_episode_length,
        gamma=gamma,
        reward_scaler=reward_scaler,
        compile_graph=compile_graph,
    )

    # --- Set up Logging ---
    logger = get_logger_factory(
        project_name,
        wandb,
        tensorboard,
    )

    # Use the max return from the dataset as the target if not specified
    if target_return is None:
        target_return = stats.max_reward
        print(f"Using max dataset return as target for evaluation: {target_return}")

    curve = make_learning_curve(
        curve_output,
        env,
        n_steps,
        steps_per_epoch,
        curve_points,
        eval_episodes,
        target_return=target_return,
        download=download,
    )

    # --- Initialize and Train the Model ---
    dt = dt_config.create(device=device, enable_ddp=enable_ddp)
    logs = dt.fit(
        dataset=data,
        n_steps=n_steps,
        n_steps_per_epoch=steps_per_epoch,
        logger_adapter=logger,
        eval_env=environment if eval_while_training else None,
        eval_target_return=target_return if eval_while_training else None,
        save_interval=1000,
        callback=curve,
    )

    out_metric = parse_loss(logs[-1]) if logs is not None else None

    if eval_with_env:
        out_metric = evaluate_with_environment(
            dt,
            environment,
            n_trials=eval_episodes,
            reward=target_return,
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
        dt.save(savep / "model.d3")
        (savep / "hyperparams.json").write_text(
            json.dumps(dt.config.serialize_to_dict(), indent=2, cls=SafeJSONEncoder)
        )

    return out_metric, dt


if __name__ == "__main__":
    out_metric, _ = tyro.cli(dt_train)
    print(out_metric)
