import d3rlpy
import numpy as np
from jumpstart.utils.d3rl_data import get_minari_chunked
from jumpstart.utils.data_stats import DatasetStats
from jumpstart.utils.json_enc import SafeJSONEncoder
from jumpstart.training.utils.factory import (
    get_optimizer_factory,
    get_logger_factory,
)
from jumpstart.training.utils.learning_curve import make_learning_curve
from jumpstart.utils.parse_loss import parse_loss
from jumpstart.utils.d3rl_evaluate import (
    evaluate_with_environment,
    EnvironmentEvaluator,
)
from jumpstart.algorithms.vq_bet import (
    VQBeT,
    VQBeTConfig,
)
import tyro
from pathlib import Path
import json


def vqbet_train(
    env: str,
    learning_rate: float = 5.5e-5,
    offset_learning_rate: float = 1e-3,
    action_chunk_size: int = 16,
    action_horizon: int | None = None,
    eval_action_horizon: int | None = None,
    vqvae_groups: int = 2,
    vqvae_n_embed: int = 16,
    vqvae_steps_ratio: float = 0.3,
    latent_dim: int = 512,
    hidden_dim: int = 256,
    commitment_weight: float = 5.0,
    focal_gamma: float = 2.0,
    offset_loss_weight: float = 1.0,
    primary_code_weight: float = 5.0,
    secondary_code_weight: float = 0.5,
    batch_size: int = 256,
    epochs: int | None = None,
    weight_decay: float = 1e-5,
    clip_grad_norm: float = 1.0,
    warmup_steps: int = 1000,
    wandb: bool = False,
    tensorboard: bool = False,
    project_name: str = "vqbet-minari",
    compile_graph: bool = True,
    env_type="continuous",
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
    save_model_path: str | None = None,
    seed: int | None = None,
) -> tuple[float, VQBeT]:
    if action_horizon is not None:
        action_horizon = min(action_chunk_size, action_horizon)
    if eval_action_horizon is not None:
        eval_action_horizon = min(action_chunk_size, eval_action_horizon)

    if seed is not None:
        d3rlpy.seed(seed)
        np.random.seed(seed)

    if save_model_path is not None:
        savep = Path(save_model_path)
        if (savep / "model.d3").exists() and (savep / "hyperparams.json").exists():
            print(f"Model and hyperparams already exist at {save_model_path}, exiting")
            exit()

    if "d4rl" in env.lower() and eval_while_training:
        raise ValueError("D4RL is not currently compatible with evaluation during training")

    data, val_data, environment, original_action_dim, _ = get_minari_chunked(
        env_name=env,
        action_chunk_size=action_chunk_size,
        download=download,
        val_split=val_split,
        val_percentile=val_percentile,
    )

    stats = DatasetStats.from_dataset(data)
    print(stats)

    if epochs is None:
        epochs = 300

    n_steps = int(np.ceil(stats.num_transitions / batch_size)) * epochs
    steps_per_epoch = int(np.ceil(stats.num_transitions / batch_size))
    vqvae_steps = int(vqvae_steps_ratio * n_steps)
    gpt_steps = n_steps - vqvae_steps
    bin_steps = gpt_steps // 2  # joint phase duration

    # Separate schedulers per phase
    vqvae_optimizer = get_optimizer_factory(
        n_steps=vqvae_steps,
        warmup_steps=min(warmup_steps, vqvae_steps // 2),
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm,
        use_scheduler=True,
    )
    bin_optimizer = get_optimizer_factory(
        n_steps=bin_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm,
        use_scheduler=True,
    )
    offset_optimizer = get_optimizer_factory(
        n_steps=gpt_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm,
        use_scheduler=True,
    )

    config = VQBeTConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        offset_learning_rate=offset_learning_rate,
        vqvae_groups=vqvae_groups,
        vqvae_n_embed=vqvae_n_embed,
        vqvae_steps=vqvae_steps,
        n_steps=n_steps,
        latent_dim=latent_dim,
        action_chunk_size=action_chunk_size,
        action_horizon=action_horizon,
        original_action_dim=original_action_dim,
        hidden_dim=hidden_dim,
        commitment_weight=commitment_weight,
        focal_gamma=focal_gamma,
        offset_loss_weight=offset_loss_weight,
        primary_code_weight=primary_code_weight,
        secondary_code_weight=secondary_code_weight,
        compile_graph=compile_graph,
        optim_factory=bin_optimizer,
        vqvae_optim_factory=vqvae_optimizer,
        offset_optim_factory=offset_optimizer,
    )

    logger = get_logger_factory(project_name, wandb, tensorboard)

    evaluators = None
    if eval_while_training and environment is not None:
        evaluators = {
            "environment": EnvironmentEvaluator(
                environment,
                n_trials=eval_episodes,
                obs_processor=env,
                n_envs=eval_num_envs,
                action_horizon=eval_action_horizon,
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
        action_horizon=eval_action_horizon,
        download=download,
    )

    model = config.create(device=device, enable_ddp=enable_ddp)
    logs = model.fit(
        data,
        n_steps=n_steps,
        evaluators=evaluators,
        n_steps_per_epoch=steps_per_epoch,
        logger_adapter=logger,
        save_interval=1000,
        callback=curve,
    )

    out_metric = parse_loss(logs[-1])

    if eval_with_env and environment is not None:
        out_metric = evaluate_with_environment(
            model,
            environment,
            n_trials=eval_episodes,
            obs_processor=env,
            n_envs=eval_num_envs,
            action_horizon=eval_action_horizon,
            vector_env_dataset_name=env,
            vector_env_download=download,
        )

    if save_model_path is not None:
        savep = Path(save_model_path)
        savep.mkdir(parents=True, exist_ok=True)
        model.save(savep / "model.d3")
        (savep / "hyperparams.json").write_text(
            json.dumps(model.config.serialize_to_dict(), indent=2, cls=SafeJSONEncoder)
        )

    return out_metric, model


if __name__ == "__main__":
    out_metric, _ = tyro.cli(vqbet_train)
    print(f"Final evaluation metric: {out_metric}")
