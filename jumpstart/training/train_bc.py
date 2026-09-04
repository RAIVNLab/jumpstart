import d3rlpy
import numpy as np
from jumpstart.utils.d3rl_data import get_minari
from jumpstart.utils.data_stats import DatasetStats
from jumpstart.utils.json_enc import SafeJSONEncoder
from jumpstart.training.utils.factory import (
    get_encoder_factory,
    get_optimizer_factory,
    get_logger_factory,
    get_reward_scaler,
)
from jumpstart.training.utils.learning_curve import make_learning_curve
from jumpstart.utils.parse_loss import parse_loss
from jumpstart.utils.d3rl_evaluate import (
    evaluate_with_environment,
    EnvironmentEvaluator,
)
from typing import List, Literal
import tyro
from functools import partial
from pathlib import Path
import json


# TODO: support custom filter sizes in Atari
# TODO: find Atari filter sizes that get full reward in most environments
#       (scaling experiment)
def bc_train(
    env: str,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    epochs: int | None = None,
    weight_decay: float = 0.01,
    clip_grad_norm: float = 1.0,
    hidden_units: List[int] = [256, 256],
    activation: str = "relu",
    dropout: float = 0.0,
    train_percentile: float = 0.0,
    env_type: Literal["continuous", "discrete"] = "continuous",
    reg_factor: float = 0.0,
    wandb: bool = False,
    tensorboard: bool = False,
    project_name: str = "bc-minari",
    policy_type: str = "deterministic",
    compile_graph: bool = True,
    device: str = "cuda",
    enable_ddp: bool = False,
    eval_with_env: bool = True,
    eval_episodes: int = 100,
    eval_num_envs: int = 10,
    warmup_steps: int = 1000,
    val_split: float = 0.0,
    val_percentile: int = 50,
    download: bool = True,
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
        train_percentile=train_percentile,
    )

    stats = DatasetStats.from_dataset(data)
    print(stats)

    if epochs is None:
        if "atari" not in env:
            epochs = 50
        else:
            # try to normalize based on the transition count since it varies widely
            epochs = int(250 / ((stats.num_transitions / batch_size) / 60))

    # auto-adjust if Atari
    if "atari" in env.lower():
        env_type = "discrete"

    encoder_config, environment = get_encoder_factory(
        stats.obs_dim,
        environment,
        hidden_units,
        activation,
        dropout,
    )

    # need to compute this for logging reasons
    n_steps = int(np.ceil(stats.num_transitions / batch_size)) * epochs
    steps_per_epoch = int(np.ceil(stats.num_transitions / batch_size))

    optimizer = get_optimizer_factory(
        n_steps=n_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm,
        use_scheduler=True,
    )

    # support both bc types
    bc_config_cls = (
        d3rlpy.algos.DiscreteBCConfig if env_type == "discrete" else d3rlpy.algos.BCConfig
    )

    if env_type == "discrete":
        # beta is a hilariously vague term. renaming it to reg_factor
        bc_config_cls = partial(bc_config_cls, beta=reg_factor)

    if policy_type == "stochastic" and env_type == "continuous":
        bc_config_cls = partial(bc_config_cls, policy_type=policy_type)

    bc_config = bc_config_cls(
        batch_size=batch_size,
        learning_rate=learning_rate,
        encoder_factory=encoder_config,
        optim_factory=optimizer,
        compile_graph=compile_graph,
    )

    logger = get_logger_factory(
        project_name,
        wandb,
        tensorboard,
    )

    evaluators = None
    if eval_while_training:
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

    bc = bc_config.create(device=device, enable_ddp=enable_ddp)
    logs = bc.fit(
        data,
        n_steps=n_steps,
        evaluators=evaluators,
        n_steps_per_epoch=steps_per_epoch,
        logger_adapter=logger,
        save_interval=1000,
        callback=curve,
    )

    out_metric = parse_loss(logs[-1])

    if eval_with_env:
        out_metric = evaluate_with_environment(
            bc,
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
        bc.save(savep / "model.d3")
        (savep / "hyperparams.json").write_text(
            json.dumps(bc.config.serialize_to_dict(), indent=2, cls=SafeJSONEncoder)
        )

    return out_metric, bc


if __name__ == "__main__":
    out_metric, _ = tyro.cli(bc_train)
    print(f"Final evaluation metric: {out_metric}")
