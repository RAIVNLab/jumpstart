#!/usr/bin/env python3
"""Download a .d3 model from a Hugging Face bucket and evaluate it locally."""

import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import tyro

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import ALGORITHMS
from scripts.query_db import fraction, query_reward


def main(
    hf_path: Annotated[str, tyro.conf.Positional],
    dataset: str,
    algorithm: str | None = None,
    db: Path | None = None,
    top_fraction: float = 0.25,
    device: str = "cuda:0",
    episodes: int = 100,
    num_envs: int = 10,
    target_return: float | None = None,
    action_horizon: int | None = None,
    output_dir: Path | None = None,
):
    """Download a model from a Hugging Face bucket and evaluate it locally.

    Args:
        hf_path: hf://buckets/OWNER/BUCKET/path/to/model.d3.
        dataset: Minari dataset ID.
        algorithm: DB algorithm; inferred from standard bucket paths when omitted.
        db: Local DB; otherwise downloaded from omi-n/jumpstart-models.
        top_fraction: Fraction of DB trials to average for the comparison baseline.
        target_return: DT target; defaults to the maximum dataset return.
        action_horizon: Override the saved action horizon.
        output_dir: New output directory; defaults to results/hf_eval_*.
    """
    top_fraction = fraction(top_fraction)
    if episodes < 1 or num_envs < 1:
        raise ValueError("--episodes and --num-envs must be positive")
    if action_horizon is not None and action_horizon < 1:
        raise ValueError("--action-horizon must be positive")
    parts = hf_path.removeprefix("hf://").removeprefix("buckets/").split("/", 2)
    if not hf_path.startswith("hf://") or len(parts) != 3 or not all(parts):
        raise ValueError("Expected hf://buckets/OWNER/BUCKET/path/to/model.d3")
    owner, bucket, remote_path = parts
    remote_parts = remote_path.split("/")
    if algorithm is None and len(remote_parts) > 1:
        if remote_parts[0] in {"optuna", "random"}:
            algorithm = remote_parts[1]
        elif remote_parts[0] == "seed_variance_models":
            algorithm = remote_parts[1].split("__", 1)[0]
    if algorithm not in ALGORITHMS:
        raise ValueError("--algorithm is required for a nonstandard model path")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    import d3rlpy
    import minari
    import numpy as np
    import torch
    from huggingface_hub import HfApi

    import jumpstart.algorithms
    import jumpstart.utils.d3rl_scheduler  # noqa: F401 -- register saved schedulers
    from jumpstart.utils.d3rl_data import get_minari
    from jumpstart.utils.d3rl_evaluate import evaluate_with_environment

    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
    output = output_dir or ROOT / "results" / datetime.now(UTC).strftime("hf_eval_%Y%m%dT%H%M%S%fZ")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    db_path = db
    if db_path is None:
        db_path = output / "all_results.db"
        print("Downloading hf://buckets/omi-n/jumpstart-models/all_results.db", flush=True)
        HfApi().download_bucket_files(
            "omi-n/jumpstart-models", [("all_results.db", db_path)], raise_on_missing_files=True
        )
    db_reward = query_reward(db_path, algorithm, dataset, top_fraction)
    local_path = output / "model.d3"
    print(f"Downloading {hf_path}", flush=True)
    HfApi().download_bucket_files(
        f"{owner}/{bucket}", [(remote_path, local_path)], raise_on_missing_files=True
    )
    model = d3rlpy.load_learnable(str(local_path), device=device)
    effective_action_horizon = action_horizon or getattr(model.config, "action_horizon", None)
    chunk_size = getattr(model.config, "action_chunk_size", None)
    if chunk_size is not None:
        effective_action_horizon = min(effective_action_horizon or chunk_size, chunk_size)
    is_transformer = isinstance(model, d3rlpy.algos.decision_transformer.TransformerAlgoBase)
    num_envs = 1 if is_transformer else num_envs
    if is_transformer and target_return is None:
        data = minari.load_dataset(dataset, download=True)
        target_return = max(float(np.sum(ep.rewards)) for ep in data.iterate_episodes())

    env = get_minari(dataset, download=True, env_only=True)
    if env is None:
        raise RuntimeError(f"Could not recover environment for {dataset}")
    print(
        f"Evaluating {type(model).__name__} on {dataset}: "
        f"{episodes} episodes, {num_envs} environments",
        flush=True,
    )
    reward = evaluate_with_environment(
        model,
        env,
        obs_processor=dataset,
        reward=target_return,
        n_trials=episodes,
        n_envs=num_envs,
        action_horizon=action_horizon,
        vector_env_dataset_name=dataset,
        vector_env_download=True,
    )
    if not math.isfinite(reward):
        raise RuntimeError(f"Non-finite evaluation reward: {reward}")
    result = {
        "hf_path": hf_path,
        "dataset": dataset,
        "algorithm": algorithm,
        "top_fraction": top_fraction,
        "db_reward": db_reward,
        "model_type": type(model).__name__,
        "device": device,
        "episodes": episodes,
        "num_envs": num_envs,
        "target_return": target_return,
        "action_horizon": effective_action_horizon,
        "mean_reward": reward,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved model and evaluation to {output}", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
