#!/usr/bin/env python3
"""Download the trial DB, select p95 hyperparameters, and train seeds 0–4."""

import ast
import csv
import gc
import inspect
import json
import math
import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import tyro

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import ALGORITHMS, _load_trainer
from scripts.query_db import fraction, mean_top_fraction


def parse_value(value):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return value


def select_trial(db, algorithm, env, top_fraction=0.25):
    """Return the p95 HP row and the mean of the top fraction of rewards."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    rows = db.execute(
        f'SELECT * FROM "{algorithm}" WHERE environment = ? ORDER BY score, rowid', (env,)
    ).fetchall()
    rows = [row for row in rows if row["score"] is not None and math.isfinite(row["score"])]
    if not rows:
        raise ValueError(f"No finite trial scores for {algorithm} on {env}")
    db_reward = mean_top_fraction((row["score"] for row in rows), top_fraction)
    return dict(rows[round(0.95 * (len(rows) - 1))]), db_reward


def main(
    algorithms: tuple[str, ...] = ("bc", "bcp"),
    envs: list[str] | None = None,
    env_file: Path | None = None,
    device: str = "cuda:0",
    db: Path | None = None,
    top_fraction: float = 0.25,
    output_dir: Path | None = None,
    dry_run: bool = False,
):
    """Download the trial DB, select p95 HPs, and train seeds 0–4.

    Args:
        envs: Space-separated Minari dataset IDs; cannot be combined with env-file.
        env_file: One dataset ID per line, with optional # comments.
        db: Local DB instead of downloading it from the bucket.
        top_fraction: Fraction of DB trials to average for the comparison baseline.
        output_dir: New output directory; defaults to results/p95_*.
        dry_run: Select and print HPs without training.
    """
    top_fraction = fraction(top_fraction)
    if not algorithms or any(algorithm not in ALGORITHMS for algorithm in algorithms):
        raise ValueError(f"Choose algorithms from {sorted(ALGORITHMS)}")
    if envs is not None and env_file is not None:
        raise ValueError("Use either --envs or --env-file")
    if envs is None:
        env_file = env_file or ROOT / "configs/continuous_envs.txt"
        envs = [line.split("#", 1)[0].strip() for line in env_file.read_text().splitlines()]
    envs = list(dict.fromkeys(env for env in envs if env))
    if not envs:
        raise ValueError("Environment list is empty")

    output = output_dir or ROOT / "results" / datetime.now(UTC).strftime("p95_%Y%m%dT%H%M%S%fZ")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    db_path = db
    if db_path is None:
        from huggingface_hub import HfApi

        db_path = output / "all_results.db"
        print("Downloading hf://buckets/omi-n/jumpstart-models/all_results.db", flush=True)
        HfApi().download_bucket_files(
            "omi-n/jumpstart-models", [("all_results.db", db_path)], raise_on_missing_files=True
        )

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    selected = []
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for algorithm in dict.fromkeys(algorithms):
            trainer = _load_trainer(ALGORITHMS[algorithm])
            parameters = inspect.signature(trainer).parameters
            for env in envs:
                trial, db_reward = select_trial(connection, algorithm, env, top_fraction)
                hps = {}
                for key, raw in trial.items():
                    value = parse_value(raw)
                    if key in parameters and value is not None:
                        hps[key] = int(value) if parameters[key].annotation is int else value
                entry = {
                    "algorithm": algorithm,
                    "env": env,
                    "top_fraction": top_fraction,
                    "db_reward": db_reward,
                    "trial_file": trial["file_path"],
                    "hyperparameters": hps,
                }
                selected.append(entry)
                print(json.dumps(entry), flush=True)
    (output / "selected.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(f"{len(selected) * 5} fits; output: {output}", flush=True)
    if dry_run:
        return

    import gymnasium as gym
    import minari
    import torch

    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["algorithm", "environment", "seed", "db_reward", "reward", "top_fraction"])
        stream.flush()
        for entry in selected:
            algorithm, env = entry["algorithm"], entry["env"]
            dataset = minari.load_dataset(env, download=True)
            if not isinstance(dataset.spec.action_space, gym.spaces.Box):
                raise TypeError(f"Expected a continuous action space: {env}")
            environment = dataset.recover_environment()
            try:
                environment.reset(seed=0)
                environment.action_space.seed(0)
                environment.step(environment.action_space.sample())
            finally:
                environment.close()
            del dataset, environment
            trainer = _load_trainer(ALGORITHMS[algorithm])
            rewards = []
            for seed in range(5):
                folder = output / algorithm / env.replace("/", "_") / f"seed{seed}"
                folder.mkdir(parents=True)
                params = entry["hyperparameters"] | {
                    "env": env,
                    "env_type": "continuous",
                    "seed": seed,
                    "device": device,
                    "project_name": str(folder / "training"),
                    "download": True,
                    "wandb": False,
                    "tensorboard": False,
                    "enable_ddp": False,
                    "eval_with_env": True,
                    "eval_episodes": 100,
                    "eval_num_envs": 10,
                    "eval_while_training": False,
                }
                print(f"Training {algorithm} on {env}, seed {seed}", flush=True)
                reward, model = trainer(**params)
                if model is None or not math.isfinite(float(reward)):
                    raise RuntimeError(f"Invalid result for {algorithm} on {env}, seed {seed}")
                model.save(str(folder / "model.d3"))
                writer.writerow(
                    [algorithm, env, seed, entry["db_reward"], float(reward), top_fraction]
                )
                stream.flush()
                rewards.append(float(reward))
                del model
                gc.collect()
                torch.cuda.empty_cache()
            mean_reward = math.fsum(rewards) / len(rewards)
            writer.writerow([algorithm, env, "mean", entry["db_reward"], mean_reward, top_fraction])
            stream.flush()
            print(f"Mean reward for {algorithm} on {env}: {mean_reward}", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
