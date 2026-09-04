import ray
from ray import tune
from ray import train
from ray.air import RunConfig as TuneRunConfig
from ray.tune.search import BasicVariantGenerator
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bayesopt import BayesOptSearch

import tempfile
import uuid
import torch
import tyro
from typing import Optional, Literal, Dict, Any
import os
import glob
import shutil
import threading
import time
from datetime import datetime

from jumpstart.hyperparam import tuner_ranges
from jumpstart.utils.infer_environment import infer_env_type


def convert_to_ray_space(range_obj: tuner_ranges.BaseRange):
    """Converts a tuner_ranges Range object to a Ray Tune search space."""
    if isinstance(range_obj, tuner_ranges.UniformRange):
        return tune.uniform(range_obj.min_value, range_obj.max_value)
    elif isinstance(range_obj, tuner_ranges.LogUniformRange):
        return tune.loguniform(range_obj.min_value, range_obj.max_value)
    elif isinstance(range_obj, tuner_ranges.IntegerRange):
        # random.randint is inclusive, ray.tune.randint is exclusive
        return tune.randint(int(range_obj.min_value), int(range_obj.max_value) + 1)
    elif isinstance(range_obj, tuner_ranges.LogIntegerRange):
        return tune.lograndint(int(range_obj.min_value), int(range_obj.max_value) + 1)
    elif isinstance(range_obj, tuner_ranges.QuantizedIntegerRange):
        return tune.qrandint(int(range_obj.min_value), int(range_obj.max_value) + 1, range_obj.q)
    elif isinstance(range_obj, tuner_ranges.LogQuantizedIntegerRange):
        return tune.qlograndint(int(range_obj.min_value), int(range_obj.max_value) + 1, range_obj.q)
    elif isinstance(range_obj, tuner_ranges.GaussianRange):
        return tune.randn(range_obj.mean, range_obj.stddev)
    elif isinstance(range_obj, tuner_ranges.ChoiceRange):
        return tune.choice(range_obj.choices)
    elif isinstance(range_obj, tuner_ranges.ConstantRange):
        return range_obj.value
    else:
        raise ValueError(f"Unknown range type: {type(range_obj)}")


def get_ray_search_space(algorithm: str, env_type: str) -> Dict[str, Any]:
    """Gets the Ray Tune search space for a given algorithm."""
    ranges = tuner_ranges.get_search_space(algorithm, env_type)
    return {k: convert_to_ray_space(v) for k, v in ranges.items()}


class TunerStateWatcher:
    """Watches tuner.pkl and backs it up whenever it changes."""

    def __init__(self, experiment_path: str, max_backups: int = 100):
        self.experiment_path = experiment_path
        self.max_backups = max_backups
        self.backup_dir = os.path.join(experiment_path, "backups")
        self.tuner_pkl = os.path.join(experiment_path, "tuner.pkl")
        self.last_mtime = None
        self._stop_event = threading.Event()
        self._thread = None

    def _get_mtime(self) -> float | None:
        try:
            return os.path.getmtime(self.tuner_pkl)
        except OSError:
            return None

    def _backup(self) -> None:
        """Create a backup of tuner.pkl."""
        if not os.path.exists(self.tuner_pkl):
            return

        os.makedirs(self.backup_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        backup_path = os.path.join(self.backup_dir, f"tuner_{timestamp}.pkl")

        try:
            shutil.copy2(self.tuner_pkl, backup_path)
            print(f"Backed up tuner.pkl -> {os.path.basename(backup_path)}")
        except Exception as e:
            print(f"Failed to backup tuner.pkl: {e}")
            return

        # Cleanup old backups
        pkl_backups = sorted(glob.glob(os.path.join(self.backup_dir, "tuner_*.pkl")))
        if len(pkl_backups) > self.max_backups:
            for old_backup in pkl_backups[: -self.max_backups]:
                try:
                    os.remove(old_backup)
                except OSError:
                    pass

    def _watch_loop(self) -> None:
        """Background thread that watches for tuner.pkl changes."""
        while not self._stop_event.is_set():
            mtime = self._get_mtime()
            if mtime is not None and mtime != self.last_mtime:
                if self.last_mtime is not None:  # Don't backup on first detection
                    self._backup()
                self.last_mtime = mtime
            time.sleep(1)  # Check every second

    def start(self) -> None:
        """Start watching for changes."""
        # Initial backup if file exists
        if os.path.exists(self.tuner_pkl):
            self._backup()
            self.last_mtime = self._get_mtime()

        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop watching and create final backup."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        # Final backup
        if os.path.exists(self.tuner_pkl):
            self._backup()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# TODO: add saving models to a specified path after each trial (save_model_path in the functions)
def run_ray_tuning(
    algorithm: Literal[
        "act", "bc", "bcp", "bcq", "cql", "diffusion", "dt", "iql", "rebrac", "vqbet"
    ],
    env: str,
    env_type: Optional[Literal["continuous", "discrete"]] = None,
    search_alg: Literal["random", "optuna", "hyperopt", "bayesopt"] = "random",
    num_samples: int = 100,
    max_concurrent_trials: Optional[int] = 0,
    cpus_per_trial: Optional[float] = 0,
    gpus_per_trial: float = 1,
    num_jobs_per_gpu: Optional[int] = None,
    storage_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
    metric: str = "score",
    mode: str = "max",
    ray_address: Optional[str] = None,
    resume: bool = True,
    checkpoint_frequency: int = 1,
    excludes: Optional[str] = None,
    download: bool = True,
):
    """
    Runs hyperparameter tuning using Ray Tune.

    Args:
        algorithm: The algorithm to tune (e.g., "bc", "cql", or "iql").
        env: The environment name.
        env_type: The type of environment ("continuous" or "discrete"). If None, inferred from env.
        search_alg: The search algorithm to use.
        num_samples: Number of trials to run.
        max_concurrent_trials: Maximum number of trials to run concurrently.
        cpus_per_trial: CPUs to allocate per trial.
        gpus_per_trial: GPUs to allocate per trial.
        num_jobs_per_gpu: Number of jobs to run per GPU. Overrides gpus_per_trial if set.
        storage_path: Path to store Ray results.
        experiment_name: Name of the experiment.
        metric: The metric to optimize.
        mode: Optimization mode ("min" or "max").
        ray_address: Address of the Ray cluster to connect to. If None, starts a local Ray instance.
                     Use "auto" to connect to an existing cluster.
        resume: Whether to resume from a previous run if it exists. Default is True.
        checkpoint_frequency: How often to checkpoint trials (every N iterations). Default is 1.
        download: Whether to download the Minari dataset when it is not cached.
    """

    # Initialize Ray
    if not ray.is_initialized():
        if ray_address:
            ray.init(
                address=ray_address,
                runtime_env={
                    "excludes": [
                        "bc-minari/",
                        "data_generation/",
                        "ray_results_bc/",
                        "configs/",
                        "rewards/",
                        ".venv/",
                        "__pycache__/",
                        "*.pyc",
                        ".git/",
                    ]
                },
            )
        else:
            ray.init(
                runtime_env={
                    "excludes": [
                        "bc-minari/",
                        "data_generation/",
                        "ray_results_bc/",
                        "configs/",
                        "rewards/",
                        ".venv/",
                        "__pycache__/",
                        "*.pyc",
                        ".git/",
                    ]
                },
            )

    if env_type is None:
        env_type = infer_env_type(env, download=download)
        print(f"Inferred env_type: {env_type}")

    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count() or 1
    total_jobs = max(1, num_gpus * (num_jobs_per_gpu or 1))
    cpus_per_trial = int(max(1, num_cpus // total_jobs))

    if cpus_per_trial is None or cpus_per_trial == 0:
        cpus_per_trial = int(max(1, num_cpus // total_jobs))

    if num_jobs_per_gpu is not None:
        gpus_per_trial = 1.0 / num_jobs_per_gpu
        if num_gpus > 0:
            print(
                f"Auto-configured cpus_per_trial: {cpus_per_trial} "
                f"(Node CPUs: {num_cpus}, Node GPUs: {num_gpus}, "
                f"Jobs/GPU: {num_jobs_per_gpu})"
            )

    # Get search space and objective function
    # We use get_search_space directly to convert, and NAME2ALGO for the function
    ray_search_space = get_ray_search_space(algorithm, env_type)

    # Add env and env_type to the config (fixed parameters)
    # Note: If they are already in the search space (as ConstantRange), they will be overwritten
    # or we should ensure we don't overwrite if they are meant to be tuned (unlikely for env).
    # But usually env is passed as an argument.

    # The objective function from tuner_ranges
    _, objective_fn_raw = tuner_ranges.get_search_space_and_objective_fn(algorithm, env_type)

    def trainable(config):
        # Inject env and env_type if not present
        run_config = config.copy()
        if "env" not in run_config:
            run_config["env"] = env
        if "env_type" not in run_config:
            run_config["env_type"] = env_type
        if "download" not in run_config:
            run_config["download"] = download

        # Call the training function
        # The training functions return (metric, model)
        try:
            result_metric, model = objective_fn_raw(**run_config)

            if model is None:
                tune.report(
                    {
                        metric: result_metric,
                        "hyperparameters": run_config,
                    }
                )
                return

            # Create checkpoint with model using temporary directory
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_{uuid.uuid4().hex}.d3")
                model.save(checkpoint_path)

                checkpoint = tune.Checkpoint.from_directory(checkpoint_dir)

                # Report the metric and hyperparameters to Ray with checkpoint
                tune.report(
                    {
                        metric: result_metric,
                        "hyperparameters": run_config,
                    },
                    checkpoint=checkpoint,
                )

        except Exception as e:
            print(f"Trial failed with error: {e}")
            raise e

    # Configure Search Algorithm
    searcher = None
    scheduler = None

    if search_alg == "random":
        searcher = BasicVariantGenerator(max_concurrent=max_concurrent_trials)
    elif search_alg == "optuna":
        searcher = OptunaSearch(metric=metric, mode=mode)
    elif search_alg == "hyperopt":
        searcher = HyperOptSearch(metric=metric, mode=mode)
    elif search_alg == "bayesopt":
        searcher = BayesOptSearch(metric=metric, mode=mode)
    else:
        raise ValueError(f"Unknown search algorithm: {search_alg}")

    spath = storage_path or os.path.join(os.getcwd(), f"ray_results_{algorithm}")
    spath = os.path.abspath(os.path.join(spath, search_alg, algorithm))

    # Check if we should resume from a previous run
    experiment_path = os.path.join(spath, experiment_name or f"{env}")
    should_restore = resume and os.path.exists(experiment_path)
    print(
        f"Experiment path: {experiment_path}, should_restore: {should_restore}, "
        f"exists: {os.path.exists(experiment_path)}"
    )

    # Create watcher to backup tuner.pkl on every update
    state_watcher = TunerStateWatcher(experiment_path)

    if should_restore:
        print(f"Resuming experiment from: {experiment_path}")
        tuner = tune.Tuner.restore(
            path=experiment_path,
            trainable=tune.with_resources(
                trainable, resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
            ),
            resume_unfinished=True,
            resume_errored=True,
        )
    else:
        print(f"Starting new experiment at: {experiment_path}")
        # Setup Tuner
        tuner = tune.Tuner(
            tune.with_resources(
                trainable, resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
            ),
            param_space=ray_search_space,
            tune_config=tune.TuneConfig(
                metric=metric,
                mode=mode,
                search_alg=searcher,
                scheduler=scheduler,
                num_samples=num_samples,
                max_concurrent_trials=max_concurrent_trials,
            ),
            run_config=TuneRunConfig(
                name=experiment_name or env,
                storage_path=spath,
            ),
        )

    with state_watcher:
        results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)
    return results


if __name__ == "__main__":
    tyro.cli(run_ray_tuning)
