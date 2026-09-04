import csv
import time
from pathlib import Path

import minari

from jumpstart.utils.d3rl_evaluate import evaluate_with_environment


class LearningCurve:
    """Evaluate reward at evenly spaced fractions of one training run."""

    def __init__(
        self,
        output: Path,
        environment: str,
        n_steps: int,
        n_steps_per_epoch: int,
        points: int,
        episodes: int,
        target_return: float | None = None,
        action_horizon: int | None = None,
        download: bool = True,
    ):
        if points < 1:
            raise ValueError("curve_points must be positive")

        self.output = output
        self.environment = environment
        self.episodes = episodes
        self.target_return = target_return
        self.action_horizon = action_horizon
        self.started_at = time.monotonic()
        self.dataset = minari.load_dataset(environment, download=download)

        self.total_steps = n_steps // n_steps_per_epoch * n_steps_per_epoch
        self.checkpoints = {
            max(1, round(self.total_steps * point / points)) for point in range(1, points + 1)
        }

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as file:
            csv.writer(file).writerow(
                ("environment", "epoch", "step", "training_fraction", "reward", "elapsed_seconds")
            )

    def __call__(self, algo, epoch: int, step: int) -> None:
        if step not in self.checkpoints:
            return

        print(f"[curve] evaluating step {step}/{self.total_steps}", flush=True)
        environment = self.dataset.recover_environment()
        if environment is None:
            raise RuntimeError(f"could not recover environment for {self.environment}")

        reward = evaluate_with_environment(
            algo,
            environment,
            reward=self.target_return,
            n_trials=self.episodes,
            obs_processor=self.environment,
            action_horizon=self.action_horizon,
        )
        with self.output.open("a", newline="") as file:
            csv.writer(file).writerow(
                (
                    self.environment,
                    epoch,
                    step,
                    step / self.total_steps,
                    reward,
                    time.monotonic() - self.started_at,
                )
            )
            file.flush()
        print(f"[curve] step {step}/{self.total_steps}: reward={reward}", flush=True)


def make_learning_curve(
    output: Path | None,
    environment: str,
    n_steps: int,
    n_steps_per_epoch: int,
    points: int,
    episodes: int,
    target_return: float | None = None,
    action_horizon: int | None = None,
    download: bool = True,
) -> LearningCurve | None:
    if output is None:
        return None
    return LearningCurve(
        output=output,
        environment=environment,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        points=points,
        episodes=episodes,
        target_return=target_return,
        action_horizon=action_horizon,
        download=download,
    )
