"""Command-line entry point for training on Minari datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Sequence


ActionType = str


@dataclass(frozen=True)
class AlgorithmSpec:
    """Location and action-space support for a trainer."""

    module: str
    function: str
    action_types: frozenset[ActionType]


_BOTH_ACTION_TYPES = frozenset({"continuous", "discrete"})
_CONTINUOUS_ONLY = frozenset({"continuous"})

ALGORITHMS: dict[str, AlgorithmSpec] = {
    "act": AlgorithmSpec("jumpstart.training.train_act", "act_train", _CONTINUOUS_ONLY),
    "bc": AlgorithmSpec("jumpstart.training.train_bc", "bc_train", _BOTH_ACTION_TYPES),
    "bcp": AlgorithmSpec("jumpstart.training.train_bc", "bc_train", _BOTH_ACTION_TYPES),
    "bcq": AlgorithmSpec("jumpstart.training.train_bcq", "bcq_train", _BOTH_ACTION_TYPES),
    "cql": AlgorithmSpec("jumpstart.training.train_cql", "cql_train", _BOTH_ACTION_TYPES),
    "diffusion": AlgorithmSpec(
        "jumpstart.training.train_diffusion", "diffusion_train", _CONTINUOUS_ONLY
    ),
    "dt": AlgorithmSpec("jumpstart.training.train_dt", "dt_train", _BOTH_ACTION_TYPES),
    "iql": AlgorithmSpec("jumpstart.training.train_iql", "iql_train", _CONTINUOUS_ONLY),
    "rebrac": AlgorithmSpec("jumpstart.training.train_rebrac", "rebrac_train", _CONTINUOUS_ONLY),
    "vqbet": AlgorithmSpec("jumpstart.training.train_vqbet", "vqbet_train", _CONTINUOUS_ONLY),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jumpstart",
        description=(
            "Train an algorithm on any compatible Minari dataset. Unrecognized options are "
            "forwarded to the selected trainer."
        ),
        epilog=(
            "Example: jumpstart --algorithm bc --dataset D4RL/pen/human-v2 --device cpu --epochs 5"
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--algorithm",
        choices=sorted(ALGORITHMS),
        help="Training algorithm to run.",
    )
    parser.add_argument("--dataset", help="Minari dataset ID.")
    parser.add_argument(
        "--list-algorithms",
        action="store_true",
        help="List algorithms and their supported action spaces, then exit.",
    )
    parser.add_argument(
        "--trainer-help",
        action="store_true",
        help="Show the selected trainer's algorithm-specific options, then exit.",
    )
    return parser


def _list_algorithms() -> None:
    for name, spec in ALGORITHMS.items():
        print(f"{name:<10} {', '.join(sorted(spec.action_types))}")


def _load_trainer(spec: AlgorithmSpec) -> Callable[..., object]:
    try:
        module = import_module(spec.module)
        trainer = getattr(module, spec.function)
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            f"Could not load trainer {spec.function!r} from {spec.module!r}: {error}"
        ) from error
    return trainer


def _download_enabled(trainer_args: Sequence[str]) -> bool:
    """Mirror the trainers' default download flag for the compatibility check."""
    download = True
    for index, argument in enumerate(trainer_args):
        if argument == "--no-download":
            download = False
        elif argument == "--download":
            download = True
            if index + 1 < len(trainer_args):
                value = trainer_args[index + 1].lower()
                if value in {"false", "0", "no"}:
                    download = False
        elif argument.startswith("--download="):
            value = argument.partition("=")[2].lower()
            download = value not in {"false", "0", "no"}
    return download


def _infer_action_type(dataset_id: str, *, download: bool) -> ActionType:
    """Load a Minari spec and classify its action space without recovering the env."""
    try:
        import gymnasium as gym
        import minari
    except ImportError as error:
        raise RuntimeError(
            "Dataset inspection requires the project dependencies; run `uv sync` first."
        ) from error

    try:
        dataset = minari.load_dataset(dataset_id, download=download)
    except Exception as error:
        raise RuntimeError(f"Could not load Minari dataset {dataset_id!r}: {error}") from error

    action_space = dataset.spec.action_space
    if isinstance(action_space, gym.spaces.Box):
        return "continuous"
    if isinstance(action_space, gym.spaces.Discrete):
        return "discrete"
    raise ValueError(
        f"Dataset {dataset_id!r} uses unsupported action space "
        f"{type(action_space).__name__}; expected Box or Discrete."
    )


def _validate_compatibility(
    algorithm: str,
    dataset_id: str,
    action_type: ActionType,
) -> AlgorithmSpec:
    spec = ALGORITHMS[algorithm]
    if action_type not in spec.action_types:
        supported = ", ".join(sorted(spec.action_types))
        raise ValueError(
            f"Algorithm {algorithm!r} does not support the {action_type} action space used by "
            f"{dataset_id!r}; it supports: {supported}."
        )
    return spec


def _check_forwarded_args(parser: argparse.ArgumentParser, trainer_args: Sequence[str]) -> None:
    protected = {"--env", "--env-type"}
    supplied = {argument.partition("=")[0] for argument in trainer_args}
    conflicts = sorted(protected & supplied)
    if conflicts:
        parser.error(f"{', '.join(conflicts)} is managed by the dispatcher; use --dataset instead.")


def main(argv: Sequence[str] | None = None) -> object | None:
    parser = _parser()
    args, trainer_args = parser.parse_known_args(argv)

    if args.list_algorithms:
        _list_algorithms()
        return None
    if args.algorithm is None:
        parser.error("--algorithm is required")

    spec = ALGORITHMS[args.algorithm]
    if args.trainer_help:
        try:
            trainer = _load_trainer(spec)
            import tyro
        except (ImportError, RuntimeError) as error:
            parser.error(str(error))
        return tyro.cli(trainer, args=["--help"])

    if args.dataset is None:
        parser.error("--dataset is required")
    _check_forwarded_args(parser, trainer_args)

    try:
        action_type = _infer_action_type(
            args.dataset,
            download=_download_enabled(trainer_args),
        )
        spec = _validate_compatibility(args.algorithm, args.dataset, action_type)
        trainer = _load_trainer(spec)
        import tyro
    except (ImportError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    result = tyro.cli(
        trainer,
        args=[
            "--env",
            args.dataset,
            "--env-type",
            action_type,
            *trainer_args,
        ],
    )
    if isinstance(result, tuple) and result:
        print(f"Final evaluation metric: {result[0]}")
    return result


if __name__ == "__main__":
    main()
