"""Random search hyperparameter configuration generator using tuner_ranges.

This module generates hyperparameter configurations for various RL algorithms
using the range specifications defined in tuner_ranges.py. It can output
configurations in either plain text format (.txt) or YAML format compatible
with job-batcher tools.
"""

import os
import random
import yaml
import tyro
from dataclasses import dataclass
from typing import Literal, List, Dict, Any

from jumpstart.utils.infer_environment import infer_env_type
from jumpstart.hyperparam.tuner_ranges import get_search_space


@dataclass
class RandomYamlConfig:
    """Configuration for generating random hyperparameter configurations."""

    # Study configuration
    experiment_name: str = "rl_hyperopt_random"
    """Name of the random search experiment"""
    n_trials: int = 100
    """Number of random configurations to generate"""
    algorithm: Literal[
        "act", "bc", "bcp", "bcq", "cql", "diffusion", "dt", "iql", "rebrac", "vqbet"
    ] = "bc"
    """Which algorithm to generate configs for"""

    # Environment configuration
    env_id: str = "mujoco/halfcheetah/medium-v0"
    """Minari dataset to train on"""

    # Randomization
    seed: int = 42
    """Random seed for reproducibility"""

    # Storage and logging
    results_dir: str = "./random_search_configs"
    """Directory to store generated configurations"""
    verbose: bool = True
    """Whether to print generation progress"""

    # Command generation
    base_command: str = ""
    """Base command for generated jobs. Defaults to ``uv run jumpstart``."""

    # Output format
    output_format: Literal["yaml", "txt"] = "yaml"
    """Output format: 'yaml' for job-batcher YAML or 'txt' for plain text commands"""

    # Job batcher settings
    workers_per_gpu: int = 1
    """Number of workers per GPU for job batcher"""
    job_prefix: str = ""
    """Prefix for job names. If empty, auto-generated from algorithm and env_id"""


def sample_hyperparams_from_ranges(
    search_space: Dict[str, Any], seed: int = None
) -> Dict[str, Any]:
    """Sample hyperparameters from the defined search space ranges.

    Args:
        search_space: Dictionary mapping hyperparameter names to Range objects
        seed: Optional random seed for reproducibility

    Returns:
        Dictionary of sampled hyperparameter values
    """
    if seed is not None:
        random.seed(seed)

    hyperparams = {}
    for param_name, param_range in search_space.items():
        hyperparams[param_name] = param_range.sample()

    return hyperparams


def format_hyperparam_value(value: Any) -> str:
    """Format a hyperparameter value for command-line usage.

    Args:
        value: The hyperparameter value to format

    Returns:
        Formatted string representation
    """
    if isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, (int, float)):
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)
    elif isinstance(value, list):
        # Handle list of lists (e.g., hidden_units)
        if value and isinstance(value[0], list):
            # Flatten and format as space-separated values
            flat = " ".join(str(x) for sublist in value for x in sublist)
            return flat
        # Handle regular lists
        return " ".join(str(x) for x in value)
    else:
        return str(value)


def build_command_from_hyperparams(
    base_cmd: str,
    hyperparams: Dict[str, Any],
    algorithm: str,
    env_id: str,
) -> str:
    """Build a command-line string from hyperparameters.

    Args:
        base_cmd: Base command (e.g., "uv run jumpstart")
        hyperparams: Dictionary of hyperparameter values
        algorithm: Algorithm name
        env_id: Minari dataset ID

    Returns:
        Complete command-line string
    """
    cmd_parts = [base_cmd, f"--algorithm {algorithm}", f"--dataset {env_id}"]

    # Add hyperparameters as command line arguments
    for param_name, param_value in hyperparams.items():
        # Convert underscores to dashes for CLI compatibility
        cli_param_name = param_name.replace("_", "-")
        formatted_value = format_hyperparam_value(param_value)

        # Handle boolean flags
        if isinstance(param_value, bool):
            if param_value:
                cmd_parts.append(f"--{cli_param_name}")
            else:
                cmd_parts.append(f"--no-{cli_param_name}")
        else:
            cmd_parts.append(f"--{cli_param_name} {formatted_value}")

    return " ".join(cmd_parts)


def generate_configs(config: RandomYamlConfig) -> List[str]:
    """Generate n_trials different hyperparameter configurations as commands.

    Args:
        config: RandomYamlConfig object with generation parameters

    Returns:
        List of command strings
    """
    # Infer environment type from the environment ID
    env_type = infer_env_type(config.env_id)

    # Get the search space for this algorithm and environment type
    search_space = get_search_space(config.algorithm, env_type)

    # Set random seed for reproducibility
    random.seed(config.seed)

    commands = []

    # Determine the base command based on algorithm if not provided
    base_cmd = config.base_command
    if not base_cmd:
        base_cmd = "uv run jumpstart"

    if config.verbose:
        print(f"Generating {config.n_trials} configurations for {config.algorithm}")
        print(f"Environment: {config.env_id} ({env_type})")
        print(f"Base command: {base_cmd}")
        print(f"Search space parameters: {list(search_space.keys())}")
        print("-" * 50)

    for trial_idx in range(config.n_trials):
        # Sample hyperparameters for this trial
        hyperparams = sample_hyperparams_from_ranges(search_space)

        # Build the command
        command = build_command_from_hyperparams(
            base_cmd=base_cmd,
            hyperparams=hyperparams,
            algorithm=config.algorithm,
            env_id=config.env_id,
        )

        commands.append(command)

        if config.verbose and (trial_idx + 1) % 10 == 0:
            print(f"Generated {trial_idx + 1}/{config.n_trials} configurations")

    if config.verbose:
        print(f"✓ Generated {len(commands)} configurations")

    return commands


def save_configs(config: RandomYamlConfig, commands: List[str]) -> None:
    """Save generated configurations in the specified output format.

    Args:
        config: RandomYamlConfig object with generation parameters
        commands: List of generated command strings
    """
    # Create output directory
    os.makedirs(config.results_dir, exist_ok=True)

    # Generate filename base
    env_safe = config.env_id.replace("-", "_").replace("/", "_")
    file_base = f"config_commands_{config.algorithm}_{env_safe}"

    if config.output_format == "txt":
        # Save as plain text file (one command per line)
        txt_file = os.path.join(config.results_dir, f"{file_base}.txt")
        with open(txt_file, "w") as f:
            for cmd in commands:
                f.write(cmd + "\n")

        if config.verbose:
            print(f"✓ Saved text commands to: {txt_file}")
            print(f"\nGenerated {len(commands)} command lines")

    else:  # yaml format
        # Save as YAML file compatible with job-batcher
        yaml_file = os.path.join(config.results_dir, f"{file_base}.yaml")

        # Extract the base command (uv run script_name)
        base_cmd_parts = commands[0].split()[:3]  # Get "uv run script.py"
        base_command = " ".join(base_cmd_parts)

        # Extract argument parts from each command
        template_args = []
        for cmd in commands:
            # Remove the base command to get just the arguments
            args_part = cmd[len(base_command) :].strip()
            template_args.append(args_part)

        # Determine job prefix
        job_prefix = config.job_prefix
        if not job_prefix:
            job_prefix = f"{config.algorithm}_{env_safe}_random"

        # Create YAML content compatible with job batcher
        yaml_content = {
            "command_template": f"{base_command} {{{{command}}}}",
            "template_args": {"command": template_args},
            "job_prefix": job_prefix,
            "workers_per_gpu": config.workers_per_gpu,
        }

        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

        if config.verbose:
            print(f"✓ Saved YAML config to: {yaml_file}")
            print(f"\nYAML structure:")
            print(f"  - Command template: {yaml_content['command_template']}")
            print(f"  - Number of jobs: {len(template_args)}")
            print(f"  - Job prefix: {job_prefix}")
            print(f"  - Workers per GPU: {config.workers_per_gpu}")


def generate_random_yaml_configs(config: RandomYamlConfig) -> None:
    """Main entry point for generating random hyperparameter configurations.

    Args:
        config: RandomYamlConfig object with all generation parameters
    """
    if config.verbose:
        print("=" * 50)
        print("RANDOM HYPERPARAMETER CONFIG GENERATOR")
        print("=" * 50)

    # Generate configurations
    commands = generate_configs(config)

    # Save configurations
    save_configs(config, commands)

    if config.verbose:
        print("=" * 50)
        print("GENERATION COMPLETE")
        print("=" * 50)
        print(f"\nGenerated {len(commands)} configurations")
        print(f"Output directory: {config.results_dir}")
        print("\nExample command:")
        print(commands[0][:200] + ("..." if len(commands[0]) > 200 else ""))


if __name__ == "__main__":
    config = tyro.cli(RandomYamlConfig)
    generate_random_yaml_configs(config)
