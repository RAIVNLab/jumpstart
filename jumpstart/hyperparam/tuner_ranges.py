from jumpstart.training.train_bc import bc_train
from jumpstart.training.train_bcq import bcq_train
from jumpstart.training.train_cql import cql_train
from jumpstart.training.train_dt import dt_train
from jumpstart.training.train_iql import iql_train
from jumpstart.training.train_rebrac import rebrac_train
from jumpstart.training.train_diffusion import diffusion_train
from jumpstart.training.train_vqbet import vqbet_train
from jumpstart.training.train_act import act_train

from abc import ABC, abstractmethod
from typing import Dict, Union, Any
from pathlib import Path
from dataclasses import dataclass
import random
import math
import json
import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

PathLike = Union[str, Path]
Numeric = Union[int, float]


############################################################
# JSON Serialization for metrics
############################################################


class MetricEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy arrays and torch tensors."""

    def default(self, obj: Any) -> Any:
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return {
                "__type__": "numpy.ndarray",
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": obj.tolist(),
            }

        # Handle numpy scalar types
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()

        # Handle torch tensors
        if TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
            return {
                "__type__": "torch.Tensor",
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
                "device": str(obj.device),
                "data": obj.detach().cpu().numpy().tolist(),
            }

        # Handle dictionaries with numpy/torch values
        if isinstance(obj, dict):
            return {
                k: (
                    self.default(v)
                    if isinstance(v, (np.ndarray, np.integer, np.floating))
                    or (TORCH_AVAILABLE and isinstance(v, torch.Tensor))
                    else v
                )
                for k, v in obj.items()
            }

        return super().default(obj)


class MetricDecoder(json.JSONDecoder):
    """Custom JSON decoder that reconstructs numpy arrays and torch tensors."""

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj: Dict) -> Any:
        if "__type__" not in obj:
            return obj

        obj_type = obj["__type__"]

        # Reconstruct numpy arrays
        if obj_type == "numpy.ndarray":
            data = np.array(obj["data"], dtype=np.dtype(obj["dtype"]))
            return data.reshape(obj["shape"])

        # Reconstruct torch tensors
        if obj_type == "torch.Tensor":
            if not TORCH_AVAILABLE:
                raise ImportError("torch is required to deserialize torch.Tensor objects")

            data = np.array(obj["data"], dtype=np.float32)
            tensor = torch.from_numpy(data).reshape(obj["shape"])

            # Restore dtype
            dtype_str = obj["dtype"].replace("torch.", "")
            if hasattr(torch, dtype_str):
                tensor = tensor.to(getattr(torch, dtype_str))

            # Restore device (but keep on CPU by default for safety)
            # User can move to GPU manually if needed
            if obj["device"] != "cpu":
                # Store original device info but keep on CPU
                tensor._original_device = obj["device"]

            return tensor

        return obj


############################################################
# Run info storage
############################################################


@dataclass
class RunResult:
    run_id: str
    hyperparameters: Dict
    metric: Numeric
    metadata: Dict = None

    def get_result(self) -> "RunResult":
        return RunResult(
            run_id=self.run_id,
            hyperparameters=self.hyperparameters,
            metric=self.metric,
        )


############################################################
# Hyperparameter ranges
############################################################


class BaseRange(ABC):
    @abstractmethod
    def sample(self) -> float:
        pass


class NumericRange(BaseRange):
    def __init__(self, min_value: float, max_value: float):
        """Defines a numeric range for hyperparameter sampling.
        Args:
            min_value (float): Minimum value of the range.
            max_value (float): Maximum value of the range.
        """
        self.min_value = min_value
        self.max_value = max_value


class LogNumericRange(NumericRange):
    def sample(self) -> float:
        log_min = math.log(self.min_value)
        log_max = math.log(self.max_value)
        return math.exp(random.uniform(log_min, log_max))


class UniformRange(NumericRange):
    def sample(self) -> float:
        return random.uniform(self.min_value, self.max_value)


class LogUniformRange(LogNumericRange):
    def sample(self) -> float:
        log_min = math.log(self.min_value)
        log_max = math.log(self.max_value)
        return math.exp(random.uniform(log_min, log_max))


class IntegerRange(NumericRange):
    def sample(self) -> int:
        return random.randint(int(self.min_value), int(self.max_value))


class LogIntegerRange(LogNumericRange):
    def sample(self) -> int:
        log_min = math.log(self.min_value)
        log_max = math.log(self.max_value)
        return int(math.exp(random.uniform(log_min, log_max)))


class QuantizedIntegerRange(NumericRange):
    def __init__(self, min_value: int, max_value: int, q: int):
        """Defines a quantized integer range for hyperparameter sampling.
        Args:
            min_value (int): Minimum value of the range.
            max_value (int): Maximum value of the range.
            q (int): Quantization step size.
        """
        super().__init__(min_value, max_value)
        self.q = q

    def sample(self) -> int:
        raw_sample = random.randint(int(self.min_value), int(self.max_value))
        quantized_sample = raw_sample // self.q * self.q
        return int(min(max(quantized_sample, self.min_value), self.max_value))


class LogQuantizedIntegerRange(LogNumericRange):
    def __init__(self, min_value: int, max_value: int, q: int):
        """Defines a log-quantized integer range for hyperparameter sampling.
        Args:
            min_value (int): Minimum value of the range.
            max_value (int): Maximum value of the range.
            q (int): Quantization step size.
        """
        super().__init__(min_value, max_value)
        self.q = q

    def sample(self) -> int:
        log_min = math.log(self.min_value)
        log_max = math.log(self.max_value)
        raw_sample = int(math.exp(random.uniform(log_min, log_max)))
        quantized_sample = raw_sample // self.q * self.q
        return int(min(max(quantized_sample, self.min_value), self.max_value))


class GaussianRange(BaseRange):
    def __init__(self, mean: float, stddev: float):
        """Defines a Gaussian range for hyperparameter sampling.
        Args:
            mean (float): Mean of the Gaussian distribution.
            stddev (float): Standard deviation of the Gaussian distribution.
        """
        self.mean = mean
        self.stddev = stddev

    def sample(self) -> float:
        return random.gauss(self.mean, self.stddev)


class ChoiceRange(BaseRange):
    def __init__(self, choices: list):
        """Defines a uniform categorical range for hyperparameter sampling.
        Args:
            categories (list): List of categorical options.
        """
        self.choices = choices

    def sample(self) -> any:
        return random.choice(self.choices)


class ConstantRange(BaseRange):
    def __init__(self, value: any):
        """Defines a constant range that always returns the same value.
        Args:
            value (any): The constant value to return.
        """
        self.value = value

    def sample(self) -> any:
        return self.value


# ============================================================
# CQL Hyperparameter Ranges (Continuous)
# ============================================================

CQL_CONTINUOUS_RANGES = {
    "actor_learning_rate": LogUniformRange(1e-5, 1e-2),
    "critic_learning_rate": LogUniformRange(1e-5, 1e-2),
    "temp_learning_rate": LogUniformRange(1e-5, 1e-2),
    "alpha_learning_rate": LogUniformRange(1e-5, 1e-2),
    "gamma": LogUniformRange(0.9, 0.999),
    "tau": UniformRange(0.001, 0.02),
    "conservative_weight": UniformRange(0.1, 10.0),
    "alpha_threshold": UniformRange(-1.0, 20.0),
    "n_action_samples": IntegerRange(5, 20),
    # "batch_size": QuantizedIntegerRange(128, 512, q=64),
    "n_critics": IntegerRange(1, 4),
    # "critic_ln": ChoiceRange([False, True]),
    # "n_steps": LogQuantizedIntegerRange(100000, 3000000, q=100000),
    # Note: soft_q_backup and max_q_backup should be handled with mutually exclusive logic
    # "backup_type": ChoiceRange(["soft_q", "max_q", "none"]),
    # Fixed configuration parameters
    "wandb": ConstantRange(False),
    "tensorboard": ConstantRange(False),
    "scale_rewards": ConstantRange(True),
    "reward_scale": ConstantRange(0.0),
    "eval_with_env": ConstantRange(True),
}


# ============================================================
# CQL Hyperparameter Ranges (Discrete)
# ============================================================

CQL_DISCRETE_RANGES = {
    "actor_learning_rate": LogUniformRange(1e-5, 1e-2),
    "gamma": LogUniformRange(0.9, 0.999),
    # "batch_size": QuantizedIntegerRange(128, 512, q=64),
    "n_critics": IntegerRange(1, 4),
    "target_update_interval": IntegerRange(1000, 10000),
    "alpha": UniformRange(0.1, 2.0),
    # "critic_ln": ChoiceRange([False, True]),
    # "n_steps": LogQuantizedIntegerRange(100000, 3000000, q=100000),
    # Fixed configuration parameters
    "wandb": ConstantRange(False),
    "tensorboard": ConstantRange(False),
    "scale_rewards": ConstantRange(True),
    "reward_scale": ConstantRange(0.0),
    "eval_with_env": ConstantRange(True),
}


# ============================================================
# BC Hyperparameter Ranges (Continuous)
# ============================================================

BC_CONTINUOUS_RANGES = {
    "learning_rate": LogUniformRange(1e-5, 1e-2),
    # "batch_size": QuantizedIntegerRange(64, 512, q=64),
    "weight_decay": LogUniformRange(1e-4, 1e-1),
    "clip_grad_norm": UniformRange(0.1, 2.0),
    "dropout": UniformRange(0.0, 0.3),
    "hidden_units": ChoiceRange([[128, 128], [256, 256], [512, 512]]),
    "policy_type": ChoiceRange(["deterministic", "stochastic"]),
    # Fixed configuration parameters
    "wandb": ConstantRange(False),
    "tensorboard": ConstantRange(False),
    "eval_with_env": ConstantRange(True),
}


# ============================================================
# BC Hyperparameter Ranges (Discrete)
# ============================================================

BC_DISCRETE_RANGES = {
    "learning_rate": LogUniformRange(1e-5, 1e-2),
    # "batch_size": QuantizedIntegerRange(64, 512, q=64),
    "weight_decay": LogUniformRange(1e-4, 1e-1),
    "clip_grad_norm": UniformRange(0.1, 2.0),
    "dropout": UniformRange(0.0, 0.3),
    "hidden_units": ChoiceRange([[128, 128], [256, 256], [512, 512]]),
    "reg_factor": UniformRange(0.0, 1.0),
    # Fixed configuration parameters
    "wandb": ConstantRange(False),
    "tensorboard": ConstantRange(False),
    "eval_with_env": ConstantRange(True),
}


# ============================================================
# BCP Hyperparameter Ranges (Continuous)
# ============================================================

BCP_CONTINUOUS_RANGES = {
    "learning_rate": LogUniformRange(1e-5, 1e-2),
    # "batch_size": QuantizedIntegerRange(64, 512, q=64),
    "weight_decay": LogUniformRange(1e-4, 1e-1),
    "clip_grad_norm": UniformRange(0.1, 2.0),
    "dropout": UniformRange(0.0, 0.3),
    "hidden_units": ChoiceRange([[128, 128], [256, 256], [512, 512]]),
    "train_percentile": UniformRange(0.0, 90.0),
    "policy_type": ChoiceRange(["deterministic", "stochastic"]),
    # Fixed configuration parameters
    "wandb": ConstantRange(False),
    "tensorboard": ConstantRange(False),
    "eval_with_env": ConstantRange(True),
}


# ============================================================
# BCP Hyperparameter Ranges (Discrete)
# ============================================================

BCP_DISCRETE_RANGES = {
    "learning_rate": LogUniformRange(1e-5, 1e-2),
    # "batch_size": QuantizedIntegerRange(64, 512, q=64),
    "weight_decay": LogUniformRange(1e-4, 1e-1),
    "clip_grad_norm": UniformRange(0.1, 2.0),
    "dropout": UniformRange(0.0, 0.3),
    "hidden_units": ChoiceRange([[128, 128], [256, 256], [512, 512]]),
    "train_percentile": UniformRange(0.0, 90.0),
    "reg_factor": UniformRange(0.0, 1.0),
    # Fixed configuration parameters
    "wandb": ConstantRange(False),
    "tensorboard": ConstantRange(False),
    "eval_with_env": ConstantRange(True),
}


# ============================================================
# DT Hyperparameter Ranges
# ============================================================

DT_RANGES = {
    "learning_rate": LogUniformRange(1e-5, 1e-2),
    # "batch_size": QuantizedIntegerRange(64, 512, q=64),
    "weight_decay": LogUniformRange(1e-4, 1e-1),
    "clip_grad_norm": UniformRange(0.1, 2.0),
    "context_size": ChoiceRange([10, 20, 30]),
    "hidden_units": ChoiceRange([[128, 128, 128], [256, 256, 256]]),
    "all_dropout": UniformRange(0.0, 0.3),
    "num_heads": ChoiceRange([1, 2, 4]),
    "num_layers": IntegerRange(2, 5),
    "gamma": LogUniformRange(0.9, 0.999),
    "reward_scale": ChoiceRange([0.0, 1.0]),
    # "scale_rewards": ChoiceRange([True, False]),
    # Fixed configuration parameters
    "wandb": ConstantRange(False),
    "tensorboard": ConstantRange(False),
    "eval_with_env": ConstantRange(True),
}

# TODO: do BCQ
# ============================================================
# BCQ Hyperparameter Ranges (Continuous)
# ============================================================

BCQ_CONTINUOUS_RANGES = {
    "actor_learning_rate": LogUniformRange(1e-5, 1e-2),
    "critic_learning_rate": LogUniformRange(1e-5, 1e-2),
    "imitator_learning_rate": LogUniformRange(1e-5, 1e-2),
    "gamma": LogUniformRange(0.9, 0.999),
    "tau": UniformRange(0.001, 0.02),
    "n_critics": IntegerRange(1, 4),
    "update_actor_interval": IntegerRange(1, 4),
    "lam": UniformRange(0.5, 0.95),
    "n_action_samples": IntegerRange(20, 200),
    "rl_start_step": IntegerRange(0, 10000),
    "beta": UniformRange(0.1, 0.9),
    "wandb": ConstantRange(False),
    "tensorboard": ConstantRange(False),
    "eval_with_env": ConstantRange(True),
    "scale_rewards": ConstantRange(True),
    "reward_scale": ConstantRange(0.0),
}


# ============================================================
# BCQ Hyperparameter Ranges (Discrete)
# ============================================================

BCQ_DISCRETE_RANGES = {
    "gamma": LogUniformRange(0.9, 0.999),
    "actor_learning_rate": LogUniformRange(1e-5, 1e-2),
    "n_critics": IntegerRange(1, 4),
    "target_update_interval": IntegerRange(2000, 20000),
    "action_flexibility": UniformRange(0.01, 0.1),
    "beta": UniformRange(0.1, 0.9),
    "wandb": ConstantRange(False),
    "tensorboard": ConstantRange(False),
    "eval_with_env": ConstantRange(True),
    "scale_rewards": ConstantRange(True),
    "reward_scale": ConstantRange(0.0),
}


# ============================================================
# ReBRAC Hyperparameter Ranges (Continuous Only)
# ============================================================

REBRAC_RANGES = {
    "actor_learning_rate": LogUniformRange(1e-5, 1e-2),
    "critic_learning_rate": LogUniformRange(1e-5, 1e-2),
    "actor_beta": LogUniformRange(1e-5, 1.0),
    "critic_beta": LogUniformRange(1e-5, 1.0),
    "gamma": LogUniformRange(0.9, 0.999),
    "tau": UniformRange(0.001, 0.02),
    "n_critics": IntegerRange(1, 4),
    "target_smoothing_sigma": UniformRange(0.0, 0.5),
    "target_smoothing_clip": UniformRange(0.0, 1.0),
    # "critic_ln": ChoiceRange([False, True]),
    #     "batch_size": QuantizedIntegerRange(256, 2048, q=64),
    # Fixed configuration parameters
    "wandb": ConstantRange(False),
    "tensorboard": ConstantRange(False),
    "scale_rewards": ConstantRange(True),
    "reward_scale": ConstantRange(0.0),
    "eval_with_env": ConstantRange(True),
}


# ============================================================
# IQL Hyperparameter Ranges (Continuous Only)
# ============================================================

IQL_RANGES = {
    "actor_learning_rate": LogUniformRange(1e-5, 1e-2),
    "critic_learning_rate": LogUniformRange(1e-5, 1e-2),
    #    "batch_size": QuantizedIntegerRange(128, 1024, q=64),
    "gamma": LogUniformRange(0.9, 0.999),
    "tau": UniformRange(0.001, 0.02),
    "n_critics": IntegerRange(1, 4),
    "expectile": UniformRange(0.5, 0.9),
    "weight_temp": UniformRange(1.0, 10.0),
    # "critic_ln": ChoiceRange([False, True]),
    # Fixed configuration parameters
    "wandb": ConstantRange(False),
    "tensorboard": ConstantRange(False),
    "scale_rewards": ConstantRange(True),
    "reward_scale": ConstantRange(0.0),
    "eval_with_env": ConstantRange(True),
}


# # ============================================================
# # DQN Hyperparameter Ranges (Discrete Only)
# # ============================================================

# DQN_RANGES = {
#     "learning_rate": LogUniformRange(1e-5, 1e-2),
#     "gamma": LogUniformRange(0.9, 0.999),
#     "n_critics": IntegerRange(1, 4),
#     "target_update_interval": IntegerRange(1000, 10000),
#     "batch_size": QuantizedIntegerRange(64, 512, q=64),
#     # Fixed configuration parameters
#     "wandb": ConstantRange(False),
#     "tensorboard": ConstantRange(False),
#     "scale_rewards": ConstantRange(True),
#     "reward_scale": ConstantRange(0.0),
#     "eval_with_env": ConstantRange(True),
# }

# # ============================================================
# # DoubleDQN Hyperparameter Ranges (Discrete Only)
# # ============================================================


# DDQN_RANGES = {
#     "learning_rate": LogUniformRange(1e-5, 1e-2),
#     "gamma": LogUniformRange(0.9, 0.999),
#     "n_critics": IntegerRange(1, 4),
#     "target_update_interval": IntegerRange(1000, 10000),
#     "batch_size": QuantizedIntegerRange(64, 512, q=64),
#     "n_steps": IntegerRange(100000, 3000000),
#     # Fixed configuration parameters
#     "wandb": ConstantRange(False),
#     "tensorboard": ConstantRange(False),
#     "scale_rewards": ConstantRange(True),
#     "reward_scale": ConstantRange(0.0),
#     "eval_with_env": ConstantRange(True),
# }


# ============================================================
# Diffusion Policy Hyperparameter Ranges (Continuous Only)
# ============================================================

DIFFUSION_RANGES = {
    "learning_rate": LogUniformRange(1e-5, 1e-3),
    "n_diffusion_steps": ChoiceRange([50, 100]),
    "num_inference_steps": IntegerRange(20, 100),
    "action_chunk_size": ChoiceRange([8, 16]),
    "action_horizon": ChoiceRange([1, 4]),
    "weight_decay": LogUniformRange(1e-6, 1e-2),
    "clip_grad_norm": UniformRange(0.1, 2.0),
    # Fixed configuration parameters
    "wandb": ConstantRange(False),
    "tensorboard": ConstantRange(False),
    "eval_with_env": ConstantRange(True),
}


# ============================================================
# VQ-BeT Hyperparameter Ranges (Continuous Only)
# hidden_dim is the GPT d_model; must be divisible by transformer_heads
# ============================================================

VQBET_RANGES = {
    "learning_rate": LogUniformRange(1e-5, 1e-2),
    "offset_learning_rate": LogUniformRange(1e-5, 1e-2),
    "action_chunk_size": ChoiceRange([2, 8]),
    "action_horizon": ChoiceRange([1, 4]),
    "vqvae_steps_ratio": UniformRange(0.1, 0.4),
    "commitment_weight": LogUniformRange(0.01, 1.0),
    "focal_gamma": UniformRange(0.5, 5.0),
    "offset_loss_weight": UniformRange(0.1, 5.0),
    "primary_code_weight": UniformRange(1.0, 10.0),
    "secondary_code_weight": UniformRange(0.1, 3.0),
    "weight_decay": LogUniformRange(1e-6, 1e-2),
    "clip_grad_norm": UniformRange(0.1, 2.0),
    # Fixed configuration parameters
    "wandb": ConstantRange(False),
    "tensorboard": ConstantRange(False),
    "eval_with_env": ConstantRange(True),
}


# ============================================================
# ACT Hyperparameter Ranges (Continuous Only)
# ============================================================

ACT_RANGES = {
    "learning_rate": LogUniformRange(1e-5, 1e-2),
    "kl_weight": LogUniformRange(1.0, 10.0),
    "action_chunk_size": ChoiceRange([2, 8]),
    "action_horizon": ChoiceRange([1, 4]),
    "weight_decay": LogUniformRange(1e-6, 1e-2),
    "clip_grad_norm": UniformRange(0.1, 2.0),
    # Fixed configuration parameters
    "wandb": ConstantRange(False),
    "tensorboard": ConstantRange(False),
    "eval_with_env": ConstantRange(True),
}


NAME2ALGO = {
    "cql": cql_train,
    "bc": bc_train,
    "bcp": bc_train,
    "bcq": bcq_train,
    "dt": dt_train,
    "rebrac": rebrac_train,
    "iql": iql_train,
    "diffusion": diffusion_train,
    "vqbet": vqbet_train,
    "act": act_train,
}

# ============================================================
# Utility function to get ranges by algorithm and environment type
# ============================================================


def get_search_space(algorithm: str, env_type: str = "continuous"):
    """Get the hyperparameter search space for a given algorithm and environment type.

    Args:
        algorithm: Algorithm name (for example, "cql", "bc", "bcp", or "dt")
        env_type: Environment type ("continuous" or "discrete")

    Returns:
        Dictionary mapping hyperparameter names to Range objects

    Raises:
        ValueError: If algorithm is unknown or env_type is invalid for the algorithm
    """
    if algorithm == "cql":
        if env_type == "continuous":
            return CQL_CONTINUOUS_RANGES.copy()
        elif env_type == "discrete":
            return CQL_DISCRETE_RANGES.copy()
        else:
            raise ValueError(f"Invalid env_type for CQL: {env_type}")
    elif algorithm == "bc":
        if env_type == "continuous":
            return BC_CONTINUOUS_RANGES.copy()
        elif env_type == "discrete":
            return BC_DISCRETE_RANGES.copy()
        else:
            raise ValueError(f"Invalid env_type for BC: {env_type}")
    elif algorithm == "bcp":
        if env_type == "continuous":
            return BCP_CONTINUOUS_RANGES.copy()
        elif env_type == "discrete":
            return BCP_DISCRETE_RANGES.copy()
        else:
            raise ValueError(f"Invalid env_type for BCP: {env_type}")
    elif algorithm == "bcq":
        if env_type == "continuous":
            return BCQ_CONTINUOUS_RANGES.copy()
        elif env_type == "discrete":
            return BCQ_DISCRETE_RANGES.copy()
        else:
            raise ValueError(f"Invalid env_type for BCQ: {env_type}")
    elif algorithm == "dt":
        if env_type not in {"continuous", "discrete"}:
            raise ValueError(f"Invalid env_type for DT: {env_type}")
        return DT_RANGES.copy()
    elif algorithm == "rebrac":
        if env_type != "continuous":
            raise ValueError("ReBRAC only supports continuous action spaces")
        return REBRAC_RANGES.copy()
    elif algorithm == "iql":
        if env_type != "continuous":
            raise ValueError("IQL only supports continuous action spaces")
        return IQL_RANGES.copy()
    elif algorithm == "diffusion":
        if env_type != "continuous":
            raise ValueError("Diffusion only supports continuous action spaces")
        return DIFFUSION_RANGES.copy()
    elif algorithm == "vqbet":
        if env_type != "continuous":
            raise ValueError("VQ-BeT only supports continuous action spaces")
        return VQBET_RANGES.copy()
    elif algorithm == "act":
        if env_type != "continuous":
            raise ValueError("ACT only supports continuous action spaces")
        return ACT_RANGES.copy()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def get_search_space_and_objective_fn(algorithm: str, env_type: str = "continuous"):
    """Get the search space and objective function for an algorithm and environment type.

    Args:
        algorithm: Algorithm name (for example, "cql", "bc", "bcp", or "dt")
        env_type: Environment type ("continuous" or "discrete")

    Returns:
        Tuple of (search space dictionary, objective function)

    Raises:
        ValueError: If algorithm is unknown or env_type is invalid for the algorithm
    """

    search_space = get_search_space(algorithm, env_type=env_type)
    objective_fn = NAME2ALGO.get(algorithm)
    if objective_fn is None:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return search_space, objective_fn
