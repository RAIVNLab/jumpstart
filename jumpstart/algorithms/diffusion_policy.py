"""Diffusion Policy with action chunking for offline RL.

Follows the official implementation: github.com/real-stanford/diffusion_policy

Implements a 1D temporal U-Net denoiser with FiLM conditioning (obs_as_global_cond),
cosine noise schedule, and DDIM inference. Continuous action space only.
Follows d3rlpy extension API.
"""

import copy
import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from d3rlpy.algos.qlearning.base import QLearningAlgoBase, QLearningAlgoImplBase
from d3rlpy.base import DeviceArg, LearnableConfig, register_learnable
from d3rlpy.constants import ActionSpace
from d3rlpy.optimizers import OptimizerFactory, make_optimizer_field
from d3rlpy.torch_utility import CudaGraphWrapper, Modules, TorchMiniBatch
from d3rlpy.types import Shape, TorchObservation

__all__ = ["DiffusionPolicyConfig", "DiffusionPolicy"]


# ============================================================
# Components (matching official: Conv->GroupNorm->Mish, FiLM)
# ============================================================


class SinusoidalPosEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish (matching official order)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two Conv1dBlocks with FiLM conditioning between them (matching official)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
            ]
        )
        # FiLM: Mish -> Linear (matching official cond_encoder)
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale, bias = embed[:, 0], embed[:, 1]
            h = scale * h + bias
        else:
            h = h + embed.unsqueeze(-1)
        h = self.blocks[1](h)
        return h + self.residual_conv(x)


class ConditionalUNet1D(nn.Module):
    """Temporal 1D U-Net for noise prediction (matching official architecture).

    Uses growing channel dimensions, 2 ResBlocks per level, 2 mid blocks,
    and FiLM conditioning from concatenated [timestep_emb, global_cond].
    """

    def __init__(
        self,
        action_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()
        dsed = diffusion_step_embed_dim

        # Timestep encoder: sinusoidal -> MLP (matching official)
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmbed(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        cond_dim = dsed + global_cond_dim

        # Input projection
        self.input_proj = nn.Conv1d(action_dim, down_dims[0], 1)

        n_levels = len(down_dims)

        def _make_block(dim_in: int, dim_out: int) -> ConditionalResidualBlock1D:
            return ConditionalResidualBlock1D(
                dim_in, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale
            )

        # Down path: 2 ResBlocks per level + Downsample (except last)
        self.down_modules = nn.ModuleList()
        for i in range(n_levels):
            dim_in = down_dims[0] if i == 0 else down_dims[i - 1]
            dim_out = down_dims[i]
            is_last = i == n_levels - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        _make_block(dim_in, dim_out),
                        _make_block(dim_out, dim_out),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Mid: 2 ResBlocks at deepest level
        self.mid_modules = nn.ModuleList(
            [_make_block(down_dims[-1], down_dims[-1]), _make_block(down_dims[-1], down_dims[-1])]
        )

        # Up path: mirrors down (excluding first level), all with Upsample
        self.up_modules = nn.ModuleList()
        for i in range(n_levels - 2, -1, -1):
            dim_out = down_dims[i + 1]  # skip connection channels
            dim_target = down_dims[i]
            self.up_modules.append(
                nn.ModuleList(
                    [
                        _make_block(dim_out * 2, dim_target),  # skip concat doubles channels
                        _make_block(dim_target, dim_target),
                        Upsample1d(dim_target),
                    ]
                )
            )

        # Final conv (matching official)
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, global_cond: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:

        t_emb = self.diffusion_step_encoder(timestep.float())
        cond = torch.cat([t_emb, global_cond], dim=-1)

        h = self.input_proj(x.transpose(1, 2))  # (B, down_dims[0], K)

        # Down with skip connections
        skips = []
        for resnet1, resnet2, downsample in self.down_modules:
            h = resnet1(h, cond)
            h = resnet2(h, cond)
            skips.append(h)
            h = downsample(h)

        # Mid
        for mid_block in self.mid_modules:
            h = mid_block(h, cond)

        # Up with skip connections (pops from deepest first)
        for resnet1, resnet2, upsample in self.up_modules:
            h = torch.cat([h, skips.pop()], dim=1)
            h = resnet1(h, cond)
            h = resnet2(h, cond)
            h = upsample(h)

        out = self.final_conv(h).transpose(1, 2)  # (B, K+pad, D)
        return out


class CosineNoiseScheduler:
    """Cosine beta schedule (squaredcos_cap_v2) with DDIM stepping."""

    def __init__(self, n_steps: int, s: float = 0.008):
        self.n_steps = n_steps
        t = torch.arange(n_steps + 1, dtype=torch.float64) / n_steps
        alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]

        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        betas = torch.clamp(betas, max=0.999)

        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self.alpha_cumprod = alpha_cumprod.float()
        self.sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod).float()
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod).float()

    def to(self, device: str) -> "CosineNoiseScheduler":
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        self.sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(device)
        self.sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod.to(device)
        return self

    def add_noise(self, x: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s_a = self.sqrt_alpha_cumprod[t].view(-1, 1, 1)
        s_om = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1)
        return s_a * x + s_om * noise

    def step_ddim(
        self, noise_pred: torch.Tensor, t: int, t_prev: int, x_t: torch.Tensor
    ) -> torch.Tensor:
        a_t = self.alpha_cumprod[t]
        a_prev = self.alpha_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
        x0_pred = (x_t - torch.sqrt(1 - a_t) * noise_pred) / torch.sqrt(a_t)
        return torch.sqrt(a_prev) * x0_pred + torch.sqrt(1 - a_prev) * noise_pred


class InputNormalizer(nn.Module):
    """Min-max normalization to [-1, 1]."""

    def __init__(self, data_min: torch.Tensor, data_max: torch.Tensor):
        super().__init__()
        data_range = data_max - data_min
        data_range = torch.clamp(data_range, min=1e-4)
        scale = 2.0 / data_range
        offset = -1.0 - scale * data_min
        self.register_buffer("scale", scale)
        self.register_buffer("offset", offset)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.offset

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.offset) / self.scale


# ============================================================
# d3rlpy Integration
# ============================================================


@dataclasses.dataclass(frozen=True)
class DiffusionPolicyModules(Modules):
    denoiser: ConditionalUNet1D
    optim: "OptimizerWrapper"  # noqa: F821


class DiffusionPolicyImpl(QLearningAlgoImplBase):
    _modules: DiffusionPolicyModules

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DiffusionPolicyModules,
        n_diffusion_steps: int,
        num_inference_steps: int,
        action_chunk_size: int,
        action_horizon: int,
        original_action_dim: int,
        obs_normalizer: InputNormalizer | None,
        action_normalizer: InputNormalizer | None,
        ema_decay: float,
        compiled: bool,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._n_diffusion_steps = n_diffusion_steps
        self._num_inference_steps = min(n_diffusion_steps, num_inference_steps)
        self._action_chunk_size = action_chunk_size
        self._action_horizon = action_horizon
        self._original_action_dim = original_action_dim
        self._scheduler = CosineNoiseScheduler(n_diffusion_steps).to(device)
        self._obs_normalizer = obs_normalizer
        self._action_normalizer = action_normalizer

        # EMA
        self._ema_decay = ema_decay
        if ema_decay > 0:
            self._ema_denoiser = copy.deepcopy(modules.denoiser)
            self._ema_denoiser.requires_grad_(False)
        else:
            self._ema_denoiser = None

        schedule = torch.linspace(n_diffusion_steps - 1, 0, self._num_inference_steps + 1).long()
        schedule[-1] = -1  # final step denoises fully (alpha_prev=1.0)
        self._inference_schedule = schedule.numpy().tolist()

        self._action_cache: torch.Tensor | None = None
        self._cache_idx: torch.Tensor | None = None  # per-env cache index

        self._compute_grad = (
            CudaGraphWrapper(self._compute_grad_impl) if compiled else self._compute_grad_impl
        )

    def _compute_grad_impl(self, batch: TorchMiniBatch) -> torch.Tensor:
        self._modules.optim.zero_grad()
        actions = batch.actions.reshape(-1, self._action_chunk_size, self._original_action_dim)
        obs = batch.observations
        # Normalize inputs (matching official: normalize obs+actions to [-1,1])
        if self._obs_normalizer is not None:
            obs = self._obs_normalizer.normalize(obs)
        if self._action_normalizer is not None:
            actions = self._action_normalizer.normalize(actions)
        B = actions.shape[0]
        t = torch.randint(0, self._n_diffusion_steps, (B,), device=actions.device)
        noise = torch.randn_like(actions)
        noisy_actions = self._scheduler.add_noise(actions, noise, t)
        noise_pred = self._modules.denoiser(noisy_actions, obs, t)
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        return loss

    def inner_update(self, batch: TorchMiniBatch, grad_step: int) -> dict[str, float]:
        loss = self._compute_grad(batch)
        self._modules.optim.step()
        # Update EMA weights
        if self._ema_denoiser is not None:
            decay = self._ema_decay
            with torch.no_grad():
                for p_ema, p_model in zip(
                    self._ema_denoiser.parameters(), self._modules.denoiser.parameters()
                ):
                    p_ema.mul_(decay).add_(p_model, alpha=1 - decay)
        return {"loss": loss.item()}

    def load_model(self, f) -> None:
        super().load_model(f)
        if self._ema_denoiser is not None:
            # d3rlpy checkpoints only include registered modules, so recoveries
            # must not use a freshly initialized EMA copy for inference.
            self._ema_denoiser.load_state_dict(self._modules.denoiser.state_dict())
            self._ema_denoiser.requires_grad_(False)

    def set_action_horizon(self, action_horizon: int | None) -> int:
        old_action_horizon = self._action_horizon
        if action_horizon is None:
            action_horizon = self._action_chunk_size
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        self._action_horizon = min(self._action_chunk_size, int(action_horizon))
        self.reset_action_cache()
        return old_action_horizon

    def _denoise(self, x: TorchObservation) -> torch.Tensor:
        obs = x
        if self._obs_normalizer is not None:
            obs = self._obs_normalizer.normalize(obs)
        B = obs.shape[0] if isinstance(obs, torch.Tensor) else obs[0].shape[0]
        x_t = torch.randn(
            B, self._action_chunk_size, self._original_action_dim, device=self._device
        )

        denoiser = self._ema_denoiser if self._ema_denoiser is not None else self._modules.denoiser
        for i in range(self._num_inference_steps):
            t = self._inference_schedule[i]
            t_prev = self._inference_schedule[i + 1]
            t_batch = torch.full((B,), t, device=self._device, dtype=torch.long)
            noise_pred = denoiser(x_t, obs, t_batch)
            x_t = self._scheduler.step_ddim(noise_pred, t, t_prev, x_t)

        # Clip to [-1, 1] and unnormalize back to original action space
        x_t = torch.clamp(x_t, -1.0, 1.0)
        if self._action_normalizer is not None:
            x_t = self._action_normalizer.unnormalize(x_t)
        return x_t

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        B = x.shape[0] if isinstance(x, torch.Tensor) else x[0].shape[0]

        if self._action_cache is not None and self._action_cache.shape[0] == B:
            # Mask of envs that still have valid cached actions
            valid = self._cache_idx < self._action_horizon
        else:
            valid = torch.zeros(B, dtype=torch.bool, device=self._device)

        if valid.all():
            # All envs served from cache
            action = self._action_cache[torch.arange(B, device=self._device), self._cache_idx]
            self._cache_idx += 1
            return action

        # Denoise for envs that need new chunks
        need = ~valid
        new_chunks = self._denoise(x)

        if self._action_cache is None or self._action_cache.shape[0] != B:
            self._action_cache = new_chunks
            self._cache_idx = torch.ones(B, dtype=torch.long, device=self._device)
            return self._action_cache[:, 0, :]

        # Merge: overwrite only the envs that needed refresh
        self._action_cache[need] = new_chunks[need]
        self._cache_idx[need] = 0

        action = self._action_cache[torch.arange(B, device=self._device), self._cache_idx.clone()]
        self._cache_idx += 1
        return action

    def reset_action_cache(self, done_mask: torch.Tensor | None = None) -> None:
        if done_mask is None:
            self._action_cache = None
            self._cache_idx = None
        elif self._cache_idx is not None:
            # Force re-denoise on next call for these envs
            self._cache_idx[done_mask] = self._action_horizon

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def inner_predict_value(self, x: TorchObservation, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Diffusion Policy does not support value estimation")


@dataclasses.dataclass()
class DiffusionPolicyConfig(LearnableConfig):
    learning_rate: float = 1e-4
    hidden_dim: int = 128  # base channel dim; U-Net uses [hidden_dim, hidden_dim*2, hidden_dim*4]
    kernel_size: int = 5
    n_diffusion_steps: int = 100
    num_inference_steps: int = 10
    action_chunk_size: int = 16
    action_horizon: int | None = None
    original_action_dim: int = 1
    compile_graph: bool = False
    obs_stat_min: tuple[float, ...] | None = None
    obs_stat_max: tuple[float, ...] | None = None
    action_stat_min: tuple[float, ...] | None = None
    action_stat_max: tuple[float, ...] | None = None
    ema_decay: float = 0.995
    optim_factory: OptimizerFactory = make_optimizer_field()

    def create(self, device: DeviceArg = False, enable_ddp: bool = False) -> "DiffusionPolicy":
        return DiffusionPolicy(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "diffusion_policy"


class DiffusionPolicy(QLearningAlgoBase[DiffusionPolicyImpl, DiffusionPolicyConfig]):
    def inner_create_impl(self, observation_shape: Shape, action_size: int) -> None:
        cfg = self._config
        obs_dim = observation_shape[0]
        h = cfg.hidden_dim

        denoiser = ConditionalUNet1D(
            action_dim=cfg.original_action_dim,
            global_cond_dim=obs_dim,
            diffusion_step_embed_dim=h,
            down_dims=(h, h * 2, h * 4),
            kernel_size=cfg.kernel_size,
        ).to(self._device)

        params_wrapper = nn.ModuleDict({"denoiser": denoiser})
        optim = cfg.optim_factory.create(
            params_wrapper.named_modules(),
            lr=cfg.learning_rate,
            compiled=self.compiled,
        )

        modules = DiffusionPolicyModules(denoiser=denoiser, optim=optim)

        # Build normalizers from dataset statistics
        obs_normalizer = None
        if cfg.obs_stat_min is not None and cfg.obs_stat_max is not None:
            obs_normalizer = InputNormalizer(
                torch.tensor(cfg.obs_stat_min, dtype=torch.float32, device=self._device),
                torch.tensor(cfg.obs_stat_max, dtype=torch.float32, device=self._device),
            )
        action_normalizer = None
        if cfg.action_stat_min is not None and cfg.action_stat_max is not None:
            action_normalizer = InputNormalizer(
                torch.tensor(cfg.action_stat_min, dtype=torch.float32, device=self._device),
                torch.tensor(cfg.action_stat_max, dtype=torch.float32, device=self._device),
            )

        self._impl = DiffusionPolicyImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            n_diffusion_steps=cfg.n_diffusion_steps,
            num_inference_steps=cfg.num_inference_steps,
            action_chunk_size=cfg.action_chunk_size,
            action_horizon=cfg.action_horizon or cfg.action_chunk_size,
            original_action_dim=cfg.original_action_dim,
            obs_normalizer=obs_normalizer,
            action_normalizer=action_normalizer,
            ema_decay=cfg.ema_decay,
            compiled=self.compiled,
            device=self._device,
        )

    def reset(self, done_mask: torch.Tensor | None = None) -> None:
        if self._impl is not None:
            self._impl.reset_action_cache(done_mask)

    def set_action_horizon(self, action_horizon: int | None) -> int:
        if self._impl is None:
            raise RuntimeError("DiffusionPolicy implementation has not been created")
        return self._impl.set_action_horizon(action_horizon)

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(DiffusionPolicyConfig)
