"""VQ-BeT (Vector Quantized Behavior Transformer) for offline RL.

Follows the official implementation: github.com/jayLEE0301/vq_bet_official

Three-stage training:
  1. VQ-VAE pretraining on action chunks (L1 recon + commitment loss)
  2. Joint code prediction + offset training (focal loss + L1 offset loss)
  3. Offset-only fine-tuning (L1 offset loss, bin head frozen)

Continuous action space only. Follows d3rlpy extension API.
"""

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses_json import config as dataclasses_json_config
from d3rlpy.algos.qlearning.base import QLearningAlgoBase, QLearningAlgoImplBase
from d3rlpy.base import DeviceArg, LearnableConfig, register_learnable
from d3rlpy.constants import ActionSpace
from d3rlpy.optimizers import AdamWFactory, OptimizerFactory, make_optimizer_field
from d3rlpy.serializable_config import CONFIG_STORAGE
from d3rlpy.torch_utility import CudaGraphWrapper, Modules, TorchMiniBatch
from d3rlpy.types import Shape, TorchObservation

__all__ = ["VQBeTConfig", "VQBeT"]


def _make_optional_optimizer_field() -> OptimizerFactory | None:
    metadata = CONFIG_STORAGE[OptimizerFactory]

    def encode(factory: OptimizerFactory | None) -> dict:
        if factory is None:
            return {"type": "none", "params": {}}
        return metadata.encoder(factory)

    def decode(data: dict) -> OptimizerFactory | None:
        if data is None or data.get("type") == "none":
            return None
        if "type" not in data:
            return AdamWFactory.deserialize_from_dict(data)
        return metadata.decoder(data)

    return dataclasses.field(
        default=None,
        metadata=dataclasses_json_config(encoder=encode, decoder=decode),
    )


# ============================================================
# Weight initialization (matching official: orthogonal for VQ-VAE)
# ============================================================


def _init_orthogonal(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ============================================================
# VQ-VAE Components (MLP-based, matching official)
# ============================================================


class VQEncoder(nn.Module):
    """MLP encoder: (B, K, D) -> (B, latent_dim)."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.apply(_init_orthogonal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1))  # (B, latent_dim)


class VQDecoder(nn.Module):
    """MLP decoder: (B, latent_dim) -> (B, K, D)."""

    def __init__(
        self,
        output_dim: int,
        latent_dim: int,
        action_chunk_size: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(_init_orthogonal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1, self.action_chunk_size, self.action_dim)


class ResidualVQ(nn.Module):
    """Residual vector quantization: each level quantizes the residual from previous levels.

    Unlike independent-group VQ, this produces a single latent_dim vector whose quantized
    form is the sum of codebook entries across all levels.
    """

    def __init__(self, n_levels: int, n_embed: int, latent_dim: int):
        super().__init__()
        self.n_levels = n_levels
        self.n_embed = n_embed
        self.latent_dim = latent_dim
        self.codebooks = nn.ParameterList(
            [
                nn.Parameter(torch.randn(n_embed, latent_dim) / latent_dim**0.5)
                for _ in range(n_levels)
            ]
        )

    def _nearest(self, z: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        dists = torch.cdist(z.unsqueeze(1), self.codebooks[level].unsqueeze(0)).squeeze(1)
        indices = dists.argmin(dim=-1)
        return self.codebooks[level][indices], indices

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) -> indices: (B, n_levels)."""
        all_indices = []
        residual = z
        for lv in range(self.n_levels):
            quantized, indices = self._nearest(residual, lv)
            residual = residual - quantized
            all_indices.append(indices)
        return torch.stack(all_indices, dim=1)

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (quantized_st, indices, commitment_loss)."""
        all_indices = []
        quantized_sum = torch.zeros_like(z)
        residual = z
        commitment_loss = torch.tensor(0.0, device=z.device)
        for lv in range(self.n_levels):
            quantized, indices = self._nearest(residual, lv)
            commitment_loss = commitment_loss + F.mse_loss(residual, quantized.detach())
            residual = residual - quantized.detach()
            quantized_sum = quantized_sum + quantized
            all_indices.append(indices)
        quantized_st = z + (quantized_sum - z).detach()
        return quantized_st, torch.stack(all_indices, dim=1), commitment_loss

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: (B, n_levels) -> (B, latent_dim). Sum of codebook entries across levels."""
        result = torch.zeros(indices.shape[0], self.latent_dim, device=indices.device)
        for lv in range(self.n_levels):
            result = result + self.codebooks[lv][indices[:, lv]]
        return result


# ============================================================
# Focal Loss (matching official, gamma=2.0)
# ============================================================


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        pt = logpt.exp()
        return (-((1 - pt) ** self.gamma) * logpt).mean()


# ============================================================
# Prediction Heads (matching official: 3-layer MLPs with 1024 hidden)
# ============================================================


class BinPredictionHead(nn.Module):
    """Predicts codebook index logits: (B, hidden) -> (B, n_levels, n_embed)."""

    def __init__(self, input_dim: int, n_levels: int, n_embed: int):
        super().__init__()
        self.n_levels = n_levels
        self.n_embed = n_embed
        out = n_levels * n_embed
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1, self.n_levels, self.n_embed)


class OffsetPredictionHead(nn.Module):
    """Predicts per-code continuous offset: (B, hidden) -> (B, total_codes, act_flat)."""

    def __init__(self, input_dim: int, n_levels: int, n_embed: int, act_flat: int):
        super().__init__()
        self.n_levels = n_levels
        self.n_embed = n_embed
        self.act_flat = act_flat
        total_codes = n_levels * n_embed
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, total_codes * act_flat),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1, self.n_levels * self.n_embed, self.act_flat)


# ============================================================
# d3rlpy Integration
# ============================================================


@dataclasses.dataclass(frozen=True)
class VQBeTModules(Modules):
    vq_encoder: VQEncoder
    vq_decoder: VQDecoder
    residual_vq: ResidualVQ
    obs_encoder: nn.Module
    bin_head: BinPredictionHead
    offset_head: OffsetPredictionHead
    vqvae_optim: "OptimizerWrapper"  # noqa: F821
    bin_optim: "OptimizerWrapper"  # noqa: F821
    offset_optim: "OptimizerWrapper"  # noqa: F821


class VQBeTImpl(QLearningAlgoImplBase):
    _modules: VQBeTModules

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: VQBeTModules,
        vqvae_steps: int,
        n_steps: int,
        action_chunk_size: int,
        action_horizon: int,
        original_action_dim: int,
        commitment_weight: float,
        focal_gamma: float,
        offset_loss_weight: float,
        primary_code_weight: float,
        secondary_code_weight: float,
        n_levels: int,
        n_embed: int,
        compiled: bool,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._vqvae_steps = vqvae_steps
        self._action_chunk_size = action_chunk_size
        self._action_horizon = action_horizon
        self._original_action_dim = original_action_dim
        self._commitment_weight = commitment_weight
        self._offset_loss_weight = offset_loss_weight
        self._primary_code_weight = primary_code_weight
        self._secondary_code_weight = secondary_code_weight
        self._n_levels = n_levels
        self._n_embed = n_embed
        self._focal_loss = FocalLoss(gamma=focal_gamma)

        # Phase 2 (joint) ends halfway through GPT training
        gpt_total = max(n_steps - vqvae_steps, 0) if n_steps > 0 else 0
        self._offset_only_start = vqvae_steps + gpt_total // 2 if gpt_total > 0 else -1

        self._action_cache: torch.Tensor | None = None
        self._cache_idx: torch.Tensor | None = None  # per-env cache index

        self._compute_vqvae_grad = (
            CudaGraphWrapper(self._vqvae_grad_impl) if compiled else self._vqvae_grad_impl
        )
        self._compute_joint_grad = (
            CudaGraphWrapper(self._joint_grad_impl) if compiled else self._joint_grad_impl
        )

    def _freeze_vqvae(self) -> None:
        for module in [
            self._modules.vq_encoder,
            self._modules.vq_decoder,
            self._modules.residual_vq,
        ]:
            for p in module.parameters():
                p.requires_grad = False

    def _get_actions(self, batch: TorchMiniBatch) -> torch.Tensor:
        return batch.actions.reshape(-1, self._action_chunk_size, self._original_action_dim)

    def _get_offset_for_codes(
        self, offsets: torch.Tensor, indices: torch.Tensor, B: int
    ) -> torch.Tensor:
        """Sum per-level offsets for given code indices."""
        batch_idx = torch.arange(B, device=offsets.device)
        total = torch.zeros(B, offsets.shape[-1], device=offsets.device)
        for lv in range(self._n_levels):
            code_idx = lv * self._n_embed + indices[:, lv]
            total = total + offsets[batch_idx, code_idx]
        return total

    # ---- Phase 1: VQ-VAE pretraining ----

    def _vqvae_grad_impl(self, batch: TorchMiniBatch) -> tuple[torch.Tensor, torch.Tensor]:
        self._modules.vqvae_optim.zero_grad()
        actions = self._get_actions(batch)
        z = self._modules.vq_encoder(actions)
        quantized, _, commitment_loss = self._modules.residual_vq.quantize(z)
        recon = self._modules.vq_decoder(quantized)
        recon_loss = F.l1_loss(recon, actions)
        loss = recon_loss + self._commitment_weight * commitment_loss
        loss.backward()
        return recon_loss, commitment_loss

    def _vqvae_update(self, batch: TorchMiniBatch) -> dict[str, float]:
        recon_loss, commitment_loss = self._compute_vqvae_grad(batch)
        self._modules.vqvae_optim.step()
        loss = recon_loss + self._commitment_weight * commitment_loss
        return {
            "vqvae_recon_loss": recon_loss.item(),
            "commitment_loss": commitment_loss.item(),
            "loss": loss.item(),
        }

    # ---- Phase 2: Joint bin + offset training ----

    def _joint_grad_impl(self, batch: TorchMiniBatch) -> tuple[torch.Tensor, torch.Tensor]:
        self._modules.bin_optim.zero_grad()
        self._modules.offset_optim.zero_grad()
        actions = self._get_actions(batch)
        B = actions.shape[0]

        with torch.no_grad():
            z = self._modules.vq_encoder(actions)
            target_indices = self._modules.residual_vq.encode(z)

        obs_emb = self._modules.obs_encoder(batch.observations)
        logits = self._modules.bin_head(obs_emb)
        offsets = self._modules.offset_head(obs_emb)

        # Focal loss per level (primary=5x, secondary=0.5x, matching official)
        weights = [self._primary_code_weight] + [self._secondary_code_weight] * (self._n_levels - 1)
        code_loss = sum(
            w * self._focal_loss(logits[:, lv], target_indices[:, lv])
            for lv, w in enumerate(weights)
        )

        # Offset loss: decode target codes, add predicted offset, L1 vs ground truth
        with torch.no_grad():
            base_action = self._modules.vq_decoder(self._modules.residual_vq.lookup(target_indices))
        total_offset = self._get_offset_for_codes(offsets, target_indices, B)
        predicted = base_action + total_offset.reshape(
            B, self._action_chunk_size, self._original_action_dim
        )
        offset_loss = self._offset_loss_weight * F.l1_loss(predicted, actions)

        loss = code_loss + offset_loss
        loss.backward()
        return code_loss, offset_loss

    def _joint_update(self, batch: TorchMiniBatch) -> dict[str, float]:
        code_loss, offset_loss = self._compute_joint_grad(batch)
        self._modules.bin_optim.step()
        self._modules.offset_optim.step()
        loss = code_loss + offset_loss
        return {
            "code_loss": code_loss.item(),
            "offset_loss": offset_loss.item(),
            "loss": loss.item(),
        }

    # ---- Phase 3: Offset-only fine-tuning ----

    def _offset_update(self, batch: TorchMiniBatch) -> dict[str, float]:
        self._modules.offset_optim.zero_grad()
        actions = self._get_actions(batch)
        B = actions.shape[0]

        with torch.no_grad():
            z = self._modules.vq_encoder(actions)
            target_indices = self._modules.residual_vq.encode(z)
            obs_emb = self._modules.obs_encoder(batch.observations)
            base_action = self._modules.vq_decoder(self._modules.residual_vq.lookup(target_indices))

        # Only offset head gets gradients
        offsets = self._modules.offset_head(obs_emb)
        total_offset = self._get_offset_for_codes(offsets, target_indices, B)
        predicted = base_action + total_offset.reshape(
            B, self._action_chunk_size, self._original_action_dim
        )
        loss = self._offset_loss_weight * F.l1_loss(predicted, actions)
        loss.backward()
        self._modules.offset_optim.step()
        return {"offset_loss": loss.item(), "loss": loss.item()}

    # ---- Training dispatch ----

    def inner_update(self, batch: TorchMiniBatch, grad_step: int) -> dict[str, float]:
        if grad_step == self._vqvae_steps:
            self._freeze_vqvae()

        if grad_step < self._vqvae_steps:
            return self._vqvae_update(batch)
        elif self._offset_only_start > 0 and grad_step >= self._offset_only_start:
            return self._offset_update(batch)
        else:
            return self._joint_update(batch)

    # ---- Inference ----

    def _decode_with_offset(self, x: TorchObservation) -> torch.Tensor:
        obs_emb = self._modules.obs_encoder(x)
        logits = self._modules.bin_head(obs_emb)
        offsets = self._modules.offset_head(obs_emb)

        B = logits.shape[0]
        all_indices = []
        for lv in range(self._n_levels):
            probs = F.softmax(logits[:, lv], dim=-1)
            all_indices.append(torch.multinomial(probs, 1).squeeze(-1))
        indices = torch.stack(all_indices, dim=1)

        base_action = self._modules.vq_decoder(self._modules.residual_vq.lookup(indices))
        total_offset = self._get_offset_for_codes(offsets, indices, B)
        return base_action + total_offset.reshape(
            B, self._action_chunk_size, self._original_action_dim
        )

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        B = x.shape[0] if isinstance(x, torch.Tensor) else x[0].shape[0]

        if self._action_cache is not None and self._action_cache.shape[0] == B:
            valid = self._cache_idx < self._action_horizon
        else:
            valid = torch.zeros(B, dtype=torch.bool, device=self._device)

        if valid.all():
            action = self._action_cache[torch.arange(B, device=self._device), self._cache_idx]
            self._cache_idx += 1
            return action

        need = ~valid
        new_chunks = self._decode_with_offset(x)

        if self._action_cache is None or self._action_cache.shape[0] != B:
            self._action_cache = new_chunks
            self._cache_idx = torch.ones(B, dtype=torch.long, device=self._device)
            return self._action_cache[:, 0, :]

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
            self._cache_idx[done_mask] = self._action_horizon

    def set_action_horizon(self, action_horizon: int | None) -> int:
        old_action_horizon = self._action_horizon
        if action_horizon is None:
            action_horizon = self._action_chunk_size
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        self._action_horizon = min(self._action_chunk_size, int(action_horizon))
        self.reset_action_cache()
        return old_action_horizon

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def inner_predict_value(self, x: TorchObservation, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("VQ-BeT does not support value estimation")


@dataclasses.dataclass()
class VQBeTConfig(LearnableConfig):
    learning_rate: float = 5.5e-5
    vqvae_groups: int = 2
    vqvae_n_embed: int = 16
    vqvae_steps: int = 10000
    n_steps: int = 0  # total training steps, set by training script for phase scheduling
    latent_dim: int = 512
    action_chunk_size: int = 16
    action_horizon: int | None = None
    original_action_dim: int = 1
    hidden_dim: int = 256
    commitment_weight: float = 5.0
    focal_gamma: float = 2.0
    offset_loss_weight: float = 1.0
    primary_code_weight: float = 5.0
    secondary_code_weight: float = 0.5
    offset_learning_rate: float = 1e-3
    compile_graph: bool = False
    optim_factory: OptimizerFactory = make_optimizer_field()
    vqvae_optim_factory: OptimizerFactory | None = _make_optional_optimizer_field()
    offset_optim_factory: OptimizerFactory | None = _make_optional_optimizer_field()

    def create(self, device: DeviceArg = False, enable_ddp: bool = False) -> "VQBeT":
        return VQBeT(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "vq_bet"


class VQBeT(QLearningAlgoBase[VQBeTImpl, VQBeTConfig]):
    def inner_create_impl(self, observation_shape: Shape, action_size: int) -> None:
        cfg = self._config
        obs_dim = observation_shape[0]
        action_dim = cfg.original_action_dim
        act_flat = cfg.action_chunk_size * action_dim

        # VQ-VAE (MLP-based, matching official)
        vq_encoder = VQEncoder(act_flat, cfg.latent_dim).to(self._device)
        vq_decoder = VQDecoder(act_flat, cfg.latent_dim, cfg.action_chunk_size, action_dim).to(
            self._device
        )
        residual_vq = ResidualVQ(cfg.vqvae_groups, cfg.vqvae_n_embed, cfg.latent_dim).to(
            self._device
        )

        # Obs encoder (MLP)
        obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        ).to(self._device)

        # Prediction heads (matching official: 3-layer MLPs)
        bin_head = BinPredictionHead(cfg.hidden_dim, cfg.vqvae_groups, cfg.vqvae_n_embed).to(
            self._device
        )
        offset_head = OffsetPredictionHead(
            cfg.hidden_dim, cfg.vqvae_groups, cfg.vqvae_n_embed, act_flat
        ).to(self._device)

        # VQ-VAE optimizer
        vqvae_params = nn.ModuleDict(
            {"vq_encoder": vq_encoder, "vq_decoder": vq_decoder, "residual_vq": residual_vq}
        )
        vqvae_factory = cfg.vqvae_optim_factory or cfg.optim_factory
        vqvae_optim = vqvae_factory.create(
            vqvae_params.named_modules(), lr=cfg.learning_rate, compiled=self.compiled
        )

        # Bin optimizer (obs_encoder + bin_head, matching official optimizer1)
        bin_params = nn.ModuleDict({"obs_encoder": obs_encoder, "bin_head": bin_head})
        bin_optim = cfg.optim_factory.create(
            bin_params.named_modules(), lr=cfg.learning_rate, compiled=self.compiled
        )

        # Offset optimizer (offset_head only, matching official optimizer2 with higher lr)
        offset_params = nn.ModuleDict({"offset_head": offset_head})
        offset_factory = cfg.offset_optim_factory or cfg.optim_factory
        offset_optim = offset_factory.create(
            offset_params.named_modules(), lr=cfg.offset_learning_rate, compiled=self.compiled
        )

        modules = VQBeTModules(
            vq_encoder=vq_encoder,
            vq_decoder=vq_decoder,
            residual_vq=residual_vq,
            obs_encoder=obs_encoder,
            bin_head=bin_head,
            offset_head=offset_head,
            vqvae_optim=vqvae_optim,
            bin_optim=bin_optim,
            offset_optim=offset_optim,
        )

        self._impl = VQBeTImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            vqvae_steps=cfg.vqvae_steps,
            n_steps=cfg.n_steps,
            action_chunk_size=cfg.action_chunk_size,
            action_horizon=cfg.action_horizon or cfg.action_chunk_size,
            original_action_dim=action_dim,
            commitment_weight=cfg.commitment_weight,
            focal_gamma=cfg.focal_gamma,
            offset_loss_weight=cfg.offset_loss_weight,
            primary_code_weight=cfg.primary_code_weight,
            secondary_code_weight=cfg.secondary_code_weight,
            n_levels=cfg.vqvae_groups,
            n_embed=cfg.vqvae_n_embed,
            compiled=self.compiled,
            device=self._device,
        )

    def reset(self, done_mask: torch.Tensor | None = None) -> None:
        if self._impl is not None:
            self._impl.reset_action_cache(done_mask)

    def set_action_horizon(self, action_horizon: int | None) -> int:
        if self._impl is None:
            raise RuntimeError("VQBeT implementation has not been created")
        return self._impl.set_action_horizon(action_horizon)

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(VQBeTConfig)
