"""ACT (Action Chunking with Transformers) for offline RL.

Follows the official implementation: github.com/tonyzhaozh/act
and LeRobot: github.com/huggingface/lerobot

CVAE architecture: during training, a VAE encoder processes [CLS, state, actions]
to produce latent z. A DETR-style transformer encoder-decoder predicts the action
chunk from [z, state]. During inference, z is set to zero (prior mean).
Loss = L1(actions) + kl_weight * KL(q(z) || N(0,I)).
Continuous action space only.
"""

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from d3rlpy.algos.qlearning.base import QLearningAlgoBase, QLearningAlgoImplBase
from d3rlpy.base import DeviceArg, LearnableConfig, register_learnable
from d3rlpy.constants import ActionSpace
from d3rlpy.optimizers import OptimizerFactory, make_optimizer_field
from d3rlpy.torch_utility import Modules, TorchMiniBatch
from d3rlpy.types import Shape, TorchObservation

__all__ = ["ACTConfig", "ACT"]


# ============================================================
# Positional Encoding
# ============================================================


class SinusoidalPositionEncoding(nn.Module):
    """1D sinusoidal position encoding (for VAE encoder)."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))  # (max_len, 1, d_model)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:seq_len]  # (seq_len, 1, d_model)


# ============================================================
# DETR-style Transformer Layers (pos embed added to Q/K only)
# ============================================================


class ACTEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        q = k = x + pos
        out, _ = self.self_attn(q, k, x)
        x = self.norm1(x + self.dropout1(out))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self, tgt: torch.Tensor, memory: torch.Tensor, tgt_pos: torch.Tensor, mem_pos: torch.Tensor
    ) -> torch.Tensor:
        q = k = tgt + tgt_pos
        out, _ = self.self_attn(q, k, tgt)
        tgt = self.norm1(tgt + self.dropout1(out))
        out, _ = self.cross_attn(tgt + tgt_pos, memory + mem_pos, memory)
        tgt = self.norm2(tgt + self.dropout2(out))
        tgt = self.norm3(tgt + self.dropout3(self.ffn(tgt)))
        return tgt


# ============================================================
# ACT Model
# ============================================================


class ACTModel(nn.Module):
    """Complete ACT model: CVAE + DETR-style transformer encoder-decoder."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        chunk_size: int,
        dim_model: int = 256,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 1,
        n_vae_encoder_layers: int = 4,
        latent_dim: int = 32,
        dropout: float = 0.1,
        use_vae: bool = True,
    ):
        super().__init__()
        self.use_vae = use_vae
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size

        # Projections
        self.state_proj = nn.Linear(obs_dim, dim_model)
        self.latent_proj = nn.Linear(latent_dim, dim_model)
        self.action_head = nn.Linear(dim_model, action_dim)

        # Learned positional embeddings
        self.encoder_pos_embed = nn.Embedding(2, dim_model)  # [z_token, state_token]
        self.decoder_pos_embed = nn.Embedding(chunk_size, dim_model)  # action queries

        # Main transformer encoder
        self.encoder_layers = nn.ModuleList(
            [
                ACTEncoderLayer(dim_model, n_heads, dim_feedforward, dropout)
                for _ in range(n_encoder_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(dim_model)

        # Transformer decoder
        self.decoder_layers = nn.ModuleList(
            [
                ACTDecoderLayer(dim_model, n_heads, dim_feedforward, dropout)
                for _ in range(n_decoder_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(dim_model)

        # VAE components (training only)
        if use_vae:
            self.cls_embed = nn.Embedding(1, dim_model)
            self.action_proj = nn.Linear(action_dim, dim_model)
            self.vae_pos_enc = SinusoidalPositionEncoding(dim_model)
            self.vae_encoder_layers = nn.ModuleList(
                [
                    ACTEncoderLayer(dim_model, n_heads, dim_feedforward, dropout)
                    for _ in range(n_vae_encoder_layers)
                ]
            )
            self.vae_encoder_norm = nn.LayerNorm(dim_model)
            self.latent_head = nn.Linear(dim_model, latent_dim * 2)  # mu + log_sigma_x2

        # Xavier init (matching official)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_vae(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """VAE encoder: [CLS, state, action_seq] → (z, kl_loss)."""
        B = obs.shape[0]
        cls = self.cls_embed.weight.unsqueeze(1).expand(-1, B, -1)  # (1, B, d)
        state = self.state_proj(obs).unsqueeze(0)  # (1, B, d)
        act_tokens = self.action_proj(actions).transpose(0, 1)  # (K, B, d)
        tokens = torch.cat([cls, state, act_tokens], dim=0)  # (2+K, B, d)

        pos = self.vae_pos_enc(tokens.shape[0]).expand(-1, B, -1)
        for layer in self.vae_encoder_layers:
            tokens = layer(tokens, pos)
        tokens = self.vae_encoder_norm(tokens)

        params = self.latent_head(tokens[0])  # CLS output → (B, latent_dim*2)
        mu, log_sigma_x2 = params.chunk(2, dim=-1)
        z = mu + torch.exp(log_sigma_x2 / 2) * torch.randn_like(mu)
        kl = -0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp()).sum(-1).mean()
        return z, kl

    def decode(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Transformer encoder-decoder: [z, state] → action chunk (B, K, action_dim)."""
        B = obs.shape[0]
        z_tok = self.latent_proj(z).unsqueeze(0)  # (1, B, d)
        s_tok = self.state_proj(obs).unsqueeze(0)  # (1, B, d)
        enc_in = torch.cat([z_tok, s_tok], dim=0)  # (2, B, d)
        enc_pos = self.encoder_pos_embed.weight.unsqueeze(1).expand(-1, B, -1)

        for layer in self.encoder_layers:
            enc_in = layer(enc_in, enc_pos)
        memory = self.encoder_norm(enc_in)

        tgt = torch.zeros(self.chunk_size, B, memory.shape[-1], device=memory.device)
        dec_pos = self.decoder_pos_embed.weight.unsqueeze(1).expand(-1, B, -1)
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, dec_pos, enc_pos)
        tgt = self.decoder_norm(tgt)

        return self.action_head(tgt).transpose(0, 1)  # (B, K, action_dim)


# ============================================================
# d3rlpy Integration
# ============================================================


@dataclasses.dataclass(frozen=True)
class ACTModules(Modules):
    model: ACTModel
    optim: "OptimizerWrapper"  # noqa: F821


class ACTImpl(QLearningAlgoImplBase):
    _modules: ACTModules

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: ACTModules,
        action_chunk_size: int,
        action_horizon: int,
        original_action_dim: int,
        kl_weight: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._action_chunk_size = action_chunk_size
        self._action_horizon = action_horizon
        self._original_action_dim = original_action_dim
        self._kl_weight = kl_weight

        self._action_cache: torch.Tensor | None = None
        self._cache_idx: torch.Tensor | None = None  # per-env cache index

    def inner_update(self, batch: TorchMiniBatch, grad_step: int) -> dict[str, float]:
        self._modules.optim.zero_grad()
        actions = batch.actions.reshape(-1, self._action_chunk_size, self._original_action_dim)
        z, kl_loss = self._modules.model.encode_vae(batch.observations, actions)
        pred = self._modules.model.decode(batch.observations, z)
        l1_loss = F.l1_loss(pred, actions)
        loss = l1_loss + self._kl_weight * kl_loss
        loss.backward()
        self._modules.optim.step()
        return {
            "l1_loss": l1_loss.item(),
            "kl_loss": kl_loss.item(),
            "loss": loss.item(),
        }

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

        # Inference: z = zeros (prior mean), no VAE encoder
        z = torch.zeros(B, self._modules.model.latent_dim, device=self._device)
        new_chunks = self._modules.model.decode(x, z)

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
        raise NotImplementedError("ACT does not support value estimation")


@dataclasses.dataclass()
class ACTConfig(LearnableConfig):
    learning_rate: float = 1e-4
    dim_model: int = 256
    n_heads: int = 8
    dim_feedforward: int = 2048
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    n_vae_encoder_layers: int = 4
    latent_dim: int = 32
    dropout: float = 0.1
    kl_weight: float = 10.0
    use_vae: bool = True
    action_chunk_size: int = 16
    action_horizon: int | None = None
    original_action_dim: int = 1
    compile_graph: bool = False
    optim_factory: OptimizerFactory = make_optimizer_field()

    def create(self, device: DeviceArg = False, enable_ddp: bool = False) -> "ACT":
        return ACT(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "act"


class ACT(QLearningAlgoBase[ACTImpl, ACTConfig]):
    def inner_create_impl(self, observation_shape: Shape, action_size: int) -> None:
        cfg = self._config
        obs_dim = observation_shape[0]

        model = ACTModel(
            obs_dim=obs_dim,
            action_dim=cfg.original_action_dim,
            chunk_size=cfg.action_chunk_size,
            dim_model=cfg.dim_model,
            n_heads=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            n_encoder_layers=cfg.n_encoder_layers,
            n_decoder_layers=cfg.n_decoder_layers,
            n_vae_encoder_layers=cfg.n_vae_encoder_layers,
            latent_dim=cfg.latent_dim,
            dropout=cfg.dropout,
            use_vae=cfg.use_vae,
        ).to(self._device)

        if cfg.compile_graph:
            model.encode_vae = torch.compile(model.encode_vae)
            model.decode = torch.compile(model.decode)

        params = nn.ModuleDict({"model": model})
        optim = cfg.optim_factory.create(
            params.named_modules(), lr=cfg.learning_rate, compiled=self.compiled
        )

        self._impl = ACTImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=ACTModules(model=model, optim=optim),
            action_chunk_size=cfg.action_chunk_size,
            action_horizon=cfg.action_horizon or cfg.action_chunk_size,
            original_action_dim=cfg.original_action_dim,
            kl_weight=cfg.kl_weight,
            device=self._device,
        )

    def reset(self, done_mask: torch.Tensor | None = None) -> None:
        if self._impl is not None:
            self._impl.reset_action_cache(done_mask)

    def set_action_horizon(self, action_horizon: int | None) -> int:
        if self._impl is None:
            raise RuntimeError("ACT implementation has not been created")
        return self._impl.set_action_horizon(action_horizon)

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(ACTConfig)
