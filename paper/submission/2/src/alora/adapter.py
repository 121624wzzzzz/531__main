"""Core A-LoRA adapter implementation.

Effective vocabulary matrix update (minimal form):

    M_eff ≈ M_0 (I + α/r · P Q) + 1_V β^T

Trainable parameter count scales with hidden dimension d and rank r,
not vocabulary size V.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn


@dataclass
class ALoraConfig:
    hidden_size: int
    rank: int = 16
    alpha: float = 32.0
    bias_scale: float = 1.0
    dropout: float = 0.0
    use_input: bool = True
    use_lm_head: bool = False
    use_input_bias: bool = True
    use_lm_head_bias: bool = False
    tie_input_lm_head_adapters: bool = False
    use_tied_lm_head_transpose: bool = False

    @property
    def scale(self) -> float:
        return self.alpha / max(1, self.rank)


class LowRankAffineMap(nn.Module):
    """Row-wise map: x -> x + (alpha/r) U(D(x)) + bias_scale * bias."""

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        alpha: float,
        dropout: float,
        use_bias: bool,
        bias_scale: float = 1.0,
    ):
        super().__init__()
        self.scale = alpha / max(1, rank)
        self.bias_scale = bias_scale
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if use_bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        x = x.to(dtype=self.down.weight.dtype)
        out = x + self.up(self.down(self.dropout(x))) * self.scale
        if self.bias is not None:
            out = out + self.bias_scale * self.bias.to(dtype=out.dtype)
        return out.to(dtype=out_dtype)


class AffineEmbedding(nn.Module):
    def __init__(
        self,
        base_embedding: nn.Embedding,
        cfg: ALoraConfig,
        affine: LowRankAffineMap | None = None,
    ):
        super().__init__()
        self.base_embedding = base_embedding
        self.affine = affine or LowRankAffineMap(
            cfg.hidden_size,
            cfg.rank,
            cfg.alpha,
            cfg.dropout,
            cfg.use_input_bias,
            bias_scale=cfg.bias_scale,
        )
        self.affine.to(device=base_embedding.weight.device)
        self.base_embedding.weight.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.affine(self.base_embedding(input_ids))


class AffineLMHead(nn.Module):
    def __init__(
        self,
        base_head: nn.Linear,
        cfg: ALoraConfig,
        affine: LowRankAffineMap | None = None,
    ):
        super().__init__()
        self.base_head = base_head
        self.affine = affine or LowRankAffineMap(
            cfg.hidden_size,
            cfg.rank,
            cfg.alpha,
            cfg.dropout,
            cfg.use_lm_head_bias,
            bias_scale=cfg.bias_scale,
        )
        self.affine.to(device=base_head.weight.device)
        self.base_head.weight.requires_grad_(False)
        if self.base_head.bias is not None:
            self.base_head.bias.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.affine(hidden_states).to(dtype=self.base_head.weight.dtype)
        return self.base_head(hidden_states)


class TiedTransposeAffineLMHead(nn.Module):
    """Mergeable tied output path matching merged input-side affine."""

    def __init__(self, base_head: nn.Linear, cfg: ALoraConfig, affine: LowRankAffineMap):
        super().__init__()
        self.base_head = base_head
        self.affine = affine
        self.base_head.weight.requires_grad_(False)
        if self.base_head.bias is not None:
            self.base_head.bias.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out_dtype = hidden_states.dtype
        x = hidden_states.to(dtype=self.affine.down.weight.dtype)
        low_rank = F.linear(F.linear(x, self.affine.up.weight.T), self.affine.down.weight.T)
        projected = (x + self.affine.scale * low_rank).to(dtype=self.base_head.weight.dtype)
        logits = self.base_head(projected)
        if self.affine.bias is not None:
            common_shift = (x * (self.affine.bias_scale * self.affine.bias.to(dtype=x.dtype))).sum(
                dim=-1, keepdim=True
            )
            logits = logits + common_shift.to(dtype=logits.dtype)
        return logits.to(dtype=out_dtype)


def _infer_hidden_size(model: nn.Module) -> int:
    return int(model.get_input_embeddings().embedding_dim)


def _freeze_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def _weights_are_tied(model: nn.Module) -> bool:
    emb = model.get_input_embeddings()
    head = getattr(model, "lm_head", None)
    return (
        emb is not None
        and head is not None
        and hasattr(emb, "weight")
        and hasattr(head, "weight")
        and emb.weight.data_ptr() == head.weight.data_ptr()
    )


def apply_alora_adapters(
    model: nn.Module,
    cfg: ALoraConfig | None = None,
    **kwargs: Any,
) -> nn.Module:
    if cfg is None:
        cfg = ALoraConfig(hidden_size=_infer_hidden_size(model), **kwargs)
    else:
        cfg.hidden_size = cfg.hidden_size or _infer_hidden_size(model)

    _freeze_model(model)
    model.alora_config = cfg
    shared: LowRankAffineMap | None = None

    if cfg.tie_input_lm_head_adapters:
        if not (cfg.use_input and cfg.use_lm_head):
            raise ValueError("tie_input_lm_head_adapters requires both input and lm_head.")
        if not _weights_are_tied(model):
            raise ValueError("tie_input_lm_head_adapters requires tied embed/lm_head weights.")
        shared = LowRankAffineMap(
            cfg.hidden_size,
            cfg.rank,
            cfg.alpha,
            cfg.dropout,
            cfg.use_input_bias,
            bias_scale=cfg.bias_scale,
        )

    if cfg.use_input:
        model.set_input_embeddings(AffineEmbedding(model.get_input_embeddings(), cfg, shared))

    if cfg.use_lm_head:
        head = getattr(model, "lm_head", None)
        if head is None or not isinstance(head, nn.Linear):
            raise TypeError("Expected model.lm_head to be nn.Linear.")
        if shared is not None and cfg.use_tied_lm_head_transpose:
            model.lm_head = TiedTransposeAffineLMHead(head, cfg, shared)
        else:
            model.lm_head = AffineLMHead(head, cfg, shared)

    return model


def _adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".affine." in name
    }


def save_alora_adapter(model: nn.Module, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = getattr(model, "alora_config", None)
    if cfg is None:
        raise ValueError("Model has no alora_config.")
    with (output / "alora_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    save_file(_adapter_state_dict(model), str(output / "alora_adapter.safetensors"))


def load_alora_adapter(model: nn.Module, adapter_dir: str | Path) -> nn.Module:
    adapter = Path(adapter_dir)
    with (adapter / "alora_config.json").open("r", encoding="utf-8") as f:
        cfg = ALoraConfig(**json.load(f))
    apply_alora_adapters(model, cfg)
    state = load_file(str(adapter / "alora_adapter.safetensors"), device="cpu")
    model.load_state_dict(state, strict=False)
    return model


def trainable_parameter_count(cfg: ALoraConfig, *, tied_vocab: bool = False) -> int:
    """Approximate trainable params for one adapter side."""
    per_side = cfg.rank * cfg.hidden_size * 2
    if cfg.use_input_bias or cfg.use_lm_head_bias:
        per_side += cfg.hidden_size
    count = 0
    if cfg.use_input:
        count += per_side
    if cfg.use_lm_head and not (cfg.tie_input_lm_head_adapters and tied_vocab):
        count += per_side
    if cfg.tie_input_lm_head_adapters and cfg.use_input and cfg.use_lm_head:
        count = per_side
    return count
