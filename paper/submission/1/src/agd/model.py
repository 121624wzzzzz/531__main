"""Minimal causal LM backbone used to demonstrate AGD at the vocabulary boundary."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AGDConfig
from .embedding_head import AGDEmbeddingHead, VARIANTS


class _TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=attn_mask)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class AGDCausalLM(nn.Module):
    """Small decoder-only LM with AGD variants on the embedding head."""

    def __init__(self, config: AGDConfig | None = None):
        super().__init__()
        cfg = config or AGDConfig()
        self.config = cfg
        spec = VARIANTS[cfg.variant]
        tie = spec.tie_word_embeddings if cfg.variant != "s2" else False

        self.vocab_head = AGDEmbeddingHead(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            variant=cfg.variant,
            rank=cfg.variant_rank,
            tie_word_embeddings=tie,
            lm_head_bias=cfg.lm_head_bias,
        )
        self.pos_emb = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.drop = nn.Dropout(0.1)
        self.layers = nn.ModuleList(
            [
                _TransformerBlock(cfg.hidden_size, cfg.num_attention_heads, cfg.intermediate_size)
                for _ in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, seqlen = input_ids.shape
        positions = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        hidden = self.vocab_head.embed(input_ids) + self.pos_emb(positions)
        hidden = self.drop(hidden)

        causal = torch.triu(
            torch.full((seqlen, seqlen), float("-inf"), device=input_ids.device),
            diagonal=1,
        )
        for layer in self.layers:
            hidden = layer(hidden, attn_mask=causal)

        hidden = self.norm(hidden)
        logits = self.vocab_head.logits(hidden)

        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            out["loss"] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return out

    @torch.no_grad()
    def parameter_budget(self) -> dict[str, float]:
        shared = self.config.vocab_size * self.config.hidden_size
        extra = self.vocab_head.extra_parameter_count()
        full_untie = AGDEmbeddingHead.full_untie_parameter_count(
            self.config.vocab_size, self.config.hidden_size
        )
        return {
            "shared_vocab_matrix": shared,
            "agd_extra_params": extra,
            "full_untie_overhead": full_untie,
            "agd_overhead_ratio_vs_full_untie": extra / max(full_untie, 1),
        }
