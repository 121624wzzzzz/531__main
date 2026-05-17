"""S1-S13 embedding relaxation definitions and the EmbeddingVariant module.

This module owns three layers of knowledge about variants:

1. Symbolic vocabulary: which short names exist and how legacy aliases like
   ``baseline`` / ``wt`` map onto the S-numbered variants.
2. Static metadata: the human-readable formula and approximate parameter
   count for each variant, used in JSON metrics and experiment tables.
3. The actual :class:`EmbeddingVariant` ``nn.Module`` that materialises the
   relaxation on top of a shared ``W`` matrix or untied ``E`` / ``U`` matrices.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

S_VARIANTS = {f"s{i}" for i in range(1, 14)}
LEGACY_VARIANTS = {"baseline", "wt", "pr", "wt_pr"}
VARIANT_CHOICES = sorted(S_VARIANTS | LEGACY_VARIANTS)


def normalize_embedding_variant(variant: str) -> str:
    aliases = {
        "wt": "s1",
        "wt_pr": "s1",
        "baseline": "s2",
        "pr": "s2",
    }
    return aliases.get(variant, variant.lower())


def variant_formula(variant: str) -> Dict[str, str]:
    formulas = {
        "s1": ("Tied", "W", "W", "0"),
        "s2": ("Untied", "E", "U", "Vd"),
        "s3": ("input_shift", "W - 1 beta^T", "W", "d"),
        "s4": ("input_lora_add", "W + AB", "W", "Vr"),
        "s5": ("output_lora_add", "W", "W + AB", "Vr"),
        "s6": ("input_mult_low_rank", "W(I + PQ)", "W", "2dr"),
        "s7": ("output_mult_low_rank", "W", "W(I + PQ)", "2dr"),
        "s8": ("untied_input_shift", "E - 1 beta^T", "U", "d"),
        "s9": ("untied_input_lora_add", "E + AB", "U", "Vr"),
        "s10": ("untied_input_mult_low_rank", "E(I + PQ)", "U", "2dr"),
        "s11": ("output_hidden_shift", "W", "W acting on h_t + beta", "d"),
        "s12": ("input_shift_plus_mult", "W(I + PQ) - 1 beta^T", "W", "d + 2dr"),
        "s13": ("input_lora_add_plus_mult", "W + AB + WPQ", "W", "Vr + 2dr"),
    }
    mode = normalize_embedding_variant(variant)
    name, input_eff, output_eff, table_params = formulas[mode]
    return {
        "variant": variant,
        "normalized_variant": mode,
        "name": name,
        "input_effective": input_eff,
        "output_effective": output_eff,
        "table_param_approx": table_params,
    }


def actual_extra_params(variant: str, vocab_size: int, hidden_size: int, rank: int) -> int:
    mode = normalize_embedding_variant(variant)
    vd = vocab_size * hidden_size
    vr_rd = vocab_size * rank + rank * hidden_size
    dr2 = 2 * hidden_size * rank
    counts = {
        "s1": 0,
        "s2": vd,
        "s3": hidden_size,
        "s4": vr_rd,
        "s5": vr_rd,
        "s6": dr2,
        "s7": dr2,
        "s8": vd + hidden_size,
        "s9": vd + vr_rd,
        "s10": vd + dr2,
        "s11": hidden_size,
        "s12": hidden_size + dr2,
        "s13": vr_rd + dr2,
    }
    return counts[mode]


class EmbeddingVariant(nn.Module):
    """Input/output embedding relaxation layer for S1-S13."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        variant: str,
        rank: int,
        init_scale: float,
        relaxation_scale: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.variant = variant
        self.mode = normalize_embedding_variant(variant)
        self.rank = rank
        self.init_scale = init_scale
        self.relaxation_scale = relaxation_scale

        if self.mode not in S_VARIANTS:
            raise ValueError(f"Unknown embedding variant: {variant}")
        if rank < 1:
            raise ValueError("--lora_rank must be >= 1")

        self.input_weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self.output_weight = (
            nn.Parameter(torch.empty(vocab_size, hidden_size)) if self.uses_untied_output else None
        )
        self.output_bias = nn.Parameter(torch.empty(vocab_size))

        self.input_shift = nn.Parameter(torch.empty(hidden_size)) if self.uses_input_shift else None
        self.input_lora_a = nn.Parameter(torch.empty(vocab_size, rank)) if self.uses_input_lora else None
        self.input_lora_b = nn.Parameter(torch.empty(rank, hidden_size)) if self.uses_input_lora else None
        self.input_mult_p = nn.Parameter(torch.empty(hidden_size, rank)) if self.uses_input_mult else None
        self.input_mult_q = nn.Parameter(torch.empty(rank, hidden_size)) if self.uses_input_mult else None

        self.output_lora_a = nn.Parameter(torch.empty(vocab_size, rank)) if self.uses_output_lora else None
        self.output_lora_b = nn.Parameter(torch.empty(rank, hidden_size)) if self.uses_output_lora else None
        self.output_mult_p = nn.Parameter(torch.empty(hidden_size, rank)) if self.uses_output_mult else None
        self.output_mult_q = nn.Parameter(torch.empty(rank, hidden_size)) if self.uses_output_mult else None
        self.hidden_shift = nn.Parameter(torch.empty(hidden_size)) if self.mode == "s11" else None

        self.reset_parameters()

    @property
    def uses_untied_output(self) -> bool:
        return self.mode in {"s2", "s8", "s9", "s10"}

    @property
    def uses_input_shift(self) -> bool:
        return self.mode in {"s3", "s8", "s12"}

    @property
    def uses_input_lora(self) -> bool:
        return self.mode in {"s4", "s9", "s13"}

    @property
    def uses_input_mult(self) -> bool:
        return self.mode in {"s6", "s10", "s12", "s13"}

    @property
    def uses_output_lora(self) -> bool:
        return self.mode == "s5"

    @property
    def uses_output_mult(self) -> bool:
        return self.mode == "s7"

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.input_weight, -self.init_scale, self.init_scale)
        if self.output_weight is not None:
            nn.init.uniform_(self.output_weight, -self.init_scale, self.init_scale)
        nn.init.uniform_(self.output_bias, -self.init_scale, self.init_scale)

        for shift in (self.input_shift, self.hidden_shift):
            if shift is not None:
                nn.init.zeros_(shift)

        for matrix in (self.input_lora_a, self.output_lora_a, self.input_mult_p, self.output_mult_p):
            if matrix is not None:
                nn.init.uniform_(matrix, -self.init_scale, self.init_scale)
        for matrix in (self.input_lora_b, self.output_lora_b, self.input_mult_q, self.output_mult_q):
            if matrix is not None:
                nn.init.zeros_(matrix)

    def input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        base = nn.functional.embedding(input_ids, self.input_weight)
        output = base

        if self.input_lora_a is not None and self.input_lora_b is not None:
            low_rank = nn.functional.embedding(input_ids, self.input_lora_a).matmul(self.input_lora_b)
            output = output + self.relaxation_scale * low_rank

        if self.input_mult_p is not None and self.input_mult_q is not None:
            output = output + self.relaxation_scale * base.matmul(self.input_mult_p).matmul(self.input_mult_q)

        if self.input_shift is not None:
            output = output - self.relaxation_scale * self.input_shift

        return output

    def output_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.hidden_shift is not None:
            hidden = hidden + self.relaxation_scale * self.hidden_shift

        weight = self.output_weight if self.output_weight is not None else self.input_weight
        assert weight is not None

        if self.output_lora_a is not None and self.output_lora_b is not None:
            weight = weight + self.relaxation_scale * self.output_lora_a.matmul(self.output_lora_b)

        if self.output_mult_p is not None and self.output_mult_q is not None:
            weight = weight + self.relaxation_scale * weight.matmul(self.output_mult_p).matmul(self.output_mult_q)

        return nn.functional.linear(hidden, weight, self.output_bias)


HiddenState = Tuple[torch.Tensor, torch.Tensor]
