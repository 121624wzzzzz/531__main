"""AGD vocabulary-boundary variants for tied / partially untied LMs.

Paper mapping (input-side AGD):
  - s1 : standard weight tying (baseline)
  - s2 : full untying (independent E and U matrices)
  - s3 : shared matrix + learnable input bias correction (E_eff = E - beta)
  - s6 : shared matrix + hidden-dimensional low-rank deformation
  - s12: s3 + s6 (main AGD input-side variant in the paper)

Output-side analogues (s11 bias-only, s7 low-rank-only) are included for ablations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VariantSpec:
    name: str
    tie_word_embeddings: bool
    input_bias: bool
    input_hidden_lr: bool
    output_bias: bool
    output_hidden_lr: bool
    description: str


VARIANTS: dict[str, VariantSpec] = {
    "s1": VariantSpec(
        "s1", True, False, False, False, False,
        "Tied baseline: shared E=U matrix.",
    ),
    "s2": VariantSpec(
        "s2", False, False, False, False, False,
        "Full untying: independent input embeddings and lm_head.",
    ),
    "s3": VariantSpec(
        "s3", True, True, False, False, False,
        "Input bias correction only: E_eff(token) = E(token) - beta.",
    ),
    "s6": VariantSpec(
        "s6", True, False, True, False, False,
        "Hidden low-rank deformation on input lookup vectors.",
    ),
    "s12": VariantSpec(
        "s12", True, True, True, False, False,
        "Main AGD input-side: bias correction + hidden low-rank deformation.",
    ),
    "s7": VariantSpec(
        "s7", True, False, False, False, True,
        "Output-side hidden low-rank deformation (ablation).",
    ),
    "s11": VariantSpec(
        "s11", True, False, False, True, False,
        "Output-side bias correction (ablation).",
    ),
}


def _zero_init(module: nn.Module) -> None:
    if hasattr(module, "weight"):
        nn.init.zeros_(module.weight)


class AGDEmbeddingHead(nn.Module):
    """Shared / untied embedding table plus AGD structured freedom."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        variant: str = "s1",
        rank: int = 32,
        tie_word_embeddings: bool = True,
        lm_head_bias: bool = False,
    ):
        super().__init__()
        variant = variant.lower()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}. Choose from {sorted(VARIANTS)}.")
        self.spec = VARIANTS[variant]
        self.variant = variant
        self.rank = rank
        self.tie_word_embeddings = tie_word_embeddings if variant != "s2" else False

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=lm_head_bias)
        if self.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        spec = self.spec
        if spec.input_bias:
            self.embedding_beta = nn.Parameter(torch.zeros(hidden_size))
        if spec.input_hidden_lr:
            self.embedding_mul_a = nn.Linear(hidden_size, rank, bias=False)
            self.embedding_mul_b = nn.Linear(rank, hidden_size, bias=False)
            nn.init.normal_(self.embedding_mul_a.weight, mean=0.0, std=0.02)
            _zero_init(self.embedding_mul_b)
        if spec.output_bias:
            self.output_beta = nn.Parameter(torch.zeros(hidden_size))
        if spec.output_hidden_lr:
            self.output_mul_a = nn.Linear(hidden_size, rank, bias=False)
            self.output_mul_b = nn.Linear(rank, hidden_size, bias=False)
            nn.init.normal_(self.output_mul_a.weight, mean=0.0, std=0.02)
            _zero_init(self.output_mul_b)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        spec = self.spec
        if spec.input_bias and spec.input_hidden_lr:
            return x - self.embedding_beta + self.embedding_mul_b(self.embedding_mul_a(x))
        if spec.input_bias:
            return x - self.embedding_beta
        if spec.input_hidden_lr:
            return x + self.embedding_mul_b(self.embedding_mul_a(x))
        return x

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        spec = self.spec
        if spec.output_hidden_lr:
            hidden_states = hidden_states + self.output_mul_b(self.output_mul_a(hidden_states))
        if spec.output_bias:
            hidden_states = hidden_states - self.output_beta
        return self.lm_head(hidden_states)

    def extra_parameter_count(self) -> int:
        """Trainable parameters beyond the shared V x d matrix."""
        total = 0
        if self.spec.input_bias:
            total += self.embedding_beta.numel()
        if self.spec.input_hidden_lr:
            total += sum(p.numel() for p in self.embedding_mul_a.parameters())
            total += sum(p.numel() for p in self.embedding_mul_b.parameters())
        if self.spec.output_bias:
            total += self.output_beta.numel()
        if self.spec.output_hidden_lr:
            total += sum(p.numel() for p in self.output_mul_a.parameters())
            total += sum(p.numel() for p in self.output_mul_b.parameters())
        if not self.tie_word_embeddings:
            total += self.lm_head.weight.numel()
        return total

    @staticmethod
    def full_untie_parameter_count(vocab_size: int, hidden_size: int) -> int:
        return vocab_size * hidden_size
