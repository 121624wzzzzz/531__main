"""S1-S21 embedding relaxation definitions and the EmbeddingVariant module.

Each variant is described by a single :class:`VariantSpec` — the one place
to edit when adding a new variant.  The spec drives parameter creation,
forward-pass logic, formula strings, and parameter-count reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

S_VARIANTS = {f"s{i}" for i in range(1, 22)}
LEGACY_VARIANTS = {"baseline", "wt", "pr", "wt_pr"}
VARIANT_CHOICES = sorted(S_VARIANTS | LEGACY_VARIANTS)

_LEGACY_ALIASES = {"wt": "s1", "wt_pr": "s1", "baseline": "s2", "pr": "s2"}


def normalize_embedding_variant(variant: str) -> str:
    """Map legacy names (``wt``, ``baseline``, …) to their S-numbered form."""
    return _LEGACY_ALIASES.get(variant, variant.lower())


# ---------------------------------------------------------------------------
# Declarative per-variant spec
# ---------------------------------------------------------------------------

# Component tokens that drive parameter creation and forward logic.
_INPUT_SHIFT = "input_shift"
_INPUT_LORA = "input_lora"
_INPUT_MULT = "input_mult"
_OUTPUT_LORA = "output_lora"
_OUTPUT_MULT = "output_mult"
_HIDDEN_SHIFT = "hidden_shift"

# Per-component parameter spec: list of (attr_name, shape_fn, init_style).
# shape_fn receives (vocab_size, hidden_size, rank) and returns a tuple.
_COMPONENT_PARAMS = {
    _INPUT_SHIFT:  [("input_shift",       lambda v, d, r: (d,),              "zero")],
    _HIDDEN_SHIFT: [("hidden_shift",      lambda v, d, r: (d,),              "zero")],
    _INPUT_LORA:   [("input_lora_a",      lambda v, d, r: (v, r),            "uniform"),
                    ("input_lora_b",      lambda v, d, r: (r, d),            "zero")],
    _OUTPUT_LORA:  [("output_lora_a",     lambda v, d, r: (v, r),            "uniform"),
                    ("output_lora_b",     lambda v, d, r: (r, d),            "zero")],
    _INPUT_MULT:   [("input_mult_p",      lambda v, d, r: (d, r),            "uniform"),
                    ("input_mult_q",      lambda v, d, r: (r, d),            "zero")],
    _OUTPUT_MULT:  [("output_mult_p",     lambda v, d, r: (d, r),            "uniform"),
                    ("output_mult_q",     lambda v, d, r: (r, d),            "zero")],
}


@dataclass(frozen=True)
class VariantSpec:
    """The single source of truth for one S variant.

    Adding a new variant means adding *one* :class:`VariantSpec` to
    ``_SPECS`` below — no other file needs to change.
    """

    name: str                        # human-readable label
    input_formula: str               # LaTeX-ish string for docs
    output_formula: str
    table_param_approx: str          # e.g. "2dr", "Vr + 2dr"
    # Which components are active on each side:
    extra_input: Tuple[str, ...] = ()   # e.g. ("input_shift", "input_mult")
    extra_output: Tuple[str, ...] = ()  # e.g. ("output_mult",)
    extra_hidden: Tuple[str, ...] = ()  # e.g. ("hidden_shift",)
    tied_output: bool = True            # False → separate U matrix

    # ---- derived helpers ----
    @property
    def uses_untied_output(self) -> bool:
        return not self.tied_output

    @property
    def uses_input_shift(self) -> bool:
        return _INPUT_SHIFT in self.extra_input

    @property
    def uses_input_lora(self) -> bool:
        return _INPUT_LORA in self.extra_input

    @property
    def uses_input_mult(self) -> bool:
        return _INPUT_MULT in self.extra_input

    @property
    def uses_output_lora(self) -> bool:
        return _OUTPUT_LORA in self.extra_output

    @property
    def uses_output_mult(self) -> bool:
        return _OUTPUT_MULT in self.extra_output

    @property
    def uses_hidden_shift(self) -> bool:
        return _HIDDEN_SHIFT in self.extra_hidden

    def extra_params(self, vocab_size: int, hidden_size: int, rank: int) -> int:
        """Compute the number of extra parameters for this variant."""
        total = 0
        if not self.tied_output:
            total += vocab_size * hidden_size
        for comp in self.extra_input + self.extra_output + self.extra_hidden:
            if comp in (_INPUT_LORA, _OUTPUT_LORA):
                total += vocab_size * rank + rank * hidden_size
            elif comp in (_INPUT_MULT, _OUTPUT_MULT):
                total += 2 * hidden_size * rank
            elif comp in (_INPUT_SHIFT, _HIDDEN_SHIFT):
                total += hidden_size
        return total


# The complete catalogue — add new variants here:
_SPECS: Dict[str, VariantSpec] = {
    "s1": VariantSpec("Tied", "W", "W", "0"),
    "s2": VariantSpec("Untied", "E", "U", "Vd", tied_output=False),
    "s3": VariantSpec("input_shift", "W - 1 beta^T", "W", "d",
                      extra_input=(_INPUT_SHIFT,)),
    "s4": VariantSpec("input_lora_add", "W + AB", "W", "Vr",
                      extra_input=(_INPUT_LORA,)),
    "s5": VariantSpec("output_lora_add", "W", "W + AB", "Vr",
                      extra_output=(_OUTPUT_LORA,)),
    "s6": VariantSpec("input_mult_low_rank", "W(I + PQ)", "W", "2dr",
                      extra_input=(_INPUT_MULT,)),
    "s7": VariantSpec("output_mult_low_rank", "W", "W(I + PQ)", "2dr",
                      extra_output=(_OUTPUT_MULT,)),
    "s8": VariantSpec("untied_input_shift", "E - 1 beta^T", "U", "d",
                      extra_input=(_INPUT_SHIFT,), tied_output=False),
    "s9": VariantSpec("untied_input_lora_add", "E + AB", "U", "Vr",
                      extra_input=(_INPUT_LORA,), tied_output=False),
    "s10": VariantSpec("untied_input_mult_low_rank", "E(I + PQ)", "U", "2dr",
                       extra_input=(_INPUT_MULT,), tied_output=False),
    "s11": VariantSpec("output_hidden_shift", "W", "W acting on h_t + beta", "d",
                       extra_hidden=(_HIDDEN_SHIFT,)),
    "s12": VariantSpec("input_shift_plus_mult", "W(I + PQ) - 1 beta^T", "W", "d + 2dr",
                       extra_input=(_INPUT_MULT, _INPUT_SHIFT)),
    "s13": VariantSpec("input_lora_add_plus_mult", "W + AB + WPQ", "W", "Vr + 2dr",
                       extra_input=(_INPUT_LORA, _INPUT_MULT)),
    "s14": VariantSpec("tied_bilateral_lora_add", "W + AB", "W + CD", "2Vr",
                       extra_input=(_INPUT_LORA,), extra_output=(_OUTPUT_LORA,)),
    "s15": VariantSpec("tied_bilateral_mult_low_rank", "W(I + PQ)", "W(I + RS)", "4dr",
                       extra_input=(_INPUT_MULT,), extra_output=(_OUTPUT_MULT,)),
    "s16": VariantSpec("tied_input_lora_output_mult", "W + AB", "W(I + PQ)", "Vr + 2dr",
                       extra_input=(_INPUT_LORA,), extra_output=(_OUTPUT_MULT,)),
    "s17": VariantSpec("tied_input_lora_mult_output_mult", "W + AB + WPQ", "W(I + RS)", "Vr + 4dr",
                       extra_input=(_INPUT_LORA, _INPUT_MULT), extra_output=(_OUTPUT_MULT,)),
    "s18": VariantSpec("tied_input_shift_mult_output_mult",
                       "W(I + PQ) - 1 beta^T", "W(I + RS)", "d + 4dr",
                       extra_input=(_INPUT_MULT, _INPUT_SHIFT), extra_output=(_OUTPUT_MULT,)),
    "s19": VariantSpec("output_mult_hidden_shift", "W", "W(I + PQ) acting on h_t + beta", "d + 2dr",
                       extra_output=(_OUTPUT_MULT,), extra_hidden=(_HIDDEN_SHIFT,)),
    "s20": VariantSpec("tied_bilateral_shift_mult",
                       "W(I + PQ) - 1 beta^T", "W(I + RS) acting on h_t + gamma", "2d + 4dr",
                       extra_input=(_INPUT_MULT, _INPUT_SHIFT),
                       extra_output=(_OUTPUT_MULT,), extra_hidden=(_HIDDEN_SHIFT,)),
    "s21": VariantSpec("output_lora_add_plus_mult", "W", "W + AB + WPQ", "Vr + 2dr",
                       extra_output=(_OUTPUT_LORA, _OUTPUT_MULT)),
}


def _get_spec(variant: str) -> VariantSpec:
    mode = normalize_embedding_variant(variant)
    if mode not in _SPECS:
        raise ValueError(f"Unknown embedding variant: {variant}")
    return _SPECS[mode]


# ---------------------------------------------------------------------------
# Public metadata helpers (compatibility shims)
# ---------------------------------------------------------------------------

def variant_formula(variant: str) -> Dict[str, str]:
    spec = _get_spec(variant)
    return {
        "variant": variant,
        "normalized_variant": normalize_embedding_variant(variant),
        "name": spec.name,
        "input_effective": spec.input_formula,
        "output_effective": spec.output_formula,
        "table_param_approx": spec.table_param_approx,
    }


def actual_extra_params(variant: str, vocab_size: int, hidden_size: int, rank: int) -> int:
    return _get_spec(variant).extra_params(vocab_size, hidden_size, rank)


# ---------------------------------------------------------------------------
# EmbeddingVariant nn.Module
# ---------------------------------------------------------------------------

class EmbeddingVariant(nn.Module):
    """Input/output embedding relaxation layer covering all S1-S21 variants.

    Parameter creation and forward logic are driven by the :class:`VariantSpec`
    — there are no hand-written per-variant if/elif chains here.
    """

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
        if rank < 1:
            raise ValueError("--lora_rank must be >= 1")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.variant = variant
        self.mode = normalize_embedding_variant(variant)
        self.rank = rank
        self.init_scale = init_scale
        self.relaxation_scale = relaxation_scale

        spec = _get_spec(variant)
        self._spec = spec
        # Expose properties so external code that checks e.g. `module.uses_output_mult` still works.
        self.uses_untied_output = spec.uses_untied_output
        self.uses_input_shift = spec.uses_input_shift
        self.uses_input_lora = spec.uses_input_lora
        self.uses_input_mult = spec.uses_input_mult
        self.uses_output_lora = spec.uses_output_lora
        self.uses_output_mult = spec.uses_output_mult

        # --- shared base ---
        self.input_weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self.output_weight = nn.Parameter(torch.empty(vocab_size, hidden_size)) if spec.uses_untied_output else None
        self.output_bias = nn.Parameter(torch.empty(vocab_size))

        # --- per-component parameters, driven by _COMPONENT_PARAMS ---
        for comp in spec.extra_input + spec.extra_output + spec.extra_hidden:
            for attr, shape_fn, _init in _COMPONENT_PARAMS[comp]:
                setattr(self, attr, nn.Parameter(torch.empty(*shape_fn(vocab_size, hidden_size, rank))))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.input_weight, -self.init_scale, self.init_scale)
        if self.output_weight is not None:
            nn.init.uniform_(self.output_weight, -self.init_scale, self.init_scale)
        nn.init.uniform_(self.output_bias, -self.init_scale, self.init_scale)

        for comp in self._spec.extra_input + self._spec.extra_output + self._spec.extra_hidden:
            for attr, _shape_fn, init_style in _COMPONENT_PARAMS[comp]:
                param = getattr(self, attr, None)
                if param is not None:
                    if init_style == "zero":
                        nn.init.zeros_(param)
                    else:
                        nn.init.uniform_(param, -self.init_scale, self.init_scale)

    # ------------------------------------------------------------------
    # Forward pass — driven by component lists, not per-variant branches
    # ------------------------------------------------------------------

    def input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = nn.functional.embedding(input_ids, self.input_weight)

        spec = self._spec
        s = self.relaxation_scale

        for comp in spec.extra_input:
            if comp == _INPUT_LORA:
                low_rank = nn.functional.embedding(input_ids, self.input_lora_a).matmul(self.input_lora_b)
                output = output + s * low_rank
            elif comp == _INPUT_MULT:
                delta = output.matmul(self.input_mult_p).matmul(self.input_mult_q)
                output = output + s * delta
            elif comp == _INPUT_SHIFT:
                output = output - s * self.input_shift

        return output

    def output_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        spec = self._spec
        s = self.relaxation_scale

        for comp in spec.extra_hidden:
            if comp == _HIDDEN_SHIFT:
                hidden = hidden + s * self.hidden_shift

        weight = self.output_weight if self.output_weight is not None else self.input_weight
        assert weight is not None

        for comp in spec.extra_output:
            if comp == _OUTPUT_LORA:
                weight = weight + s * self.output_lora_a.matmul(self.output_lora_b)
            elif comp == _OUTPUT_MULT:
                weight = weight + s * weight.matmul(self.output_mult_p).matmul(self.output_mult_q)

        return nn.functional.linear(hidden, weight, self.output_bias)


HiddenState = Tuple[torch.Tensor, torch.Tensor]
