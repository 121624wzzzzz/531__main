"""LSTM language models for the PTB experiments.

Two architectures are exposed via :func:`build_model`:

- :class:`StandardLSTMModel`: Zaremba-style stacked LSTM that drives the
  paper's ``small`` / ``large`` rows and the local 4090 screening rows.
- :class:`VariationalDropoutLSTMModel`: Gal / Bayesian-dropout variant used
  to reproduce the BayesianRNN rows.

Both share the :class:`~variants.EmbeddingVariant` for input/output
embedding relaxation and accept the same ``(variant, rank, relaxation_scale)``
tuple, so swapping architectures does not change the variant API.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn

from configs import PTBConfig
from variants import VARIANT_CHOICES, EmbeddingVariant, HiddenState


class StandardLSTMModel(nn.Module):
    """Zaremba-style stacked LSTM used by the TensorFlow PTB experiments."""

    def __init__(self, config: PTBConfig, variant: str, rank: int, relaxation_scale: float):
        super().__init__()
        self.config = config
        self.variant = variant
        self.drop = nn.Dropout(1.0 - config.keep_prob)
        self.embeddings = EmbeddingVariant(
            config.vocab_size,
            config.hidden_size,
            variant,
            rank,
            config.init_scale,
            relaxation_scale,
        )
        self.lstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            config.num_layers,
            batch_first=True,
            dropout=1.0 - config.keep_prob if config.num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(config.hidden_size, config.hidden_size) if variant in {"pr", "wt_pr"} else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for param in self.parameters():
            nn.init.uniform_(param, -self.config.init_scale, self.config.init_scale)
        self.embeddings.reset_parameters()

    def forward(self, input_ids: torch.Tensor, hidden: Optional[HiddenState]) -> Tuple[torch.Tensor, HiddenState]:
        emb = self.drop(self.embeddings.input_embeddings(input_ids))
        output, hidden = self.lstm(emb, hidden)
        output = self.drop(output)
        if self.proj is not None:
            output = self.proj(output)
        return self.embeddings.output_logits(output), hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> HiddenState:
        weight = next(self.parameters())
        shape = (self.config.num_layers, batch_size, self.config.hidden_size)
        return weight.new_zeros(shape, device=device), weight.new_zeros(shape, device=device)


class VariationalLSTMCell(nn.Module):
    """LSTM cell with per-sequence gate masks matching the old Torch7 setup."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))

    def forward(
        self,
        inputs: torch.Tensor,
        state: HiddenState,
        input_gate_mask: torch.Tensor,
        hidden_gate_mask: torch.Tensor,
    ) -> HiddenState:
        h_prev, c_prev = state
        size = self.hidden_size

        x = inputs.unsqueeze(1) * input_gate_mask
        h = h_prev.unsqueeze(1) * hidden_gate_mask
        weight_ih = self.weight_ih.view(4, size, size)
        weight_hh = self.weight_hh.view(4, size, size)
        gates = torch.bmm(x.transpose(0, 1), weight_ih.transpose(1, 2))
        gates = gates + torch.bmm(h.transpose(0, 1), weight_hh.transpose(1, 2))
        gates = gates.transpose(0, 1)
        gates = gates + self.bias_ih.view(4, size) + self.bias_hh.view(4, size)

        in_gate = torch.sigmoid(gates[:, 0])
        in_transform = torch.tanh(gates[:, 1])
        forget_gate = torch.sigmoid(gates[:, 2])
        out_gate = torch.sigmoid(gates[:, 3])
        c_next = forget_gate * c_prev + in_gate * in_transform
        h_next = out_gate * torch.tanh(c_next)
        return h_next, c_next


class VariationalDropoutLSTMModel(nn.Module):
    """Gal-style PTB LSTM branch formerly implemented in Lua/Torch7."""

    def __init__(self, config: PTBConfig, variant: str, rank: int, relaxation_scale: float):
        super().__init__()
        self.config = config
        self.variant = variant
        self.embeddings = EmbeddingVariant(
            config.vocab_size,
            config.hidden_size,
            variant,
            rank,
            config.init_scale,
            relaxation_scale,
        )
        self.cells = nn.ModuleList([VariationalLSTMCell(config.hidden_size) for _ in range(config.num_layers)])
        self.proj = nn.Linear(config.hidden_size, config.hidden_size) if variant in {"pr", "wt_pr"} else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for param in self.parameters():
            nn.init.uniform_(param, -self.config.init_scale, self.config.init_scale)
        self.embeddings.reset_parameters()

    def inverted_bernoulli(self, shape: Tuple[int, ...], dropout: float, device: torch.device) -> torch.Tensor:
        if not self.training or dropout <= 0.0:
            return torch.ones(shape, device=device)
        keep = 1.0 - dropout
        return torch.empty(shape, device=device).bernoulli_(keep).div_(keep)

    def word_input_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, num_steps = input_ids.shape
        device = input_ids.device
        dropout = self.config.dropout_x
        if not self.training or dropout <= 0.0:
            return torch.ones(batch_size, num_steps, 1, device=device)

        keep = 1.0 - dropout
        return torch.empty(batch_size, num_steps, 1, device=device).bernoulli_(keep).div_(keep)

    def forward(self, input_ids: torch.Tensor, hidden: Optional[HiddenState]) -> Tuple[torch.Tensor, HiddenState]:
        batch_size, num_steps = input_ids.shape
        device = input_ids.device
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)

        h_layers = list(torch.unbind(hidden[0], dim=0))
        c_layers = list(torch.unbind(hidden[1], dim=0))
        gate_input_masks = [
            self.inverted_bernoulli((batch_size, 4, self.config.hidden_size), self.config.dropout_i, device)
            for _ in range(self.config.num_layers)
        ]
        gate_hidden_masks = [
            self.inverted_bernoulli((batch_size, 4, self.config.hidden_size), self.config.dropout_h, device)
            for _ in range(self.config.num_layers)
        ]
        output_mask = self.inverted_bernoulli((batch_size, self.config.hidden_size), self.config.dropout_o, device)
        input_mask = self.word_input_mask(input_ids)
        embeddings = self.embeddings.input_embeddings(input_ids)
        outputs: List[torch.Tensor] = []

        for step_idx in range(num_steps):
            layer_input = embeddings[:, step_idx, :] * input_mask[:, step_idx, :]
            for layer_idx, cell in enumerate(self.cells):
                h_next, c_next = cell(
                    layer_input,
                    (h_layers[layer_idx], c_layers[layer_idx]),
                    gate_input_masks[layer_idx],
                    gate_hidden_masks[layer_idx],
                )
                h_layers[layer_idx] = h_next
                c_layers[layer_idx] = c_next
                layer_input = h_next
            outputs.append(layer_input * output_mask)

        output = torch.stack(outputs, dim=1)
        if self.proj is not None:
            output = self.proj(output)
        logits = self.embeddings.output_logits(output)
        next_hidden = torch.stack(h_layers, dim=0), torch.stack(c_layers, dim=0)
        return logits, next_hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> HiddenState:
        weight = next(self.parameters())
        shape = (self.config.num_layers, batch_size, self.config.hidden_size)
        return weight.new_zeros(shape, device=device), weight.new_zeros(shape, device=device)


def build_model(
    config: PTBConfig,
    variant: str,
    architecture: str,
    rank: int,
    relaxation_scale: float,
) -> nn.Module:
    if variant not in VARIANT_CHOICES:
        raise ValueError(f"Unknown variant: {variant}")
    if architecture == "zaremba":
        return StandardLSTMModel(config, variant, rank, relaxation_scale)
    if architecture == "variational":
        return VariationalDropoutLSTMModel(config, variant, rank, relaxation_scale)
    raise ValueError(f"Unknown architecture: {architecture}")
