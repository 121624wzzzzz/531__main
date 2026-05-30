"""Recurrent Highway Network language model with Gal variational dropout.

Architecture follows Zilly et al. (2017) PTB setup used by Press & Wolf (2017):
single RHN layer, configurable recurrence depth, coupled carry/transform gates,
and variational dropout masks shared across time and internal recurrence steps.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from configs import PTBConfig
from nn_utils import inverted_bernoulli, word_input_mask
from variants import EmbeddingVariant, HiddenState


class VariationalDropoutRHNModel(nn.Module):
    """Press & Wolf / Zilly-style variational RHN for PTB word-level LM."""

    def __init__(self, config: PTBConfig, variant: str, rank: int, relaxation_scale: float):
        super().__init__()
        if config.recurrence_depth < 1:
            raise ValueError("recurrence_depth must be >= 1")

        self.config = config
        self.variant = variant
        self.depth = config.recurrence_depth
        hidden = config.hidden_size

        self.embeddings = EmbeddingVariant(
            config.vocab_size,
            hidden,
            variant,
            rank,
            config.init_scale,
            relaxation_scale,
        )

        self.weight_h_x = nn.Parameter(torch.empty(hidden, hidden))
        self.weight_t_x = nn.Parameter(torch.empty(hidden, hidden))
        self.weight_h_r = nn.ParameterList(
            nn.Parameter(torch.empty(hidden, hidden)) for _ in range(self.depth)
        )
        self.weight_t_r = nn.ParameterList(
            nn.Parameter(torch.empty(hidden, hidden)) for _ in range(self.depth)
        )
        self.bias_h = nn.ParameterList(nn.Parameter(torch.empty(hidden)) for _ in range(self.depth))
        self.bias_t = nn.ParameterList(nn.Parameter(torch.empty(hidden)) for _ in range(self.depth))
        self.proj = nn.Linear(hidden, hidden) if variant in {"pr", "wt_pr"} else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        scale = self.config.init_scale
        for param in self.parameters():
            if param is not None:
                nn.init.uniform_(param, -scale, scale)
        for bias in self.bias_h:
            nn.init.zeros_(bias)
        for bias in self.bias_t:
            nn.init.constant_(bias, self.config.t_gate_bias)
        self.embeddings.reset_parameters()

    def forward(self, input_ids: torch.Tensor, hidden: Optional[HiddenState]) -> Tuple[torch.Tensor, HiddenState]:
        batch_size, num_steps = input_ids.shape
        device = input_ids.device
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)

        state, _dummy = hidden
        input_gate_mask = inverted_bernoulli((batch_size, 1, self.config.hidden_size), self.config.dropout_i, self.training, device)
        state_mask = inverted_bernoulli((batch_size, 1, self.config.hidden_size), self.config.dropout_h, self.training, device)
        output_mask = inverted_bernoulli((batch_size, self.config.hidden_size), self.config.dropout_o, self.training, device)
        word_masks = word_input_mask(input_ids, self.config.dropout_x, self.training)
        embeddings = self.embeddings.input_embeddings(input_ids)
        outputs: List[torch.Tensor] = []

        input_gate_mask = input_gate_mask.squeeze(1)
        state_mask = state_mask.squeeze(1)

        for step_idx in range(num_steps):
            x = embeddings[:, step_idx, :] * word_masks[:, step_idx, :]
            state = self._step(x, state, input_gate_mask, state_mask)
            outputs.append(state * output_mask)

        output = torch.stack(outputs, dim=1)
        if self.proj is not None:
            output = self.proj(output)
        logits = self.embeddings.output_logits(output)
        return logits, (state, state.new_zeros(1))

    def _step(
        self,
        x: torch.Tensor,
        prev_state: torch.Tensor,
        input_gate_mask: torch.Tensor,
        state_mask: torch.Tensor,
    ) -> torch.Tensor:
        """One timestep of variational RHN matching jzilly/RecurrentHighwayNetworks."""
        state = prev_state
        for layer_idx in range(self.depth):
            masked_state = state * state_mask
            if layer_idx == 0:
                masked_x = x * input_gate_mask
                h_pre = (
                    F.linear(masked_x, self.weight_h_x)
                    + F.linear(masked_state, self.weight_h_r[layer_idx])
                    + self.bias_h[layer_idx]
                )
                t_pre = (
                    F.linear(masked_x, self.weight_t_x)
                    + F.linear(masked_state, self.weight_t_r[layer_idx])
                    + self.bias_t[layer_idx]
                )
            else:
                h_pre = F.linear(masked_state, self.weight_h_r[layer_idx]) + self.bias_h[layer_idx]
                t_pre = F.linear(masked_state, self.weight_t_r[layer_idx]) + self.bias_t[layer_idx]
            transform = torch.sigmoid(t_pre)
            state = torch.tanh(h_pre) * transform + state * (1.0 - transform)
        return state

    def init_hidden(self, batch_size: int, device: torch.device) -> HiddenState:
        weight = next(self.parameters())
        state = weight.new_zeros(batch_size, self.config.hidden_size, device=device)
        return state, state.new_zeros(1)
