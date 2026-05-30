"""Shared neural-network utilities for variational dropout models.

Both :class:`~model.VariationalDropoutLSTMModel` and
:class:`~rhn.VariationalDropoutRHNModel` use inverted-bernoulli dropout
masks; these standalone functions keep the logic in one place.
"""

from __future__ import annotations

from typing import Tuple

import torch


def inverted_bernoulli(
    shape: Tuple[int, ...],
    dropout: float,
    training: bool,
    device: torch.device,
) -> torch.Tensor:
    """Return a tensor where each element is 1 with prob ``1 - dropout``,
    else scaled by ``1 / (1 - dropout)`` (inverted dropout).

    When ``training`` is ``False`` or ``dropout <= 0``, returns all-ones.
    """
    if not training or dropout <= 0.0:
        return torch.ones(shape, device=device)
    keep = 1.0 - dropout
    return torch.empty(shape, device=device).bernoulli_(keep).div_(keep)


def word_input_mask(
    input_ids: torch.Tensor,
    dropout: float,
    training: bool,
) -> torch.Tensor:
    """Return a ``[batch, num_steps, 1]`` inverted-bernoulli mask for
    word-level input dropout.
    """
    batch_size, num_steps = input_ids.shape
    device = input_ids.device
    return inverted_bernoulli((batch_size, num_steps, 1), dropout, training, device)
