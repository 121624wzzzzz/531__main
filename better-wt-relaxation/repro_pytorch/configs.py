"""PTB experiment configurations.

Defines :class:`PTBConfig` and the registry :data:`CONFIGS` used by every
training run. Pulled out of ``train_ptb.py`` so configuration tweaks do not
require touching the model, training loop, or CLI plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict


@dataclass(frozen=True)
class PTBConfig:
    init_scale: float
    learning_rate: float
    max_grad_norm: float
    num_layers: int
    num_steps: int
    hidden_size: int
    max_epoch: int
    max_max_epoch: int
    keep_prob: float
    lr_decay: float
    batch_size: int
    architecture: str = "zaremba"
    dropout_x: float = 0.0
    dropout_i: float = 0.0
    dropout_h: float = 0.0
    dropout_o: float = 0.0
    legacy_weight_decay: float = 0.0
    vocab_size: int = 10000
    recurrence_depth: int = 1
    t_gate_bias: float = 0.0


CONFIGS: Dict[str, PTBConfig] = {
    "small": PTBConfig(
        init_scale=0.1,
        learning_rate=1.0,
        max_grad_norm=5.0,
        num_layers=2,
        num_steps=20,
        hidden_size=200,
        max_epoch=4,
        max_max_epoch=13,
        keep_prob=1.0,
        lr_decay=0.5,
        batch_size=20,
    ),
    "medium": PTBConfig(
        init_scale=0.05,
        learning_rate=1.0,
        max_grad_norm=5.0,
        num_layers=2,
        num_steps=35,
        hidden_size=650,
        max_epoch=6,
        max_max_epoch=39,
        keep_prob=0.5,
        lr_decay=0.8,
        batch_size=20,
    ),
    "large": PTBConfig(
        init_scale=0.04,
        learning_rate=1.0,
        max_grad_norm=10.0,
        num_layers=2,
        num_steps=35,
        hidden_size=1500,
        max_epoch=14,
        max_max_epoch=55,
        keep_prob=0.35,
        lr_decay=1 / 1.15,
        batch_size=20,
    ),
    "bayes1500": PTBConfig(
        init_scale=0.04,
        learning_rate=1.0,
        max_grad_norm=10.0,
        num_layers=2,
        num_steps=35,
        hidden_size=1500,
        max_epoch=10,
        max_max_epoch=55,
        keep_prob=1.0,
        lr_decay=1 / 1.15,
        batch_size=20,
        architecture="variational",
        dropout_x=0.3,
        dropout_i=0.5,
        dropout_h=0.3,
        dropout_o=0.5,
        legacy_weight_decay=1e-7,
    ),
    "rhn830": PTBConfig(
        init_scale=0.04,
        learning_rate=0.2,
        max_grad_norm=10.0,
        num_layers=1,
        num_steps=35,
        hidden_size=830,
        max_epoch=19,
        max_max_epoch=55,
        keep_prob=1.0,
        lr_decay=1 / 1.02,
        batch_size=20,
        architecture="rhn",
        dropout_x=0.25,
        dropout_i=0.75,
        dropout_h=0.25,
        dropout_o=0.75,
        legacy_weight_decay=1e-7,
        recurrence_depth=10,
        t_gate_bias=-4.0,
    ),
    "gpt512": PTBConfig(
        init_scale=0.02,
        learning_rate=0.0005,
        max_grad_norm=1.0,
        num_layers=4,
        num_steps=128,
        hidden_size=512,
        max_epoch=10,
        max_max_epoch=50,
        keep_prob=0.9,
        lr_decay=1.0,
        batch_size=128,
        architecture="transformer",
    ),
    "test": PTBConfig(
        init_scale=0.1,
        learning_rate=1.0,
        max_grad_norm=1.0,
        num_layers=1,
        num_steps=2,
        hidden_size=2,
        max_epoch=1,
        max_max_epoch=1,
        keep_prob=1.0,
        lr_decay=0.5,
        batch_size=20,
    ),
}

# Derived screening configs — same model, different batch / schedule only.
# Kept in CONFIGS so --model large4090 / bayes4090 / rhn4090 work as CLI shortcuts.
CONFIGS["large4090"] = replace(CONFIGS["large"], max_epoch=8, max_max_epoch=35, batch_size=128)
CONFIGS["bayes4090"] = replace(CONFIGS["bayes1500"], max_epoch=8, max_max_epoch=35, batch_size=64)
CONFIGS["rhn4090"] = replace(CONFIGS["rhn830"], max_epoch=20, max_max_epoch=80, batch_size=512, t_gate_bias=-2.0)


def large_with_batch(batch_size: int, max_epoch: int = 14, max_max_epoch: int = 55) -> PTBConfig:
    """Derive a ``large`` variant that only differs in batch size and schedule.

    Used for sweep configurations where the hidden size, dropout, LR, and
    other hyperparameters are kept identical to the paper ``large`` setting.
    """
    return PTBConfig(
        init_scale=0.04,
        learning_rate=1.0,
        max_grad_norm=10.0,
        num_layers=2,
        num_steps=35,
        hidden_size=1500,
        max_epoch=max_epoch,
        max_max_epoch=max_max_epoch,
        keep_prob=0.35,
        lr_decay=1 / 1.15,
        batch_size=batch_size,
    )
