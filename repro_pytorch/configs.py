"""PTB experiment configurations.

Defines :class:`PTBConfig` and the registry :data:`CONFIGS` used by every
training run. Pulled out of ``train_ptb.py`` so configuration tweaks do not
require touching the model, training loop, or CLI plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    "large4090": PTBConfig(
        init_scale=0.04,
        learning_rate=1.0,
        max_grad_norm=10.0,
        num_layers=2,
        num_steps=35,
        hidden_size=1500,
        max_epoch=8,
        max_max_epoch=35,
        keep_prob=0.35,
        lr_decay=1 / 1.15,
        batch_size=128,
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
    "bayes4090": PTBConfig(
        init_scale=0.04,
        learning_rate=1.0,
        max_grad_norm=10.0,
        num_layers=2,
        num_steps=35,
        hidden_size=1500,
        max_epoch=8,
        max_max_epoch=35,
        keep_prob=1.0,
        lr_decay=1 / 1.15,
        batch_size=64,
        architecture="variational",
        dropout_x=0.3,
        dropout_i=0.5,
        dropout_h=0.3,
        dropout_o=0.5,
        legacy_weight_decay=1e-7,
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
