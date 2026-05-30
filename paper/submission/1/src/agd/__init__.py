"""Asymmetric Geometric Decoupling (AGD) for weight-tied language models."""

from .config import AGDConfig
from .embedding_head import AGDEmbeddingHead, VariantSpec, VARIANTS
from .model import AGDCausalLM

__all__ = [
    "AGDConfig",
    "AGDEmbeddingHead",
    "AGDCausalLM",
    "VariantSpec",
    "VARIANTS",
]
