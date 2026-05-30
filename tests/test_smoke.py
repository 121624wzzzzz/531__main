"""Minimal smoke tests — run with pytest or directly:

    PYTHONPATH=repro_pytorch pytest tests/ -v
    PYTHONPATH=repro_pytorch python tests/test_smoke.py   # no pytest needed
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch

# Allow running from repo root without PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "repro_pytorch"))

from configs import CONFIGS, PTBConfig
from model import build_model
from nn_utils import inverted_bernoulli
from ptb_data import PTBBatchedSplit
from variants import VARIANT_CHOICES, EmbeddingVariant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

V, D, R = 100, 32, 4  # tiny vocab / hidden / rank

BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

_passed = 0
_failed = 0


def _make_inputs(batch=4, steps=8):
    return torch.randint(0, V, (batch, steps)), torch.randn(batch, steps, D)


def check(name: str, fn):
    global _passed, _failed
    try:
        fn()
        _passed += 1
        print(f"  {GREEN}PASS{RESET} {name}")
    except Exception:
        _failed += 1
        print(f"  {RED}FAIL{RESET} {name}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all_variants_forward():
    """Every variant produces finite outputs of the correct shape."""
    for v in sorted(VARIANT_CHOICES):
        emb = EmbeddingVariant(V, D, v, R, 0.04, 0.1)
        emb.eval()
        input_ids, hidden = _make_inputs()
        with torch.no_grad():
            out_emb = emb.input_embeddings(input_ids)
            out_logits = emb.output_logits(hidden)
        assert out_emb.shape == (4, 8, D), f"{v}: bad emb shape {out_emb.shape}"
        assert out_logits.shape == (4, 8, V), f"{v}: bad logits shape"
        assert out_logits.isfinite().all(), f"{v}: non-finite logits"


def test_all_variants_backward():
    """Every variant completes a full forward+backward pass with no NaN grads."""
    for v in sorted(VARIANT_CHOICES):
        emb = EmbeddingVariant(V, D, v, R, 0.04, 0.1)
        emb.train()
        input_ids, _ = _make_inputs()
        emb_out = emb.input_embeddings(input_ids)
        hidden = emb_out + torch.randn(4, 8, D) * 0.1
        logits = emb.output_logits(hidden)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, V), input_ids.reshape(-1))
        loss.backward()
        for pname, param in emb.named_parameters():
            assert param.grad is not None, f"{v}: {pname} no grad"
            assert param.grad.isfinite().all(), f"{v}: {pname} non-finite grad"


def test_all_architectures():
    """Each architecture builds and does a forward pass."""
    for arch in ["zaremba", "variational", "transformer"]:
        cfg = PTBConfig(
            init_scale=0.1, learning_rate=1.0, max_grad_norm=5.0,
            num_layers=2, num_steps=8, hidden_size=D, max_epoch=4,
            max_max_epoch=13, keep_prob=1.0, lr_decay=0.5, batch_size=4,
            architecture=arch, vocab_size=V,
        )
        model = build_model(cfg, "s7", arch, R, 0.05)
        model.eval()
        input_ids = torch.randint(0, V, (4, 8))
        hidden = model.init_hidden(4, torch.device("cpu"))
        with torch.no_grad():
            logits, _ = model(input_ids, hidden)
        assert logits.isfinite().all(), f"{arch}: non-finite logits"


def test_data_loading():
    """PTBBatchedSplit yields correct shapes."""
    split = PTBBatchedSplit(
        list(range(1000)), batch_size=10, num_steps=5, device=torch.device("cpu"))
    batches = list(split)
    assert len(batches) > 0
    for x, y in batches:
        assert x.shape == (10, 5)
        assert y.shape == (10, 5)


def test_configs_load():
    """Every named config is valid."""
    for name in CONFIGS:
        cfg = CONFIGS[name]
        assert cfg.hidden_size > 0
        assert cfg.batch_size > 0
        assert cfg.num_layers > 0


def test_nn_utils():
    """inverted_bernoulli produces correct shapes and scaling."""
    for training in [True, False]:
        mask = inverted_bernoulli((2, 3), 0.5, training, torch.device("cpu"))
        assert mask.shape == (2, 3)
        if not training:
            assert (mask == 1.0).all()


# ---------------------------------------------------------------------------
# Runner (works with or without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"{BOLD}EmbeddingVariant tests{RESET}")
    check("forward pass (all variants)", test_all_variants_forward)
    check("backward pass (all variants)", test_all_variants_backward)
    print(f"{BOLD}Model tests{RESET}")
    check("architecture forward pass", test_all_architectures)
    print(f"{BOLD}Data tests{RESET}")
    check("PTBBatchedSplit shapes", test_data_loading)
    print(f"{BOLD}Config tests{RESET}")
    check("all configs load", test_configs_load)
    print(f"{BOLD}nn_utils tests{RESET}")
    check("inverted_bernoulli", test_nn_utils)
    print(f"\n{BOLD}{_passed} passed, {_failed} failed{RESET}")
    raise SystemExit(1 if _failed else 0)
