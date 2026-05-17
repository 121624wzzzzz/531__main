"""Per-epoch training loop, helper utilities, and checkpoint persistence.

The functions here intentionally take everything they need as arguments
(model, config, device, optimizer, ...) so they can be reused from
``train_ptb.py`` and from quick scripts/tests without pulling in the full
CLI parser.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch import nn

from configs import PTBConfig
from ptb_data import PTBBatchedSplit, ptb_iterator
from variants import HiddenState


def detach_hidden(hidden: HiddenState) -> HiddenState:
    return hidden[0].detach(), hidden[1].detach()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def sequence_loss(logits: torch.Tensor, targets: torch.Tensor, batch_size: int) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="sum",
    ) / batch_size


def projection_regularizer(model: nn.Module, batch_size: int) -> torch.Tensor:
    proj = getattr(model, "proj", None)
    if proj is None:
        return torch.zeros((), device=next(model.parameters()).device)
    return 0.5 * proj.weight.pow(2).sum() / (batch_size * 6.5)


@torch.no_grad()
def apply_legacy_weight_decay(model: nn.Module, amount: float) -> None:
    if amount <= 0.0:
        return
    for param in model.parameters():
        param.add_(param, alpha=-amount)


@dataclass
class EpochStats:
    perplexity: float
    elapsed_sec: float
    batches: int
    tokens: int
    words_per_sec: float


def run_epoch(
    model: nn.Module,
    data: Iterable[int],
    config: PTBConfig,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    max_batches: Optional[int],
    verbose: bool,
    legacy_weight_decay: float,
    batch_cache: Optional[PTBBatchedSplit] = None,
    log_every: int = 100,
) -> EpochStats:
    is_train = optimizer is not None
    model.train(is_train)
    hidden = model.init_hidden(config.batch_size, device)  # type: ignore[attr-defined]
    loss_accumulator = torch.zeros((), device=device)
    last_logged_loss = 0.0
    total_iters = 0
    total_batches = 0
    start_time = time.perf_counter()

    iterator = ptb_iterator(
        data,
        config.batch_size,
        config.num_steps,
        device,
        cache=batch_cache,
    )
    for step, (inputs, targets) in enumerate(iterator):
        if max_batches is not None and step >= max_batches:
            break

        hidden = detach_hidden(hidden)
        logits, hidden = model(inputs, hidden)  # type: ignore[operator]
        loss = sequence_loss(logits, targets, config.batch_size)

        if is_train:
            train_loss = loss + projection_regularizer(model, config.batch_size)
            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            apply_legacy_weight_decay(model, legacy_weight_decay)

        loss_accumulator = loss_accumulator + loss.detach()
        total_iters += config.num_steps
        total_batches += 1

        if verbose and step > 0 and step % log_every == 0:
            running_loss = float(loss_accumulator.item())
            last_logged_loss = running_loss
            ppl = math.exp(running_loss / total_iters)
            elapsed = max(time.perf_counter() - start_time, 1e-9)
            words_per_sec = total_iters * config.batch_size / elapsed
            print(
                f"  batch={step:5d} perplexity={ppl:8.3f} wps={words_per_sec:8.0f}",
                flush=True,
            )

    if total_iters == 0:
        raise RuntimeError("No batches were processed")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    total_loss = float(loss_accumulator.item())
    if verbose and last_logged_loss and total_loss < last_logged_loss * 0.5:
        # Defensive: a non-finite loss mid-epoch would still be reported by
        # the GPU-side accumulator. Surface the discrepancy when it appears.
        print(f"  warning: epoch loss {total_loss:.4f} dropped below last log {last_logged_loss:.4f}", flush=True)
    perplexity = math.exp(total_loss / total_iters)
    words_per_sec = total_iters * config.batch_size / elapsed
    return EpochStats(
        perplexity=perplexity,
        elapsed_sec=elapsed,
        batches=total_batches,
        tokens=total_iters * config.batch_size,
        words_per_sec=words_per_sec,
    )


def maybe_save_checkpoint(
    output_dir: Optional[Path],
    model: nn.Module,
    args: argparse.Namespace,
    config: PTBConfig,
    architecture: str,
    metrics: Dict[str, float],
    save_model: bool = True,
) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    stem = f"ptb_{args.model}_{architecture}_{args.variant}"
    if save_model:
        torch.save(
            {
                "model_state": model.state_dict(),
                "args": serialized_args,
                "config": asdict(config),
                "architecture": architecture,
                "metrics": metrics,
            },
            output_dir / f"{stem}.pt",
        )
    (output_dir / f"{stem}.json").write_text(
        json.dumps(
            {"args": serialized_args, "config": asdict(config), "architecture": architecture, "metrics": metrics},
            indent=2,
        ),
        encoding="utf-8",
    )
