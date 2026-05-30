"""Per-epoch training loop, helper utilities, and checkpoint persistence.

The functions here intentionally take everything they need as arguments
(model, config, device, optimizer, ...) so they can be reused from
``train_ptb.py`` and from quick scripts/tests without pulling in the full
CLI parser.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

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


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: PTBConfig,
    *,
    optimizer_name: str = "sgd",
    schedule_name: str = "epoch_decay",
    weight_decay: float = 0.0,
    adam_beta2: float = 0.95,
    train_batches: int = 1,
    max_epochs: int = 1,
    warmup_steps: Optional[int] = None,
    warmup_fraction: float = 0.05,
    min_lr_ratio: float = 0.1,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None, str, str]:
    """Build an optimizer and (optionally) a warmup-cosine scheduler.

    Returns ``(optimizer, scheduler_or_None, optimizer_name, schedule_name)``.
    """
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate,
            betas=(0.9, adam_beta2), weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    if schedule_name != "warmup_cosine":
        return optimizer, None, optimizer_name, schedule_name

    total_steps = max(train_batches * max_epochs, 1)
    ws = warmup_steps
    if ws is None:
        ws = max(1, int(total_steps * warmup_fraction))
    ws = min(ws, max(total_steps - 1, 1))
    cosine_steps = max(total_steps - ws, 1)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=ws),
            CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=config.learning_rate * min_lr_ratio),
        ],
        milestones=[ws],
    )
    return optimizer, scheduler, optimizer_name, schedule_name


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
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
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
            if scheduler is not None:
                scheduler.step()
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


@dataclass
class TrainingResult:
    metrics: Dict[str, float]
    best_valid_ppl: float
    best_epoch: int
    best_state: Optional[Dict[str, torch.Tensor]]


@dataclass
class TrainingConfig:
    """Hyperparameters that control the training loop itself (not the model)."""
    max_epochs: int
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None
    legacy_weight_decay: float = 0.0
    quiet: bool = False
    log_every: int = 100
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0
    no_restore_best: bool = False


def run_training(
    model: nn.Module,
    config: PTBConfig,
    test_config: PTBConfig,
    train_data: Iterable[int],
    valid_data: Iterable[int],
    test_data: Iterable[int],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    train_cache: PTBBatchedSplit,
    valid_cache: PTBBatchedSplit,
    test_cache: PTBBatchedSplit,
    schedule_name: str,
    tc: TrainingConfig,
    scheduler: Optional[Any] = None,
) -> TrainingResult:
    """Run a full multi-epoch train / valid / test pipeline.

    Returns a :class:`TrainingResult` with per-epoch metrics and the best
    validation state (if early-stopping restore is enabled).
    """
    metrics: Dict[str, float] = {}
    train_total_sec = 0.0
    eval_total_sec = 0.0
    epoch_throughput: List[float] = []
    best_valid_ppl = float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stale_epochs = 0

    for epoch in range(tc.max_epochs):
        if schedule_name == "epoch_decay":
            lr_decay = config.lr_decay ** max(epoch - config.max_epoch, 0.0)
            lr = config.learning_rate * lr_decay
            for group in optimizer.param_groups:
                group["lr"] = lr
        else:
            lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch + 1}/{tc.max_epochs} lr={lr:.5f}", flush=True)
        train_stats = run_epoch(
            model, train_data, config, device, optimizer,
            tc.max_train_batches, verbose=not tc.quiet,
            legacy_weight_decay=tc.legacy_weight_decay,
            batch_cache=train_cache, log_every=tc.log_every,
            scheduler=scheduler,
        )
        valid_stats = run_epoch(
            model, valid_data, config, device, optimizer=None,
            max_batches=tc.max_eval_batches, verbose=False,
            legacy_weight_decay=0.0, batch_cache=valid_cache,
            log_every=tc.log_every,
        )
        train_total_sec += train_stats.elapsed_sec
        eval_total_sec += valid_stats.elapsed_sec
        epoch_throughput.append(train_stats.words_per_sec)

        metrics[f"epoch_{epoch + 1}_train_ppl"] = train_stats.perplexity
        metrics[f"epoch_{epoch + 1}_valid_ppl"] = valid_stats.perplexity
        metrics[f"epoch_{epoch + 1}_train_sec"] = train_stats.elapsed_sec
        metrics[f"epoch_{epoch + 1}_valid_sec"] = valid_stats.elapsed_sec
        metrics[f"epoch_{epoch + 1}_train_wps"] = train_stats.words_per_sec

        improved = valid_stats.perplexity < best_valid_ppl - tc.early_stopping_min_delta
        if improved:
            best_valid_ppl = valid_stats.perplexity
            best_epoch = epoch + 1
            stale_epochs = 0
            if tc.early_stopping_patience is not None and not tc.no_restore_best:
                best_state = copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )
        else:
            stale_epochs += 1

        print(
            f"Epoch {epoch + 1} train_ppl={train_stats.perplexity:.3f} "
            f"valid_ppl={valid_stats.perplexity:.3f} "
            f"train_sec={train_stats.elapsed_sec:.1f} "
            f"valid_sec={valid_stats.elapsed_sec:.1f} "
            f"train_wps={train_stats.words_per_sec:.0f}",
            flush=True,
        )
        if tc.early_stopping_patience is not None and stale_epochs >= tc.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch + 1}; "
                f"best_epoch={best_epoch} best_valid_ppl={best_valid_ppl:.3f}",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})

    test_stats = run_epoch(
        model, test_data, test_config, device, optimizer=None,
        max_batches=tc.max_eval_batches, verbose=False,
        legacy_weight_decay=0.0, batch_cache=test_cache,
        log_every=tc.log_every,
    )
    eval_total_sec += test_stats.elapsed_sec

    avg_train_wps = (
        sum(epoch_throughput) / len(epoch_throughput) if epoch_throughput else 0.0
    )
    metrics["test_ppl"] = test_stats.perplexity
    metrics["test_sec"] = test_stats.elapsed_sec
    metrics["timing_total_sec"] = 0.0  # filled by caller with wall-clock
    metrics["timing_train_sec"] = train_total_sec
    metrics["timing_eval_sec"] = eval_total_sec
    metrics["timing_avg_train_wps"] = avg_train_wps
    metrics["best_epoch"] = best_epoch
    metrics["best_valid_ppl"] = best_valid_ppl
    metrics["restored_best_for_test"] = best_state is not None

    return TrainingResult(
        metrics=metrics,
        best_valid_ppl=best_valid_ppl,
        best_epoch=best_epoch,
        best_state=best_state,
    )
