"""Modern PyTorch training entrypoint for Press and Wolf PTB LM experiments.

This module owns only the CLI surface and the high-level orchestration.
Configuration tables, embedding variants, model architectures, and the
training loop live in dedicated sibling modules:

- :mod:`configs`     -- ``PTBConfig`` and the named ``CONFIGS`` registry.
- :mod:`variants`    -- S1-S21 metadata and the ``EmbeddingVariant`` module.
- :mod:`model`       -- standard / variational LSTM models and ``build_model``.
- :mod:`train_loop`  -- ``run_epoch``, helpers, and checkpoint persistence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import nn

from configs import CONFIGS
from model import build_model
from ptb_data import PTBBatchedSplit, ptb_raw_data
from train_loop import (
    TrainingConfig,
    build_optimizer_and_scheduler,
    maybe_save_checkpoint,
    resolve_device,
    run_training,
    set_seed,
)
from variants import VARIANT_CHOICES, actual_extra_params, variant_formula


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True, help="Path containing ptb.train.txt, ptb.valid.txt, ptb.test.txt")
    parser.add_argument("--model", default="small", choices=sorted(CONFIGS))
    parser.add_argument("--variant", default="wt", choices=VARIANT_CHOICES)
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank r for S4-S7/S9-S10/S12-S13")
    parser.add_argument(
        "--relaxation_scale",
        type=float,
        default=1.0,
        help="Multiplier applied to every non-base S1-S21 relaxation term",
    )
    parser.add_argument("--architecture", default=None, choices=["zaremba", "variational", "rhn", "transformer"])
    parser.add_argument("--legacy_weight_decay", type=float, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=None, help="Override the paper epoch count for quick checks")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Limit training batches for smoke tests")
    parser.add_argument("--max_eval_batches", type=int, default=None, help="Limit validation/test batches for smoke tests")
    parser.add_argument("--paper_test_eval", action="store_true", help="Use batch_size=1,num_steps=1 for final test")
    parser.add_argument("--tf32", action="store_true", help="Allow TF32 matmul on NVIDIA Ampere/Ada GPUs")
    parser.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        help="Set torch.backends.cudnn.benchmark=True (non-deterministic, fixed shapes only)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model in torch.compile (PyTorch 2.x); approximate equivalence only",
    )
    parser.add_argument(
        "--compile_mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode if --compile is set",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="Log a running perplexity every N training batches (forces a small GPU sync)",
    )
    parser.add_argument(
        "--metrics_only",
        action="store_true",
        help="Write JSON metrics but skip the .pt model checkpoint (useful for large sweeps)",
    )
    parser.add_argument(
        "--speed_mode",
        default=None,
        choices=["strict-fp32", "modern-fast"],
        help=(
            "Convenience preset. strict-fp32 disables TF32/cuDNN benchmark/compile; "
            "modern-fast enables TF32 and cuDNN benchmark."
        ),
    )
    parser.add_argument("--optimizer", default=None, choices=["sgd", "adamw"])
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--decay_start_epoch", type=int, default=None)
    parser.add_argument("--lr_decay_override", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--lr_schedule", default=None, choices=["epoch_decay", "warmup_cosine"])
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--warmup_fraction", type=float, default=0.05)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument(
        "--no_restore_best",
        action="store_true",
        help="Do not restore the best-validation checkpoint before final test",
    )
    parser.add_argument("--output_dir", type=Path, default=None, help="Optional checkpoint/metric output directory")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _apply_speed_mode(args: argparse.Namespace) -> None:
    """Translate ``--speed_mode`` presets into the underlying flag values.

    Explicit per-flag overrides on the CLI win: a preset never downgrades a
    flag the user already enabled. ``strict-fp32`` is intentionally
    conservative (it does not silently turn off ``--tf32`` if the user passed
    it) so the recorded ``args`` always reflect what actually ran.
    """

    if args.speed_mode is None:
        return
    if args.speed_mode == "strict-fp32":
        # Pure documentation marker; no flags to enable.
        return
    if args.speed_mode == "modern-fast":
        args.tf32 = True
        args.cudnn_benchmark = True


def _maybe_compile(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    if not args.compile:
        return model
    if not hasattr(torch, "compile"):
        print("warning: torch.compile not available in this PyTorch build; skipping", flush=True)
        return model
    print(f"Compiling model with torch.compile(mode={args.compile_mode!r})", flush=True)
    return torch.compile(model, mode=args.compile_mode)  # type: ignore[arg-type]


def _build_run_metadata(
    args, config, architecture, device, train_data, valid_data, test_data,
    vocab_size, train_cache, valid_cache, test_cache,
    optimizer_name, schedule_name, legacy_weight_decay,
) -> dict:
    """Assemble the JSON metadata block printed at the start of every run."""
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "model": args.model,
        "architecture": architecture,
        "variant": args.variant,
        "variant_info": {
            **variant_formula(args.variant),
            "lora_rank": args.lora_rank,
            "relaxation_scale": args.relaxation_scale,
            "actual_extra_params": actual_extra_params(
                args.variant, config.vocab_size, config.hidden_size, args.lora_rank,
            ),
        },
        "legacy_weight_decay": legacy_weight_decay,
        "paper_test_eval": args.paper_test_eval,
        "tf32": args.tf32,
        "cudnn_benchmark": args.cudnn_benchmark,
        "compile": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "speed_mode": args.speed_mode,
        "optimizer": optimizer_name,
        "lr_schedule": schedule_name,
        "weight_decay": args.weight_decay if optimizer_name == "adamw" else 0.0,
        "adam_beta2": args.adam_beta2 if optimizer_name == "adamw" else None,
        "warmup_fraction": args.warmup_fraction if schedule_name == "warmup_cosine" else None,
        "warmup_steps": args.warmup_steps if schedule_name == "warmup_cosine" else None,
        "min_lr_ratio": args.min_lr_ratio if schedule_name == "warmup_cosine" else None,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "restore_best": not args.no_restore_best,
        "config": asdict(config),
        "dataset_tokens": {
            "train": len(train_data),
            "valid": len(valid_data),
            "test": len(test_data),
            "vocab": vocab_size,
        },
        "epoch_size": {
            "train_batches": len(train_cache),
            "valid_batches": len(valid_cache),
            "test_batches": len(test_cache),
        },
    }


def main() -> int:
    args = parse_args()
    _apply_speed_mode(args)

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    set_seed(args.seed)
    device = resolve_device(args.device)

    train_data, valid_data, test_data, vocab_size = ptb_raw_data(args.data_path)
    base_config = CONFIGS[args.model]
    architecture = args.architecture or base_config.architecture
    legacy_weight_decay = (
        base_config.legacy_weight_decay if args.legacy_weight_decay is None else args.legacy_weight_decay
    )
    config = replace(
        base_config,
        architecture=architecture,
        vocab_size=vocab_size,
        legacy_weight_decay=legacy_weight_decay,
        learning_rate=args.learning_rate if args.learning_rate is not None else base_config.learning_rate,
        max_epoch=args.decay_start_epoch if args.decay_start_epoch is not None else base_config.max_epoch,
        lr_decay=args.lr_decay_override if args.lr_decay_override is not None else base_config.lr_decay,
    )
    test_config = replace(config, batch_size=1, num_steps=1) if args.paper_test_eval else config

    max_epochs = args.max_epochs if args.max_epochs is not None else config.max_max_epoch
    model = build_model(config, args.variant, architecture, args.lora_rank, args.relaxation_scale).to(device)
    model = _maybe_compile(model, args)

    train_cache = PTBBatchedSplit(train_data, config.batch_size, config.num_steps, device)
    valid_cache = PTBBatchedSplit(valid_data, config.batch_size, config.num_steps, device)
    test_cache = PTBBatchedSplit(test_data, test_config.batch_size, test_config.num_steps, device)
    optimizer_name = args.optimizer or ("adamw" if architecture == "transformer" else "sgd")
    schedule_name = args.lr_schedule or ("warmup_cosine" if optimizer_name == "adamw" else "epoch_decay")
    optimizer, scheduler, _opt_name, _sched_name = build_optimizer_and_scheduler(
        model, config,
        optimizer_name=optimizer_name,
        schedule_name=schedule_name,
        weight_decay=args.weight_decay,
        adam_beta2=args.adam_beta2,
        train_batches=len(train_cache) if args.max_train_batches is None else min(len(train_cache), args.max_train_batches),
        max_epochs=max_epochs,
        warmup_steps=args.warmup_steps,
        warmup_fraction=args.warmup_fraction,
        min_lr_ratio=args.min_lr_ratio,
    )

    metadata = _build_run_metadata(
        args, config, architecture, device, train_data, valid_data, test_data,
        vocab_size, train_cache, valid_cache, test_cache,
        optimizer_name, schedule_name, legacy_weight_decay,
    )
    print(json.dumps(metadata, indent=2), flush=True)

    overall_start = time.perf_counter()
    tc = TrainingConfig(
        max_epochs=max_epochs,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        legacy_weight_decay=legacy_weight_decay,
        quiet=args.quiet,
        log_every=args.log_every,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        no_restore_best=args.no_restore_best,
    )
    result = run_training(
        model=model,
        config=config,
        test_config=test_config,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        device=device,
        optimizer=optimizer,
        train_cache=train_cache,
        valid_cache=valid_cache,
        test_cache=test_cache,
        schedule_name=schedule_name,
        tc=tc,
        scheduler=scheduler,
    )
    metrics = result.metrics
    overall_elapsed = max(time.perf_counter() - overall_start, 1e-9)
    metrics["timing_total_sec"] = overall_elapsed

    print(
        f"Test perplexity={metrics['test_ppl']:.3f} test_sec={metrics['test_sec']:.1f} "
        f"best_epoch={result.best_epoch} best_valid_ppl={result.best_valid_ppl:.3f}",
        flush=True,
    )
    print(
        f"Timing total={overall_elapsed:.1f}s train={metrics['timing_train_sec']:.1f}s "
        f"eval={metrics['timing_eval_sec']:.1f}s avg_train_wps={metrics['timing_avg_train_wps']:.0f}",
        flush=True,
    )
    maybe_save_checkpoint(
        args.output_dir,
        model,
        args,
        config,
        architecture,
        metrics,
        save_model=not args.metrics_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
