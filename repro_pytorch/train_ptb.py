"""Modern PyTorch training entrypoint for Press and Wolf PTB LM experiments.

This module owns only the CLI surface and the high-level orchestration.
Configuration tables, embedding variants, model architectures, and the
training loop live in dedicated sibling modules:

- :mod:`configs`     -- ``PTBConfig`` and the named ``CONFIGS`` registry.
- :mod:`variants`    -- S1-S13 metadata and the ``EmbeddingVariant`` module.
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
from typing import Dict, List

import torch
from torch import nn

from configs import CONFIGS
from model import build_model
from ptb_data import PTBBatchedSplit, ptb_raw_data
from train_loop import (
    maybe_save_checkpoint,
    resolve_device,
    run_epoch,
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
        help="Multiplier applied to every non-base S1-S13 relaxation term",
    )
    parser.add_argument("--architecture", default=None, choices=["zaremba", "variational"])
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
    )
    test_config = replace(config, batch_size=1, num_steps=1) if args.paper_test_eval else config

    max_epochs = args.max_epochs if args.max_epochs is not None else config.max_max_epoch
    model = build_model(config, args.variant, architecture, args.lora_rank, args.relaxation_scale).to(device)
    model = _maybe_compile(model, args)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    train_cache = PTBBatchedSplit(train_data, config.batch_size, config.num_steps, device)
    valid_cache = PTBBatchedSplit(valid_data, config.batch_size, config.num_steps, device)
    test_cache = PTBBatchedSplit(test_data, test_config.batch_size, test_config.num_steps, device)

    print(
        json.dumps(
            {
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
                        args.variant,
                        config.vocab_size,
                        config.hidden_size,
                        args.lora_rank,
                    ),
                },
                "legacy_weight_decay": legacy_weight_decay,
                "paper_test_eval": args.paper_test_eval,
                "tf32": args.tf32,
                "cudnn_benchmark": args.cudnn_benchmark,
                "compile": args.compile,
                "compile_mode": args.compile_mode if args.compile else None,
                "speed_mode": args.speed_mode,
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
            },
            indent=2,
        ),
        flush=True,
    )

    metrics: Dict[str, float] = {}
    train_total_sec = 0.0
    eval_total_sec = 0.0
    epoch_throughput: List[float] = []
    overall_start = time.perf_counter()

    for epoch in range(max_epochs):
        lr_decay = config.lr_decay ** max(epoch - config.max_epoch, 0.0)
        lr = config.learning_rate * lr_decay
        for group in optimizer.param_groups:
            group["lr"] = lr

        print(f"Epoch {epoch + 1}/{max_epochs} lr={lr:.5f}", flush=True)
        train_stats = run_epoch(
            model,
            train_data,
            config,
            device,
            optimizer,
            args.max_train_batches,
            verbose=not args.quiet,
            legacy_weight_decay=legacy_weight_decay,
            batch_cache=train_cache,
            log_every=args.log_every,
        )
        valid_stats = run_epoch(
            model,
            valid_data,
            config,
            device,
            optimizer=None,
            max_batches=args.max_eval_batches,
            verbose=False,
            legacy_weight_decay=0.0,
            batch_cache=valid_cache,
            log_every=args.log_every,
        )
        train_total_sec += train_stats.elapsed_sec
        eval_total_sec += valid_stats.elapsed_sec
        epoch_throughput.append(train_stats.words_per_sec)

        metrics[f"epoch_{epoch + 1}_train_ppl"] = train_stats.perplexity
        metrics[f"epoch_{epoch + 1}_valid_ppl"] = valid_stats.perplexity
        metrics[f"epoch_{epoch + 1}_train_sec"] = train_stats.elapsed_sec
        metrics[f"epoch_{epoch + 1}_valid_sec"] = valid_stats.elapsed_sec
        metrics[f"epoch_{epoch + 1}_train_wps"] = train_stats.words_per_sec
        print(
            f"Epoch {epoch + 1} train_ppl={train_stats.perplexity:.3f} "
            f"valid_ppl={valid_stats.perplexity:.3f} "
            f"train_sec={train_stats.elapsed_sec:.1f} "
            f"valid_sec={valid_stats.elapsed_sec:.1f} "
            f"train_wps={train_stats.words_per_sec:.0f}",
            flush=True,
        )

    test_stats = run_epoch(
        model,
        test_data,
        test_config,
        device,
        optimizer=None,
        max_batches=args.max_eval_batches,
        verbose=False,
        legacy_weight_decay=0.0,
        batch_cache=test_cache,
        log_every=args.log_every,
    )
    eval_total_sec += test_stats.elapsed_sec

    overall_elapsed = max(time.perf_counter() - overall_start, 1e-9)
    avg_train_wps = (
        sum(epoch_throughput) / len(epoch_throughput) if epoch_throughput else 0.0
    )

    metrics["test_ppl"] = test_stats.perplexity
    metrics["test_sec"] = test_stats.elapsed_sec
    metrics["timing_total_sec"] = overall_elapsed
    metrics["timing_train_sec"] = train_total_sec
    metrics["timing_eval_sec"] = eval_total_sec
    metrics["timing_avg_train_wps"] = avg_train_wps

    print(
        f"Test perplexity={test_stats.perplexity:.3f} test_sec={test_stats.elapsed_sec:.1f}",
        flush=True,
    )
    print(
        f"Timing total={overall_elapsed:.1f}s train={train_total_sec:.1f}s "
        f"eval={eval_total_sec:.1f}s avg_train_wps={avg_train_wps:.0f}",
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
