#!/usr/bin/env python
"""Compute eval_loss for the frozen base model — Claim 2 zero-training reference.

Loads ``--model-path`` without any adapter and computes cross-entropy loss on
``--eval-data`` using the same tokenization as ``train_affine_vocab_lora.py``
(prompt prefix masked out with -100). The single number reported is the formal
zero-training reference for Claim 2 in ``docs/RESULTS_SO_FAR.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_affine_vocab_lora import tokenize_row  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--report-file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    ds = load_dataset("json", data_files=args.eval_data, split="train")
    if args.max_eval_samples:
        ds = ds.select(range(min(args.max_eval_samples, len(ds))))
    tokenized = ds.map(
        lambda row: tokenize_row(row, tokenizer, args.max_seq_len),
        remove_columns=ds.column_names,
        desc="Tokenizing eval",
    )

    training_args = TrainingArguments(
        output_dir="/tmp/eval_base_loss",
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        bf16=args.bf16,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
        ),
    )
    metrics = trainer.evaluate(eval_dataset=tokenized)
    report = {
        "model_path": args.model_path,
        "eval_data": args.eval_data,
        "num_eval_samples": len(tokenized),
        "eval_loss": metrics.get("eval_loss"),
        "eval_runtime": metrics.get("eval_runtime"),
        "eval_samples_per_second": metrics.get("eval_samples_per_second"),
    }
    print(json.dumps(report, indent=2))
    if args.report_file:
        out = Path(args.report_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
