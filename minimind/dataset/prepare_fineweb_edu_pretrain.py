#!/usr/bin/env python3
"""Prepare FineWeb-Edu parquet files as packed MiniMind pretrain blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "fineweb_edu" / "raw" / "sample" / "10BT",
        help="Directory containing FineWeb-Edu parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "fineweb_edu" / "gpt2_packed",
        help="Directory for the packed Hugging Face dataset.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("gpt2"),
        help="Tokenizer path or Hugging Face name.",
    )
    parser.add_argument("--max-seq-len", type=int, default=340, help="Packed training sequence length.")
    parser.add_argument("--num-proc", type=int, default=8, help="Parallel processes for dataset.map.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Raw documents per map batch.")
    parser.add_argument("--writer-batch-size", type=int, default=1000, help="Arrow writer batch size.")
    parser.add_argument("--min-chars", type=int, default=20, help="Drop very short documents before tokenization.")
    parser.add_argument("--max-samples", type=int, default=0, help="Debug only: process first N raw rows.")
    parser.add_argument("--max-tokens", type=int, default=6_000_000_000, help="Keep first N packed tokens; 0 means no limit.")
    parser.add_argument("--file-limit", type=int, default=0, help="Debug only: process first N parquet files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists.")
    return parser.parse_args()


def parquet_files(input_dir: Path) -> list[str]:
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    return [str(path) for path in files]


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already exists; pass --overwrite to replace it")

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must define eos_token_id")

    data_files = parquet_files(args.input_dir)
    if args.file_limit > 0:
        data_files = data_files[: args.file_limit]
    print(f"[prepare] input files: {len(data_files)}")
    print(f"[prepare] output dir : {args.output_dir}")
    print(f"[prepare] tokenizer  : {args.tokenizer}")
    print(f"[prepare] seq len    : {args.max_seq_len}")

    raw = load_dataset("parquet", data_files=data_files, split="train")
    if args.max_samples > 0:
        raw = raw.select(range(min(args.max_samples, len(raw))))
        print(f"[prepare] debug rows : {len(raw)}")

    def tokenize_and_pack(batch):
        buffer: list[int] = []
        for text in batch["text"]:
            if not isinstance(text, str) or len(text.strip()) < args.min_chars:
                continue
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if ids:
                buffer.extend(ids)
                buffer.append(eos_id)

        usable = len(buffer) // args.max_seq_len * args.max_seq_len
        blocks = [
            buffer[i : i + args.max_seq_len]
            for i in range(0, usable, args.max_seq_len)
        ]
        return {"input_ids": blocks}

    packed = raw.map(
        tokenize_and_pack,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=raw.column_names,
        features=Features({"input_ids": Sequence(Value("uint16"))}),
        writer_batch_size=args.writer_batch_size,
        desc="tokenize and pack",
    )
    if args.max_tokens > 0:
        max_blocks = args.max_tokens // args.max_seq_len
        if max_blocks <= 0:
            raise ValueError("--max-tokens must be at least --max-seq-len")
        if len(packed) > max_blocks:
            packed = packed.select(range(max_blocks))
    packed.save_to_disk(str(args.output_dir))

    metadata = {
        "source": str(args.input_dir),
        "num_input_files": len(data_files),
        "num_rows": len(packed),
        "max_seq_len": args.max_seq_len,
        "tokenizer": str(args.tokenizer),
        "min_chars": args.min_chars,
        "file_limit": args.file_limit,
        "max_samples": args.max_samples,
        "max_tokens": args.max_tokens,
        "kept_tokens": len(packed) * args.max_seq_len,
    }
    (args.output_dir / "preprocess_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[prepare] packed rows: {len(packed):,}")
    print(f"[prepare] tokens     : {len(packed) * args.max_seq_len:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
