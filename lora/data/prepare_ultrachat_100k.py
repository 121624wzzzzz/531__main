#!/usr/bin/env python
"""Prepare a fixed UltraChat-200k subset for AffLoRA SFT validation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from datasets import load_dataset


ULTRACHAT_200K_PARQUET = {
    "train_sft": [
        "data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
        "data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
        "data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet",
    ],
    "test_sft": [
        "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--train-split", default="train_sft")
    parser.add_argument("--eval-split", default="test_sft")
    parser.add_argument(
        "--out-dir",
        default="/home/wz/projects/mypro/im_exp/lora/data/ultrachat_100k",
    )
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--eval-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-backend",
        choices=["modelscope", "hf", "hf-direct"],
        default="modelscope",
        help=(
            "modelscope uses swift/ultrachat_200k and is usually fastest in CN;"
            " hf uses datasets.load_dataset; hf-direct downloads parquet URLs first."
        ),
    )
    parser.add_argument("--modelscope-dataset-id", default="swift/ultrachat_200k")
    parser.add_argument(
        "--hf-base-url",
        default="https://hf-mirror.com/datasets/HuggingFaceH4/ultrachat_200k/resolve/main",
        help="Base URL for direct parquet loading. Used for the default UltraChat-200k dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before writing.",
    )
    return parser.parse_args()


def load_ultrachat_split(dataset_name: str, split: str, hf_base_url: str):
    if dataset_name == "HuggingFaceH4/ultrachat_200k" and split in ULTRACHAT_200K_PARQUET:
        urls = [f"{hf_base_url.rstrip('/')}/{name}" for name in ULTRACHAT_200K_PARQUET[split]]
        return load_dataset("parquet", data_files=urls, split="train")
    return load_dataset(dataset_name, split=split)


def ensure_ultrachat_parquet(split: str, hf_base_url: str, parquet_dir: Path) -> list[str]:
    parquet_dir.mkdir(parents=True, exist_ok=True)
    local_files: list[str] = []
    for rel_path in ULTRACHAT_200K_PARQUET[split]:
        url = f"{hf_base_url.rstrip('/')}/{rel_path}"
        target = parquet_dir / Path(rel_path).name
        if not target.exists() or target.stat().st_size == 0:
            print(f"[download] {url} -> {target}", flush=True)
            subprocess.run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--retry",
                    "5",
                    "--retry-delay",
                    "5",
                    "-C",
                    "-",
                    "-o",
                    str(target),
                    url,
                ],
                check=True,
            )
        local_files.append(str(target))
    return local_files


def load_split_from_local_parquet(split: str, hf_base_url: str, parquet_dir: Path):
    files = ensure_ultrachat_parquet(split, hf_base_url, parquet_dir)
    return load_dataset("parquet", data_files=files, split="train")


def load_modelscope_split(dataset_id: str, split: str):
    from modelscope.msdatasets import MsDataset

    return MsDataset.load(dataset_id, split=split)


def normalize_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages") or row.get("conversations")
    if not isinstance(messages, list):
        return []

    normalized: list[dict[str, str]] = []
    for turn in messages:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role") or turn.get("from")
        content = turn.get("content") or turn.get("value")
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        if role not in {"user", "assistant"}:
            continue
        text = str(content or "").strip()
        if not text:
            continue
        normalized.append({"role": role, "content": text})
    return normalized


def usable(row: dict[str, Any]) -> bool:
    roles = [turn["role"] for turn in normalize_messages(row)]
    return "user" in roles and "assistant" in roles


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> int:
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            conversations = normalize_messages(row)
            if not conversations:
                continue
            out = {
                "prompt": row.get("prompt", ""),
                "prompt_id": row.get("prompt_id", ""),
                "conversations": conversations,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
    return written


def sample_split(
    dataset_name: str,
    split: str,
    size: int,
    seed: int,
    hf_base_url: str,
    parquet_dir: Path,
    source_backend: str,
    modelscope_dataset_id: str,
) -> list[dict[str, Any]]:
    if source_backend == "modelscope":
        ds = load_modelscope_split(modelscope_dataset_id, split)
    elif source_backend == "hf-direct" and dataset_name == "HuggingFaceH4/ultrachat_200k" and split in ULTRACHAT_200K_PARQUET:
        ds = load_split_from_local_parquet(split, hf_base_url, parquet_dir)
    else:
        ds = load_ultrachat_split(dataset_name, split, hf_base_url)
    ds = ds.filter(usable, desc=f"Filtering {split}")
    if size > len(ds):
        raise ValueError(f"Requested {size} rows from {split}, but only {len(ds)} usable rows exist.")
    ds = ds.shuffle(seed=seed).select(range(size))
    return [dict(row) for row in ds]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} exists; pass --overwrite to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    train_rows = sample_split(
        args.dataset_name,
        args.train_split,
        args.train_size,
        args.seed,
        args.hf_base_url,
        out_dir / "parquet_cache",
        args.source_backend,
        args.modelscope_dataset_id,
    )
    eval_rows = sample_split(
        args.dataset_name,
        args.eval_split,
        args.eval_size,
        args.seed,
        args.hf_base_url,
        out_dir / "parquet_cache",
        args.source_backend,
        args.modelscope_dataset_id,
    )

    train_written = write_jsonl(train_rows, out_dir / "train.jsonl")
    eval_written = write_jsonl(eval_rows, out_dir / "eval.jsonl")

    metadata = {
        "dataset_name": args.dataset_name,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "hf_base_url": args.hf_base_url,
        "source_backend": args.source_backend,
        "modelscope_dataset_id": args.modelscope_dataset_id,
        "seed": args.seed,
        "train_size": train_written,
        "eval_size": eval_written,
        "out_dir": str(out_dir),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
