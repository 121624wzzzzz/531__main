#!/usr/bin/env python3
"""Fit full-vocabulary affine maps between base and instruct matrices."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from affine_fit import full_affine_stream


def load_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    from safetensors.numpy import load_file

    tensors = load_file(str(path))
    if len(tensors) == 1:
        return np.asarray(next(iter(tensors.values())), dtype=np.float32)
    for key in ("model.lm_head.weight", "model.embed_tokens.weight", "weight"):
        if key in tensors:
            return np.asarray(tensors[key], dtype=np.float32)
    raise KeyError(f"Could not infer matrix key in {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-vocab affine fit X->Y.")
    parser.add_argument("--base-matrix", type=Path, required=True)
    parser.add_argument("--instruct-matrix", type=Path, required=True)
    parser.add_argument("--role", default="lm_head", choices=["embed_tokens", "lm_head", "shared"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/affine_fit"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    X = load_matrix(args.base_matrix)
    Y = load_matrix(args.instruct_matrix)
    if X.shape != Y.shape:
        raise ValueError(f"Matrix shape mismatch: {X.shape} vs {Y.shape}")

    import torch

    row = full_affine_stream(X, Y, device=torch.device(args.device))
    row["role"] = args.role
    row["base_matrix"] = str(args.base_matrix)
    row["instruct_matrix"] = str(args.instruct_matrix)

    json_path = args.output_dir / f"affine_fit_{args.role}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    csv_path = args.output_dir / f"affine_fit_{args.role}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2))
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
