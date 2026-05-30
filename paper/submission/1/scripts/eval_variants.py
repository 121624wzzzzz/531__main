#!/usr/bin/env python3
"""Evaluate AGD variants against tied (s1) and full-untied (s2) baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agd import AGDConfig, AGDCausalLM, VARIANTS
from train_pretrain import JsonlTokenDataset, collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        default="s1,s2,s3,s6,s12",
        help="Comma-separated AGD variant names.",
    )
    parser.add_argument("--variant-rank", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=6400)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def eval_variant(model: AGDCausalLM, loader: DataLoader, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch)["loss"]
        total += float(loss.item())
        count += 1
    return total / max(count, 1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = JsonlTokenDataset(args.eval_data, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    rows: list[dict] = []
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        if name not in VARIANTS:
            raise ValueError(f"Unknown variant {name}")
        cfg = AGDConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            max_position_embeddings=args.max_length,
            variant=name,
            variant_rank=args.variant_rank,
        )
        model = AGDCausalLM(cfg).to(args.device)
        loss = eval_variant(model, loader, args.device)
        budget = model.parameter_budget()
        rows.append(
            {
                "variant": name,
                "description": VARIANTS[name].description,
                "eval_loss": loss,
                "agd_extra_params": budget["agd_extra_params"],
                "full_untie_overhead": budget["full_untie_overhead"],
            }
        )
        print(f"{name}: loss={loss:.4f} extra={budget['agd_extra_params']}")

    s1 = next((r["eval_loss"] for r in rows if r["variant"] == "s1"), None)
    s2 = next((r["eval_loss"] for r in rows if r["variant"] == "s2"), None)
    for row in rows:
        row["delta_vs_s1"] = None if s1 is None else row["eval_loss"] - s1
        row["delta_vs_s2"] = None if s2 is None else row["eval_loss"] - s2

    csv_path = args.output_dir / "variant_eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "variant_eval.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
