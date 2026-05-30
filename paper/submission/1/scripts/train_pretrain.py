#!/usr/bin/env python3
"""Minimal AGD pretraining loop on tokenized JSONL data.

Expected JSONL format (one sample per line):
  {"input_ids": [1, 2, 3, ...]}

For full MiniMind + FineWeb-Edu reproduction, see the upstream training scripts
referenced in README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agd import AGDConfig, AGDCausalLM, VARIANTS


class JsonlTokenDataset(Dataset):
    def __init__(self, path: Path, max_length: int):
        self.samples: list[list[int]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                ids = row.get("input_ids") or row.get("text_ids")
                if not ids:
                    continue
                self.samples.append(ids[:max_length])
        if not self.samples:
            raise ValueError(f"No usable samples found in {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.samples[idx]
        x = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": x, "labels": x.clone()}


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids, labels = [], []
    for item in batch:
        pad = max_len - item["input_ids"].shape[0]
        input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad), value=0))
        labels.append(torch.nn.functional.pad(item["labels"], (0, pad), value=-100))
    return {"input_ids": torch.stack(input_ids), "labels": torch.stack(labels)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AGD causal LM variant.")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default="s12", choices=sorted(VARIANTS.keys()))
    parser.add_argument("--variant-rank", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=6400)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = AGDConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        max_position_embeddings=args.max_length,
        variant=args.variant,
        variant_rank=args.variant_rank,
    )
    model = AGDCausalLM(cfg).to(args.device)
    budget = model.parameter_budget()
    print(f"[budget] variant={args.variant} {budget}")

    dataset = JsonlTokenDataset(args.train_data, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out["loss"]
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            global_step += 1
            if global_step % 20 == 0:
                print(f"epoch={epoch+1} step={global_step} loss={loss.item():.4f}")

    ckpt = {
        "config": cfg.__dict__,
        "state_dict": model.state_dict(),
        "parameter_budget": budget,
    }
    out_path = args.output_dir / f"agd_{args.variant}.pt"
    torch.save(ckpt, out_path)
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"variant": args.variant, "steps": global_step, **budget}, f, indent=2)
    print(f"Saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
