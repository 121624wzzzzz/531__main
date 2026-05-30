#!/usr/bin/env python3
"""Fine-tune a causal LM with A-LoRA at the vocabulary boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alora import ALoraConfig, apply_alora_adapters, save_alora_adapter, trainable_parameter_count


class ToySFTDataset(Dataset):
    """Minimal instruction dataset for smoke tests."""

    def __init__(self, vocab_size: int, length: int, num_samples: int):
        self.samples = [
            torch.randint(1, vocab_size, (length,), dtype=torch.long) for _ in range(num_samples)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.samples[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}


class TinyCausalLM(torch.nn.Module):
    """Toy tied LM for demonstrating A-LoRA without downloading weights."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.transform = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
        )

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        hidden = self.transform(self.embed_tokens(input_ids))
        logits = self.lm_head(hidden)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return out

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, module):
        self.embed_tokens = module


VARIANT_TO_CFG = {
    "affine_input": dict(use_input=True, use_lm_head=False, use_input_bias=True),
    "affine_lm_head": dict(use_input=False, use_lm_head=True, use_lm_head_bias=False),
    "affine_input_lm_head": dict(use_input=True, use_lm_head=True, use_input_bias=True, use_lm_head_bias=False),
    "affine_shared_tied": dict(
        use_input=True,
        use_lm_head=True,
        use_input_bias=True,
        use_lm_head_bias=True,
        tie_input_lm_head_adapters=True,
        use_tied_lm_head_transpose=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_TO_CFG.keys()),
        default="affine_input",
    )
    parser.add_argument("--model-path", type=Path, default=None, help="HF model path (optional).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--hidden-size", type=int, default=896)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_hf_model(path: Path, device: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.bfloat16)
    return model.to(device)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_path:
        model = load_hf_model(args.model_path, args.device)
        hidden_size = model.get_input_embeddings().embedding_dim
        tied = model.get_input_embeddings().weight.data_ptr() == model.lm_head.weight.data_ptr()
    else:
        model = TinyCausalLM(args.vocab_size, args.hidden_size).to(args.device)
        hidden_size = args.hidden_size
        tied = True

    cfg = ALoraConfig(hidden_size=hidden_size, rank=args.rank, alpha=args.alpha, **VARIANT_TO_CFG[args.variant])
    apply_alora_adapters(model, cfg)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"variant={args.variant} trainable_params={sum(p.numel() for p in trainable)} "
        f"budget_estimate={trainable_parameter_count(cfg, tied_vocab=tied)}"
    )

    dataset = ToySFTDataset(args.vocab_size if not args.model_path else 512, length=32, num_samples=256)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optim = torch.optim.AdamW(trainable, lr=args.lr)

    model.train()
    step = 0
    for batch in loader:
        if step >= args.steps:
            break
        batch = {k: v.to(args.device) for k, v in batch.items()}
        loss = model(**batch)["loss"]
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        step += 1
        if step % 20 == 0:
            print(f"step={step} loss={loss.item():.4f}")

    save_alora_adapter(model, args.output_dir)
    meta = {
        "variant": args.variant,
        "rank": args.rank,
        "alpha": args.alpha,
        "trainable_params": sum(p.numel() for p in trainable),
        "steps": step,
    }
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
