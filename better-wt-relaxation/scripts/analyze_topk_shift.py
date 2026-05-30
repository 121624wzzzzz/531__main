"""Compare S7 top-K predictions with and without the adapter.

For each token in the PTB test set, compute the hidden state h and compare
the top-K predicted words WITH the S7 adapter (W' = W + s·W·P·Q) vs WITHOUT
(plain W).  Classify the change as:
  - "meaningful refinement" — new top-K words include semantically-relevant
    low-frequency words that the plain model missed
  - "high-freq shuffling"  — the change just swaps among high-frequency words

Usage:
    PY=/home/wz/anaconda3/envs/torch24/bin/python
    $PY scripts/analyze_topk_shift.py --checkpoint runs/_analysis/large__s7__r32__s0p03__papertest__tf32__seed1
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

_repo = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(_repo / "repro_pytorch"))
sys.path.insert(0, str(_repo / "scripts"))

from configs import PTBConfig, CONFIGS
from model import build_model
from ptb_data import ptb_raw_data, PTBBatchedSplit
from variants import EmbeddingVariant

PTB_PATH = _repo / "data" / "ptb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_and_params(checkpoint: Path):
    """Load a trained S7 model and extract the adapter parameters."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg_dict = {k: v for k, v in ckpt["config"].items()
                if k in PTBConfig.__dataclass_fields__}
    config = PTBConfig(**cfg_dict)
    model = build_model(config, ckpt["args"]["variant"],
                        ckpt["architecture"],
                        int(ckpt["args"]["lora_rank"]),
                        float(ckpt["args"]["relaxation_scale"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    emb = model.embeddings
    W = emb.input_weight.detach().float()          # [V, d]
    s = emb.relaxation_scale
    P = emb.output_mult_p.detach().float()         # [d, r]
    Q = emb.output_mult_q.detach().float()         # [r, d]
    delta = s * W @ P @ Q                           # [V, d]
    W_prime = W + delta                             # [V, d]
    return model, config, W, W_prime, delta


def word_frequencies() -> Tuple[Dict[int, int], List[int]]:
    """Return (token_id → count, sorted token_ids by frequency)."""
    from ptb_data import build_vocab, file_to_word_ids
    train_path = PTB_PATH / "ptb.train.txt"
    w2id = build_vocab(train_path)
    ids = file_to_word_ids(train_path, w2id)
    counts = dict(Counter(ids))
    sorted_by_freq = sorted(counts, key=counts.get)  # rare → frequent
    return counts, sorted_by_freq


def freq_percentile(wid: int, counts: Dict[int, int], sorted_ids: List[int]) -> int:
    """Return 0–99 percentile of this word's frequency."""
    if wid not in counts:
        return 0
    return int(np.searchsorted(sorted_ids, wid) / len(sorted_ids) * 100)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_topk_shift(
    model, config,
    W: torch.Tensor,
    W_prime: torch.Tensor,
    counts: Dict[int, int],
    sorted_ids: List[int],
    K: int = 20,
    max_samples: int = 5000,
) -> dict:
    """Compare top-K predictions with vs without the S7 adapter."""
    # Load test data
    _, _, test_data, _ = ptb_raw_data(str(PTB_PATH))
    batch_size = 1
    test_cache = PTBBatchedSplit(test_data, batch_size, config.num_steps,
                                  device=torch.device("cpu"))

    results = {
        "total_tokens": 0,
        "top1_changed": 0,
        "topK_changed": 0,
        "by_freq_bucket": defaultdict(lambda: {
            "count": 0, "top1_changed": 0, "topK_changed": 0,
            "new_tokens_mean_rank": 0.0, "new_tokens_mean_freq_percentile": 0.0,
        }),
        "examples": [],  # store up to 20 interesting cases
    }

    with torch.no_grad():
        hidden = model.init_hidden(batch_size, torch.device("cpu"))
        for inputs, targets in test_cache:
            if results["total_tokens"] >= max_samples:
                break

            # --- Forward pass to get hidden state h ---
            emb_raw = model.embeddings.input_embeddings(inputs)
            emb = model.drop(emb_raw)
            h_raw, hidden = model.lstm(emb, hidden)
            h = model.drop(h_raw).squeeze(0)  # [num_steps, d]
            hidden = (hidden[0].detach(), hidden[1].detach())  # detach for BPTT boundary

            # Process each timestep individually
            num_steps = h.shape[0]
            for t in range(num_steps):
                if results["total_tokens"] >= max_samples:
                    break
                ht = h[t]  # [d]
                actual = int(targets[0, t].item())

                # --- Logits with vs without adapter ---
                logits_with = ht @ W_prime.T + model.embeddings.output_bias  # [V]
                logits_without = ht @ W.T + model.embeddings.output_bias     # [V]

                topk_with = torch.topk(logits_with, K).indices.tolist()
                topk_without = torch.topk(logits_without, K).indices.tolist()

                results["total_tokens"] += 1

                if topk_with[0] != topk_without[0]:
                    results["top1_changed"] += 1

                new_in_topk = set(topk_with) - set(topk_without)
                if new_in_topk:
                    results["topK_changed"] += 1

                # Frequency bucket
                pct = freq_percentile(actual, counts, sorted_ids)
                if pct < 20: bucket = "0-20% (rarest)"
                elif pct < 50: bucket = "20-50%"
                elif pct < 80: bucket = "50-80%"
                else: bucket = "80-100% (most freq)"

                b = results["by_freq_bucket"][bucket]
                b["count"] += 1
                if topk_with[0] != topk_without[0]:
                    b["top1_changed"] += 1
                if new_in_topk:
                    b["topK_changed"] += 1
                    for tid in new_in_topk:
                        rank_in_with = topk_with.index(tid) + 1
                        b["new_tokens_mean_rank"] += rank_in_with
                        b["new_tokens_mean_freq_percentile"] += freq_percentile(tid, counts, sorted_ids)

                # Save interesting examples
                if len(results["examples"]) < 20 and (topk_with[0] != topk_without[0]
                        or pct < 30):
                    results["examples"].append({
                        "actual_token_id": actual,
                        "actual_freq_pct": pct,
                        "top1_without": topk_without[0],
                        "top1_with": topk_with[0],
                        "new_in_top5": list(set(topk_with[:5]) - set(topk_without[:5])),
                        "dropped_from_top5": list(set(topk_without[:5]) - set(topk_with[:5])),
                    })

    # Normalize bucket averages (running sums → means)
    for b in results["by_freq_bucket"].values():
        if b["topK_changed"] > 0:
            b["new_tokens_mean_rank"] /= b["topK_changed"]
            b["new_tokens_mean_freq_percentile"] /= b["topK_changed"]

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to S7 .pt checkpoint")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Max tokens to analyze (default 5000)")
    parser.add_argument("--K", type=int, default=20,
                        help="Top-K size (default 20)")
    args = parser.parse_args()

    print("Loading model ...")
    model, config, W, W_prime, delta = load_model_and_params(args.checkpoint)
    counts, sorted_ids = word_frequencies()
    print(f"Vocab={len(counts)}  test tokens to analyze={args.max_samples}")

    print("Running top-K shift analysis ...")
    results = analyze_topk_shift(
        model, config, W, W_prime, counts, sorted_ids,
        K=args.K, max_samples=args.max_samples,
    )

    total = results["total_tokens"]
    print(f"\n{'='*60}")
    print(f"Analyzed {total} tokens  (K={args.K})")
    print(f"{'='*60}")
    print(f"  Top-1 changed:  {results['top1_changed']}/{total} "
          f"({results['top1_changed']/total*100:.1f}%)")
    print(f"  Top-K changed:  {results['topK_changed']}/{total} "
          f"({results['topK_changed']/total*100:.1f}%)")
    print()

    print(f"{'Frequency bucket':<22s} {'tokens':>7s} {'top1Δ%':>7s} {'topKΔ%':>7s} "
          f"{'new_avg_rank':>12s} {'new_freq_pct':>12s}")
    print(f"{'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*12} {'-'*12}")
    for bucket in ["0-20% (rarest)", "20-50%", "50-80%", "80-100% (most freq)"]:
        b = results["by_freq_bucket"][bucket]
        if b["count"] == 0:
            continue
        t1 = b["top1_changed"] / b["count"] * 100
        tk = b["topK_changed"] / b["count"] * 100
        avg_rank = b["new_tokens_mean_rank"]
        avg_pct = b["new_tokens_mean_freq_percentile"]
        print(f"  {bucket:<20s} {b['count']:>7d} {t1:>6.1f}% {tk:>6.1f}% "
              f"{avg_rank:>11.1f} {avg_pct:>11.1f}")

    # Overall: are the newly-introduced tokens more or less frequent than the actual tokens?
    print()
    print("--- Interpretation ---")
    # Collect all new tokens' frequency percentiles
    all_actual_pct = []
    all_new_pct = []
    # We'll recompute from examples for simplicity
    # ... (skip detailed recompute — use bucket-level averages)
    rarest_b = results["by_freq_bucket"]["0-20% (rarest)"]
    freq_b = results["by_freq_bucket"]["80-100% (most freq)"]
    if rarest_b["count"] > 0 and freq_b["count"] > 0:
        rr = rarest_b["topK_changed"] / rarest_b["count"] * 100
        fr = freq_b["topK_changed"] / freq_b["count"] * 100
        if rr > fr * 1.5:
            print(f"  Adapter affects RARE tokens {rr:.0f}% vs frequent {fr:.0f}% "
                  f"— suggests meaningful semantic refinement")
        elif fr > rr * 1.5:
            print(f"  Adapter affects FREQUENT tokens {fr:.0f}% vs rare {rr:.0f}% "
                  f"— suggests high-frequency shuffling")
        else:
            print(f"  Adapter affects all frequency ranges similarly "
                  f"(rare {rr:.0f}% / freq {fr:.0f}%)")

    print()
    print("--- Example predictions (top1 changed or rare tokens) ---")
    for ex in results["examples"][:10]:
        print(f"  actual_token={ex['actual_token_id']} "
              f"(freq pct={ex['actual_freq_pct']})")
        print(f"    without S7 top1: {ex['top1_without']}")
        print(f"    with S7    top1: {ex['top1_with']}")
        if ex["new_in_top5"]:
            print(f"    new in top5:     {ex['new_in_top5']}")
        if ex["dropped_from_top5"]:
            print(f"    dropped from top5: {ex['dropped_from_top5']}")
        print()


if __name__ == "__main__":
    main()
