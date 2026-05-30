"""Compare signal fidelity: input-side vs output-side perturbation.

The right question for S6 vs S7:
  When P,Q add perturbation δ_t to embedding at EVERY timestep (S6),
  does δ_t's contribution to logits survive the LSTM gating?

We measure:
  1. Input-side: add δ_t at each embedding, measure ∂logits/∂δ_t (Jacobian)
  2. Output-side: add δ to output weight, measure ∂logits/∂δ (Jacobian)
  3. Compare the effective gradient magnitude: which side transmits signal better?

Usage:
    PY=/home/wz/anaconda3/envs/torch24/bin/python
    $PY scripts/measure_signal_fidelity.py \
        --ckpt runs/_analysis/large__s7__r32__s0p03__papertest__tf32__seed1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "repro_pytorch"))

from configs import PTBConfig
from model import build_model
from ptb_data import ptb_raw_data, PTBBatchedSplit
from variants import EmbeddingVariant


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg_dict = {k: v for k, v in ckpt["config"].items()
                if k in PTBConfig.__dataclass_fields__}
    config = PTBConfig(**cfg_dict)
    model = build_model(
        config, ckpt["args"]["variant"],
        ckpt["architecture"],
        int(ckpt["args"]["lora_rank"]),
        float(ckpt["args"]["relaxation_scale"]),
    )
    model.load_state_dict(ckpt["model_state"])
    return model, config


def make_s6_model(model, config):
    """Create S6 variant: same as S7 but perturbation on INPUT side.
    Returns a new EmbeddingVariant module with S6 behavior."""
    emb = model.embeddings
    s6_emb = EmbeddingVariant(
        config.vocab_size, config.hidden_size, "s6",
        emb.rank, emb.init_scale, emb.relaxation_scale,
    )
    # Copy W from S7
    s6_emb.input_weight.data.copy_(emb.input_weight.data)
    s6_emb.output_bias.data.copy_(emb.output_bias.data)
    # Copy P,Q to input side
    s6_emb.input_mult_p.data.copy_(emb.output_mult_p.data)
    s6_emb.input_mult_q.data.copy_(emb.output_mult_q.data)
    return s6_emb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, default=_repo / "data" / "ptb")
    parser.add_argument("--num_steps", type=int, default=35)
    parser.add_argument("--num_batches", type=int, default=100)
    args = parser.parse_args()

    model, config = load_model(args.ckpt)
    model.eval()

    # Build S6 version with same P,Q,W
    s6_emb = make_s6_model(model, config)

    # Load test data
    _, _, test_data, _ = ptb_raw_data(str(args.data_path))
    test_cache = PTBBatchedSplit(
        test_data, batch_size=1, num_steps=args.num_steps,
        device=torch.device("cpu"),
    )

    d = config.hidden_size
    V = config.vocab_size
    s = model.embeddings.relaxation_scale
    P = model.embeddings.output_mult_p.detach().clone()  # [d, r]
    Q = model.embeddings.output_mult_q.detach().clone()  # [r, d]

    # === Experiment: measure d(logits)/d(P,Q) for input vs output side ===

    # We compare how much the final logits change when we make a SMALL
    # perturbation to P (or Q), placed on the INPUT side vs OUTPUT side.
    # The ratio tells us: does the LSTM amplify or attenuate the signal?

    input_gains = []
    output_gains = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_cache):
            if batch_idx >= args.num_batches:
                break

            # --- Output-side gain (S7): perturb P, measure logit change ---
            # dW' = s * W * dP * Q,  dLogits = h @ dW'^T
            hidden = model.init_hidden(1, torch.device("cpu"))
            emb = model.embeddings.input_embeddings(inputs)
            emb_out = model.drop(emb)
            h_raw, _ = model.lstm(emb_out, hidden)
            h = model.drop(h_raw).squeeze(0)  # [T, d]
            W = model.embeddings.input_weight  # [V, d]

            # For each timestep, compute ||d(logits)/d(P_ij)||
            # d(logits)/dP = s * W^T * h * Q^T
            # Jacobian is [V, d, r] — too big. Use random projection.
            for t in range(args.num_steps):
                ht = h[t]  # [d]
                # Random direction in P-space
                dP = torch.randn(d, model.embeddings.rank) * 0.01
                dP = dP / dP.norm() * 0.01
                # Effect on output weight: dW' = s * W @ dP @ Q
                dW = s * (W.float() @ dP @ Q.float())  # [V, d]
                # Effect on logits at step t: dLogits = ht @ dW^T
                dLogits = ht.float() @ dW.T  # [V]
                output_gains.append(float(dLogits.norm().item()))

            # --- Input-side gain (S6): perturb P, measure logit change ---
            # embed' = embed + s * embed @ dP @ Q
            # Then this passes through LSTM → h' → logits
            hidden = model.init_hidden(1, torch.device("cpu"))
            emb = model.embeddings.input_embeddings(inputs)

            # Baseline: no perturbation
            h_base = []
            hidden_base = model.init_hidden(1, torch.device("cpu"))
            for t in range(args.num_steps):
                et = emb[:, t, :]  # [1, d]
                et_out = model.drop(et)
                ht, hidden_base = model.lstm(et_out.unsqueeze(1), hidden_base)
                h_base.append(model.drop(ht).squeeze())

            # Perturbation at EVERY timestep (matching S6 behavior)
            dP = torch.randn(d, model.embeddings.rank) * 0.01
            dP = dP / dP.norm() * 0.01
            hidden_pert = model.init_hidden(1, torch.device("cpu"))
            for t in range(args.num_steps):
                et = emb[:, t, :]  # [1, d]
                # S6 perturbation: e' = e + s * e @ dP @ Q
                et_pert = et + s * (et.float() @ dP @ Q.float()).to(et.dtype)
                et_out = model.drop(et_pert)
                ht, hidden_pert = model.lstm(et_out.unsqueeze(1), hidden_pert)
                ht_out = model.drop(ht).squeeze()

                # Effect on logits at step t
                dLogits_input = ht_out.float() @ W.float().T - h_base[t].float() @ W.float().T
                input_gains.append(float(dLogits_input.norm().item()))

    # --- Results ---
    out_mean = np.mean(output_gains)
    in_mean = np.mean(input_gains)
    ratio = in_mean / out_mean if out_mean > 0 else 0

    print("=" * 60)
    print("Signal Fidelity: Input-side vs Output-side")
    print("=" * 60)
    print(f"  Samples: {len(output_gains)} timesteps × {args.num_batches} batches")
    print(f"  Output-side mean gain:  {out_mean:.4f}")
    print(f"  Input-side mean gain:   {in_mean:.4f}")
    print(f"  Input/Output ratio:     {ratio:.4f} ({ratio*100:.1f}%)")
    print()
    if ratio < 0.1:
        print(f"  ★ LSTM attenuates input-side signal to {ratio*100:.1f}% of output-side")
        print(f"    → Strong evidence for gating bottleneck")
    elif ratio < 0.5:
        print(f"  △ LSTM partially attenuates input-side signal ({ratio*100:.0f}%)")
        print(f"    → Moderate evidence for gating bottleneck")
    else:
        print(f"  ✗ LSTM does NOT significantly attenuate input-side signal")
        print(f"    → Gating bottleneck hypothesis is WRONG")

    # Bonus: per-timestep breakdown
    print()
    print("--- Per-timestep input/output gain ratio ---")
    io_input = np.array(input_gains).reshape(args.num_batches, args.num_steps)
    io_output = np.array(output_gains).reshape(args.num_batches, args.num_steps)
    for t in [0, 1, 2, 5, 10, 20, 34]:
        inp_t = io_input[:, t].mean()
        out_t = io_output[:, t].mean()
        r_t = inp_t / out_t if out_t > 0 else 0
        print(f"  step {t:>2d}:  ratio = {r_t:.4f} ({r_t*100:.1f}%)")

    return 0


if __name__ == "__main__":
    main()
