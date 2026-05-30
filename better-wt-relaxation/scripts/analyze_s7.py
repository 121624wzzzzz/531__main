"""Analyze what the S7 output-multiplicative low-rank adapter (PQ) learned.

Four analyses on the best-stability S7 config (r=32, s=0.03):
  1. SVD of PQ        — effective rank & singular value spectrum
  2. Word-class correction — per-frequency-bin correction magnitude
  3. Untying gap       — does s·WPQ approximate the untied model's U−W?
  4. Cross-seed consistency — do PQ subspaces converge across seeds?

Prerequisites (run first — ~3.5 h on 6×A100):
    # Train S1 (WT), S2 (untied), S7 (r=32, s=0.03) × 5 seeds each
    cd /home/wz/projects/mypro/im_exp/UsingTheOutputEmbedding-repro

    PY=/home/wz/anaconda3/envs/torch24/bin/python
    for seed in 1 2 3 4 5; do
      for spec in "s1:8:0.1" "s2:1:1.0" "s7:32:0.03"; do
        v=${spec%%:*}; r_scale=${spec#*:}; r=${r_scale%%:*}; s=${r_scale#*:}
        $PY repro_pytorch/train_ptb.py \
          --data_path data/ptb --model large --variant $v \
          --lora_rank $r --relaxation_scale $s --seed $seed \
          --tf32 --paper_test_eval --device cuda:0 \
          --output_dir runs/_analysis/${v}_r${r}_s${s}_seed${seed}
      done
    done

Usage:
    PY=/home/wz/anaconda3/envs/torch24/bin/python
    $PY scripts/analyze_s7.py --runs_dir runs/_analysis [--seeds 5]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Insert repro_pytorch so we can import project modules.
_repo = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(_repo / "repro_pytorch"))

from variants import EmbeddingVariant, HiddenState
from configs import PTBConfig

# Matplotlib is optional — only needed for --plot.
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PTB_PATH = _repo / "data" / "ptb"
PTB_VOCAB = 10000
PTB_HIDDEN = 1500


def _scale_tag(scale: float) -> str:
    text = f"{scale:.6g}"
    return text.replace("-", "m").replace(".", "p")


def find_checkpoint(runs_dir: Path, variant: str, rank: int, scale: float, seed: int) -> Path:
    """Locate a .pt checkpoint under *runs_dir* by matching output_dir naming."""
    stag = _scale_tag(scale)
    # Try both s-number and legacy name (s2 ↔ baseline, s1 ↔ wt)
    candidates = {variant}
    alias_map = {"s1": "wt", "s2": "baseline", "wt": "s1", "baseline": "s2",
                 "wt_pr": "s1", "pr": "s2"}
    if variant in alias_map:
        candidates.add(alias_map[variant])
    for try_v in candidates:
        pattern = f"__{try_v}__r{rank}__s{stag}"
        for pt in sorted(runs_dir.rglob("*.pt")):
            if pattern in str(pt.parent) and f"seed{seed}" in str(pt.parent):
                return pt
    raise FileNotFoundError(
        f"No checkpoint found for {variant} r={rank} s={scale} seed={seed}"
    )


def load_variant_params(pt_path: Path) -> Dict[str, torch.Tensor]:
    """Load a checkpoint and return only the EmbeddingVariant's parameters."""
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
    state = ckpt["model_state"]
    return {k.replace("embeddings.", ""): v for k, v in state.items() if k.startswith("embeddings.")}


def svd_spectrum(M: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Return (singular_values, cumulative_energy_ratio) for matrix M."""
    U, S, Vh = np.linalg.svd(M.cpu().float().numpy(), full_matrices=False)
    total = float((S ** 2).sum())
    cum = np.cumsum(S ** 2) / max(total, 1e-30)
    return S, cum


def subspace_angles(A: torch.Tensor, B: torch.Tensor) -> dict:
    """Principal angles (degrees) between the column spaces of A and B.
    Returns min, max, and mean angle across all canonical directions."""
    A_np = A.cpu().float().numpy()
    B_np = B.cpu().float().numpy()
    Qa, _ = np.linalg.qr(A_np)
    Qb, _ = np.linalg.qr(B_np)
    # S = cosines of principal angles, in *descending* order.
    # The first principal angle is the *smallest* angle (largest cosine).
    S_full = np.linalg.svd(Qa.T @ Qb, full_matrices=False, compute_uv=False)
    S = np.clip(S_full, 0.0, 1.0)
    angles = np.arccos(S) * 180.0 / math.pi
    return {
        "min_deg": float(angles[0]),           # closest alignment
        "max_deg": float(angles[-1]),           # farthest alignment
        "mean_deg": float(np.mean(angles)),
        "median_deg": float(np.median(angles)),
        "cosines_top5": S[:5].tolist(),
        "cosines_bottom5": S[-5:].tolist() if len(S) >= 5 else S.tolist(),
    }


def ptb_word_frequencies() -> Dict[int, int]:
    """Count token frequencies in PTB training set."""
    from collections import Counter
    train_path = PTB_PATH / "ptb.train.txt"
    text = train_path.read_text(encoding="utf-8").replace("\n", "<eos>")
    from ptb_data import build_vocab, file_to_word_ids
    w2id = build_vocab(train_path)
    ids = file_to_word_ids(train_path, w2id)
    return dict(Counter(ids))


# ---------------------------------------------------------------------------
# Analysis 1: SVD of PQ
# ---------------------------------------------------------------------------

def analyze_svd(params: Dict[str, torch.Tensor], rank: int) -> dict:
    """SVD of the learned P @ Q matrix."""
    P = params["output_mult_p"]  # [d, r]
    Q = params["output_mult_q"]  # [r, d]
    PQ = P @ Q  # [d, d]
    S, cum = svd_spectrum(PQ)

    effective_rank = int(np.sum(cum < 0.95)) + 1  # rank needed for 95% energy
    return {
        "singular_values_top10": S[:10].tolist(),
        "singular_values_all": S.tolist(),
        "effective_rank_95pct": effective_rank,
        "top1_energy_ratio": float(cum[0]) if len(cum) > 0 else 0.0,
        "top4_energy_ratio": float(cum[3]) if len(cum) > 3 else float(cum[-1]),
        "condition_number": float(S[0] / max(S[-1], 1e-30)),
    }


# ---------------------------------------------------------------------------
# Analysis 2: Word-class correction magnitude
# ---------------------------------------------------------------------------

def analyze_word_correction(
    pt_path: Path, params: Dict[str, torch.Tensor], rank: int, scale: float
) -> dict:
    """Measure how much the PQ adapter shifts the hidden state,
    grouped by word frequency percentile.
    """
    from ptb_data import ptb_raw_data, PTBBatchedSplit
    from model import build_model
    from configs import CONFIGS

    # Build model and load full state
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
    config = PTBConfig(**{k: v for k, v in ckpt["config"].items()
                           if k in PTBConfig.__dataclass_fields__})
    model = build_model(config, "s7", config.architecture, rank, scale)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load test data
    _, _, test_data, vocab_size = ptb_raw_data(str(PTB_PATH))
    test_cache = PTBBatchedSplit(test_data, batch_size=1, num_steps=1, device=torch.device("cpu"))

    # Word frequencies
    freqs = ptb_word_frequencies()
    # Partition into 5 buckets by frequency percentile
    sorted_freqs = sorted(freqs.values())
    n = len(sorted_freqs)
    boundaries = [sorted_freqs[int(n * p)] for p in [0.2, 0.4, 0.6, 0.8]]
    boundaries = sorted(set(boundaries))

    def freq_bucket(wid: int) -> str:
        f = freqs.get(wid, 0)
        if f <= boundaries[0]: return "very_rare"
        if f <= boundaries[1]: return "rare"
        if f <= boundaries[2]: return "mid"
        if f <= boundaries[3]: return "frequent"
        return "very_frequent"

    # Compute per-token correction
    hidden_corrections: Dict[str, List[float]] = defaultdict(list)
    with torch.no_grad():
        hidden = model.init_hidden(1, torch.device("cpu"))
        for inputs, targets in test_cache:
            emb = model.embeddings.input_embeddings(inputs)
            # Get hidden state WITH and WITHOUT the adapter
            # We need the intermediate hidden state h before output_logits
            h_raw, hidden = model.lstm(model.drop(emb), hidden)
            h = model.drop(h_raw).squeeze(0).squeeze(0)  # [d]

            # Correction = s * h @ Q^T @ P^T
            P = params["output_mult_p"]  # [d, r]
            Q = params["output_mult_q"]  # [r, d]
            correction = scale * (h @ Q.T @ P.T)  # [d]
            corr_norm = float(torch.norm(correction).item())
            h_norm = float(torch.norm(h).item())

            for t in range(inputs.shape[1]):
                wid = int(inputs[0, t].item())
                bucket = freq_bucket(wid)
                hidden_corrections[bucket].append(corr_norm / max(h_norm, 1e-8))

    return {
        bucket: {
            "mean_rel_correction": float(np.mean(vals)),
            "std_rel_correction": float(np.std(vals)),
            "n_tokens": len(vals),
        }
        for bucket, vals in sorted(hidden_corrections.items())
    }


# ---------------------------------------------------------------------------
# Analysis 3: Untying gap
# ---------------------------------------------------------------------------

def analyze_untying_gap(s7_params: Dict, s2_params: Dict, scale: float) -> dict:
    """Compare the S7 perturbation s·W·P·Q to the untied gap U − W."""
    W_s7 = s7_params["input_weight"]    # [V, d]
    P = s7_params["output_mult_p"]      # [d, r]
    Q = s7_params["output_mult_q"]      # [r, d]
    W_s2 = s2_params["input_weight"]    # E (input embedding in untied)
    U_s2 = s2_params["output_weight"]   # U (output softmax weight in untied)

    # S7 perturbation: s * W @ P @ Q  [V, d]
    delta_s7 = scale * (W_s7 @ P @ Q)

    # Untied gap: U - W  (note: in untied model W=E, U is separate)
    gap_untied = U_s2 - W_s2  # [V, d]

    # Align delta_s7 to gap_untied via a global scalar fit
    delta_flat = delta_s7.flatten().float()
    gap_flat = gap_untied.flatten().float()

    # Cosine similarity
    cos_sim = float(torch.nn.functional.cosine_similarity(
        delta_flat.unsqueeze(0), gap_flat.unsqueeze(0)).item())

    # Optimal scalar: argmin_a || gap - a * delta ||^2
    a_opt = float((gap_flat @ delta_flat) / max(delta_flat @ delta_flat, 1e-30))
    residual = gap_flat - a_opt * delta_flat
    explained_var = float(1.0 - (residual @ residual) / max(gap_flat @ gap_flat, 1e-30))

    # Frobenius norms
    norm_delta = float(torch.norm(delta_s7).item())
    norm_gap = float(torch.norm(gap_untied).item())

    return {
        "cosine_similarity": cos_sim,
        "explained_variance_ratio": explained_var,
        "optimal_scalar_a": a_opt,
        "frob_norm_delta_s7": norm_delta,
        "frob_norm_untied_gap": norm_gap,
        "norm_ratio_delta_over_gap": norm_delta / max(norm_gap, 1e-30),
    }


# ---------------------------------------------------------------------------
# Analysis 4: Cross-seed consistency
# ---------------------------------------------------------------------------

def analyze_cross_seed(
    runs_dir: Path, variant: str, rank: int, scale: float, seeds: List[int]
) -> dict:
    """Compare PQ subspaces across seeds."""
    pq_matrices: Dict[int, torch.Tensor] = {}
    for seed in seeds:
        pt = find_checkpoint(runs_dir, variant, rank, scale, seed)
        p = load_variant_params(pt)
        pq_matrices[seed] = p["output_mult_p"] @ p["output_mult_q"]  # [d, d]

    # Pairwise principal angles
    seed_list = sorted(pq_matrices.keys())
    pairwise: Dict[str, dict] = {}
    for i, s1 in enumerate(seed_list):
        for s2 in seed_list[i + 1:]:
            angles = subspace_angles(pq_matrices[s1].T, pq_matrices[s2].T)
            pairwise[f"{s1}_vs_{s2}"] = angles

    # Mean PQ across seeds and its norm
    mean_pq = sum(pq_matrices.values()) / len(pq_matrices)
    mean_norm = float(torch.norm(mean_pq).item())
    # Std of per-seed PQ norms
    per_seed_norms = [float(torch.norm(v).item()) for v in pq_matrices.values()]
    norm_std = float(np.std(per_seed_norms))
    # Aggregate angles
    all_min = [v["min_deg"] for v in pairwise.values()]
    all_max = [v["max_deg"] for v in pairwise.values()]
    all_mean = [v["mean_deg"] for v in pairwise.values()]

    return {
        "pairwise_principal_angles_deg": pairwise,
        "mean_angle_deg": float(np.mean(all_min)),
        "mean_max_angle_deg": float(np.mean(all_max)),
        "min_angle_deg": float(np.min(all_min)),
        "max_angle_deg": float(np.max(all_max)),
        "mean_pq_frob_norm": mean_norm,
        "per_seed_frob_norms": per_seed_norms,
        "norm_std": norm_std,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs_dir", type=Path, required=True,
                        help="Directory containing trained checkpoints")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds to analyze (default 5)")
    parser.add_argument("--rank", type=int, default=32,
                        help="Rank used for S7 (default 32)")
    parser.add_argument("--scale", type=float, default=0.03,
                        help="Relaxation scale used for S7 (default 0.03)")
    parser.add_argument("--no_word_correction", action="store_true",
                        help="Skip word-class correction analysis (slow)")
    parser.add_argument("--plot", type=Path, default=None,
                        help="Save SVD spectrum plot to this path (requires matplotlib)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save full JSON results to this path")
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))
    rank = args.rank
    scale = args.scale

    results: Dict[str, any] = {"config": {"rank": rank, "scale": scale, "seeds": seeds}}

    # ------------------------------------------------------------------
    # 1. SVD of PQ (single seed or averaged)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Analysis 1: SVD of PQ")
    print("=" * 60)
    try:
        pt_s7 = find_checkpoint(args.runs_dir, "s7", rank, scale, seeds[0])
        p_s7 = load_variant_params(pt_s7)
        svd_result = analyze_svd(p_s7, rank)
        results["svd"] = svd_result
        print(f"  Effective rank (95% energy): {svd_result['effective_rank_95pct']}")
        print(f"  Top-1 energy ratio:          {svd_result['top1_energy_ratio']:.3f}")
        print(f"  Top-4 energy ratio:          {svd_result['top4_energy_ratio']:.3f}")
        print(f"  Condition number:            {svd_result['condition_number']:.1f}")
        print(f"  Top 10 singular values:      {[f'{v:.3f}' for v in svd_result['singular_values_top10']]}")
        if args.plot and _HAS_MPL:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(svd_result["singular_values_all"], "b.-", markersize=3)
            ax1.set_xlabel("Index"); ax1.set_ylabel("Singular value")
            ax1.set_title(f"PQ SVD spectrum (r={rank})")
            ax2.plot(np.cumsum(svd_result["singular_values_all"]) / np.sum(svd_result["singular_values_all"]), "r-")
            ax2.axhline(0.95, color="gray", linestyle="--", label="95%")
            ax2.set_xlabel("Index"); ax2.set_ylabel("Cumulative energy")
            ax2.set_title("Cumulative energy")
            ax2.legend()
            fig.tight_layout()
            fig.savefig(args.plot, dpi=150)
            print(f"  Plot saved to {args.plot}")
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")

    # ------------------------------------------------------------------
    # 2. Word-class correction
    # ------------------------------------------------------------------
    if not args.no_word_correction:
        print()
        print("=" * 60)
        print("Analysis 2: Word-class correction magnitude")
        print("=" * 60)
        try:
            wc = analyze_word_correction(pt_s7, p_s7, rank, scale)
            results["word_correction"] = wc
            print(f"  {'Bucket':<16s} {'mean_rel':>10s} {'std_rel':>10s} {'n_tokens':>10s}")
            print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10}")
            for bucket, stats in wc.items():
                print(f"  {bucket:<16s} {stats['mean_rel_correction']:>10.4f} "
                      f"{stats['std_rel_correction']:>10.4f} {stats['n_tokens']:>10d}")
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")

    # ------------------------------------------------------------------
    # 3. Untying gap
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Analysis 3: Untying gap — does s·WPQ approximate U−W?")
    print("=" * 60)
    try:
        pt_s2 = find_checkpoint(args.runs_dir, "s2", 1, 1.0, seeds[0])
        p_s2 = load_variant_params(pt_s2)
        gap_result = analyze_untying_gap(p_s7, p_s2, scale)
        results["untying_gap"] = gap_result
        print(f"  Cosine similarity (s·WPQ vs U−W):     {gap_result['cosine_similarity']:.4f}")
        print(f"  Explained variance ratio:               {gap_result['explained_variance_ratio']:.4f}")
        print(f"  Optimal scalar a (gap ≈ a·delta_s7):    {gap_result['optimal_scalar_a']:.2f}")
        print(f"  Frobenius norm ratio (δ / gap):          {gap_result['norm_ratio_delta_over_gap']:.4f}")
        # Interpretation
        if gap_result['cosine_similarity'] > 0.8:
            print("  ★ S7 closely aligns with the untying direction — strong evidence")
        elif gap_result['cosine_similarity'] > 0.5:
            print("  △ S7 partially aligns with untying direction")
        else:
            print("  ✗ S7 does NOT align with untying direction — it fixes something else")
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")

    # ------------------------------------------------------------------
    # 4. Cross-seed consistency
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Analysis 4: Cross-seed consistency of PQ subspaces")
    print("=" * 60)
    try:
        seed_result = analyze_cross_seed(args.runs_dir, "s7", rank, scale, seeds)
        results["cross_seed"] = seed_result
        print(f"  Pairwise principal angles (min / mean / max deg):")
        for pair, angles in seed_result["pairwise_principal_angles_deg"].items():
            print(f"    {pair}: min={angles['min_deg']:.1f}° mean={angles['mean_deg']:.1f}° max={angles['max_deg']:.1f}°")
        print(f"  Mean min angle: {seed_result['mean_angle_deg']:.1f}°")
        print(f"  Mean max angle: {seed_result['mean_max_angle_deg']:.1f}°")
        print(f"  Per-seed Frobenius norms: {[f'{n:.3f}' for n in seed_result['per_seed_frob_norms']]}")
        if seed_result.get('mean_angle_deg', 90) < 10:
            print("  ★ PQ subspaces converge near-perfectly across seeds — highly structural")
        elif seed_result.get('mean_angle_deg', 90) < 30:
            print("  ★ PQ subspaces converge across seeds — structural, not noise")
        elif seed_result.get('mean_angle_deg', 90) < 60:
            print("  △ Partial convergence — some seed variation exists")
        else:
            print("  ✗ PQ subspaces vary widely — high seed sensitivity")
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    if args.output:
        import json as _json
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(_json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"\nFull results saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
