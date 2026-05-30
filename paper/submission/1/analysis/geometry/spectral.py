"""Static geometry audit for embedding (E) and unembedding (U) matrices.

This is a lightweight port of the spectral tooling used in the paper's
cross-model diagnostics. It expects per-model safetensors extracts with keys
such as ``model.embed_tokens.weight`` and ``model.lm_head.weight``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_matrix(path: Path, key: str) -> np.ndarray:
    """Load a single weight matrix from safetensors or .npy."""
    if path.suffix == ".npy":
        return np.load(path)
    try:
        from safetensors.numpy import load_file
    except ImportError as exc:
        raise ImportError("Install safetensors to read .safetensors extracts.") from exc
    tensors = load_file(str(path))
    if key not in tensors:
        raise KeyError(f"{key!r} not in {path}. Available: {list(tensors)[:5]}...")
    return np.asarray(tensors[key], dtype=np.float32)


def spectral_metrics(matrix: np.ndarray) -> dict[str, float]:
    """Compute rank-1 / rank-5 energy and mean-axis statistics."""
    x = matrix.astype(np.float64, copy=False)
    n, _ = x.shape
    row_norms = np.linalg.norm(x, axis=1)
    mu = x.mean(axis=0)
    mean_row_norm = float(row_norms.mean())
    mean_vec_norm = float(np.linalg.norm(mu))
    mu_ratio = mean_vec_norm / mean_row_norm if mean_row_norm else float("nan")

    gram = (x.T @ x) / max(n, 1)
    evals = np.linalg.eigvalsh(gram)
    evals = np.clip(evals * n, 0.0, None)
    total = float(evals.sum())
    if total <= 0:
        return {
            "mean_row_norm": mean_row_norm,
            "mean_vec_norm": mean_vec_norm,
            "mu_ratio": mu_ratio,
            "rank1_energy_frac": 0.0,
            "rank5_energy_frac": 0.0,
            "effective_rank": 0.0,
        }
    p = evals / total
    p_sorted = np.sort(p)
    rank1 = float(p_sorted[-1])
    rank5 = float(p_sorted[-5:].sum()) if p_sorted.size >= 5 else float(p_sorted.sum())
    entropy = -np.sum(p_sorted[p_sorted > 0] * np.log(p_sorted[p_sorted > 0] + 1e-20))
    return {
        "mean_row_norm": mean_row_norm,
        "mean_vec_norm": mean_vec_norm,
        "mu_ratio": mu_ratio,
        "rank1_energy_frac": rank1,
        "rank5_energy_frac": rank5,
        "effective_rank": float(math.exp(entropy)),
    }


def gcorr(a: np.ndarray, b: np.ndarray, max_rows: int = 4096) -> float:
    """Generalized correlation between two same-shaped matrices (subsampled rows)."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    n = a.shape[0]
    if n > max_rows:
        idx = np.linspace(0, n - 1, max_rows, dtype=int)
        a, b = a[idx], b[idx]
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    num = float(np.sum(a * b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return num / den


def audit_model_extract(
    extract_path: Path,
    *,
    embed_key: str = "model.embed_tokens.weight",
    head_key: str = "model.lm_head.weight",
    info_json: Path | None = None,
) -> dict[str, Any]:
    """Return geometry metrics for E, U, and tied/shared matrix diagnostics."""
    e = load_matrix(extract_path, embed_key)
    u = load_matrix(extract_path, head_key)
    tied = np.shares_memory(e, u) or np.allclose(e, u)
    if info_json and info_json.exists():
        meta = json.loads(info_json.read_text(encoding="utf-8"))
        tied = bool(meta.get("tie_word_embeddings", tied))

    out: dict[str, Any] = {
        "extract": str(extract_path),
        "tied": tied,
        "vocab_size": int(e.shape[0]),
        "hidden_dim": int(e.shape[1]),
    }
    out.update({f"E_{k}": v for k, v in spectral_metrics(e).items()})
    out.update({f"U_{k}": v for k, v in spectral_metrics(u).items()})
    if not tied:
        out["gcorr_E_U"] = gcorr(e, u)
        out["gcorr_E_minus_U"] = gcorr(e, u - e)
    else:
        out["note"] = "Shared matrix is closer to output-side geometry in the paper's cross-model study."
    return out
