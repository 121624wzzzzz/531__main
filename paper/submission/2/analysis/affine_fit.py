"""Full-vocabulary affine fit for Base->Instruct matrix deltas.

Given aligned token rows X (base) and Y (instruct), estimate:

    Y ≈ X A + b

and report R², A-relative-to-identity diagnostics, and low-rank energy of (A - I).
This supports the paper's claim that output LM heads / tied shared matrices are
better approximated by hidden-dimensional affine structure than untied input sides.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

BATCH_ROWS = 8192
ENERGY_THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.99)


def _as_tensor(arr: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(device=device, dtype=dtype)


def svd_energy_from_gram(gram: torch.Tensor, *, prefix: str) -> dict[str, Any]:
    eigvals = torch.linalg.eigvalsh(gram)
    eigvals = torch.clamp(eigvals, min=0.0).flip(0)
    total = float(eigvals.sum().item())
    out: dict[str, Any] = {f"{prefix}_total_energy": total}
    if total <= 0:
        for threshold in ENERGY_THRESHOLDS:
            out[f"{prefix}_rank_{int(threshold * 100)}"] = 0
        out[f"{prefix}_effective_rank"] = 0.0
        return out
    energy = eigvals / total
    cumulative = torch.cumsum(energy, dim=0)
    for threshold in ENERGY_THRESHOLDS:
        idx = torch.searchsorted(cumulative, torch.tensor(threshold, device=cumulative.device))
        out[f"{prefix}_rank_{int(threshold * 100)}"] = int(idx.item()) + 1
    nonzero = energy[energy > 0]
    entropy = -torch.sum(nonzero * torch.log(nonzero)).item()
    out[f"{prefix}_effective_rank"] = float(math.exp(entropy))
    return out


def a_diagnostics(A: torch.Tensor) -> dict[str, float]:
    d = int(A.shape[0])
    eye = torch.eye(d, device=A.device, dtype=A.dtype)
    norm_a = torch.linalg.matrix_norm(A, ord="fro")
    diff_i = torch.linalg.matrix_norm(A - eye, ord="fro")
    norm_i = float(d) ** 0.5
    return {
        "norm_A_minus_I": float(diff_i.item()),
        "rel_A_minus_I_over_I": float(diff_i.item() / (norm_i + 1e-20)),
        "identity_cosine": float((torch.trace(A) / ((norm_a * norm_i) + 1e-20)).item()),
    }


def full_affine_stream(
    X_np: np.ndarray,
    Y_np: np.ndarray,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    batch_rows: int = BATCH_ROWS,
) -> dict[str, Any]:
    """Streaming centered normal equations for Y ~= X A + b."""
    if X_np.shape != Y_np.shape:
        raise ValueError(f"shape mismatch: {X_np.shape} vs {Y_np.shape}")
    device = device or torch.device("cpu")
    n, d = X_np.shape

    sum_x = np.zeros(d, dtype=np.float64)
    sum_y = np.zeros(d, dtype=np.float64)
    y2_sum = 0.0
    for start in range(0, n, batch_rows):
        end = min(n, start + batch_rows)
        xb = X_np[start:end].astype(np.float64, copy=False)
        yb = Y_np[start:end].astype(np.float64, copy=False)
        sum_x += xb.sum(axis=0)
        sum_y += yb.sum(axis=0)
        y2_sum += float((yb * yb).sum())

    mean_x64 = sum_x / n
    mean_y64 = sum_y / n
    ss_tot = float(y2_sum - n * float(mean_y64 @ mean_y64))

    G = torch.zeros((d, d), device=device, dtype=dtype)
    C = torch.zeros((d, d), device=device, dtype=dtype)
    mx = torch.from_numpy(mean_x64.astype(np.float32)).to(device=device, dtype=dtype)
    my = torch.from_numpy(mean_y64.astype(np.float32)).to(device=device, dtype=dtype)

    for start in range(0, n, batch_rows):
        end = min(n, start + batch_rows)
        x = _as_tensor(X_np[start:end], device, dtype) - mx
        y = _as_tensor(Y_np[start:end], device, dtype) - my
        G.add_(x.T @ x)
        C.add_(x.T @ y)

    A = torch.linalg.solve(G, C)
    b = my - mx @ A

    ss_res = 0.0
    y_norm_sq = 0.0
    for start in range(0, n, batch_rows):
        end = min(n, start + batch_rows)
        x = _as_tensor(X_np[start:end], device, dtype)
        y = _as_tensor(Y_np[start:end], device, dtype)
        pred = x @ A + b
        diff = y - pred
        ss_res += float((diff * diff).sum().item())
        y_norm_sq += float((y * y).sum().item())

    out: dict[str, Any] = {
        "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "rel_err": (ss_res**0.5) / ((y_norm_sq**0.5) + 1e-20),
        "norm_b": float(torch.linalg.vector_norm(b).item()),
        "num_rows": n,
        "hidden_dim": d,
    }
    out.update(a_diagnostics(A))
    a_delta = A - torch.eye(d, device=A.device, dtype=A.dtype)
    out.update(svd_energy_from_gram(a_delta.T @ a_delta, prefix="A_delta"))
    return out
