#!/usr/bin/env python3
"""Plot FineEdu/GPT-2 S1 vs S2 gradient norms (3-seed mean) as vector PDF."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
SEEDS = (42, 123, 2026)

# Publication-friendly defaults (vector text stays crisp in PDF).
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.9,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.0,
})


def load_grad_series(variant: str) -> dict[str, list[float]]:
    by_key: dict[str, list[list[float]]] = defaultdict(list)
    steps_ref: list[int] | None = None

    for seed in SEEDS:
        path = LOGS / f"fineedu-gpt2-6b-seed{seed}" / f"{variant}_grad_stats.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        steps = [row["step"] for row in rows]
        if steps_ref is None:
            steps_ref = steps
        elif steps != steps_ref:
            raise ValueError(f"step mismatch for {variant} seed={seed}")

        by_key["embed"].append([row["embed_grad"]["norm"] for row in rows])
        by_key["unemb"].append([row["unemb_grad"]["norm"] for row in rows])
        if variant == "s1":
            by_key["shared"].append([row["shared_total_grad"]["norm"] for row in rows])

    out: dict[str, list[float]] = {"step_k": [s / 1000.0 for s in steps_ref]}
    for key, runs in by_key.items():
        n = len(runs[0])
        out[key] = [sum(runs[s][i] for s in range(len(runs))) / len(runs) for i in range(n)]
    return out


def plot_panel(ax, data: dict[str, list[float]], title: str, series: list[tuple[str, str, str]]) -> None:
    x = data["step_k"]
    for label, key, color in series:
        ax.plot(x, data[key], label=label, color=color)
    ax.set_title(title, fontweight="600", pad=10)
    ax.set_xlabel("Step (k)")
    ax.set_ylabel("Gradient norm")
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, 0.13)
    ax.set_yticks([i * 0.02 for i in range(7)])
    ax.grid(True, linestyle="-", alpha=0.35)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#ccc")


def main() -> None:
    s1 = load_grad_series("s1")
    s2 = load_grad_series("s2")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=150)
    fig.subplots_adjust(wspace=0.28, left=0.07, right=0.98, top=0.90, bottom=0.14)

    plot_panel(
        axes[0],
        s1,
        "S1 tied baseline",
        [
            ("input embedding path", "embed", "#1f77b4"),
            ("output unembedding path", "unemb", "#ff7f0e"),
            ("shared total gradient", "shared", "#2ca02c"),
        ],
    )
    plot_panel(
        axes[1],
        s2,
        "S2 untied baseline",
        [
            ("input embedding gradient", "embed", "#1f77b4"),
            ("output head gradient", "unemb", "#ff7f0e"),
        ],
    )

    out_pdf = LOGS / "fineedu-tied-vs-untied-grad-mean.pdf"
    out_svg = LOGS / "fineedu-tied-vs-untied-grad-mean.svg"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_svg}")


if __name__ == "__main__":
    main()
