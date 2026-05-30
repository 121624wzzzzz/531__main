"""Combined gradient plot: all variants in one figure, 3-seed mean."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

LOGS = Path(__file__).resolve().parent.parent / "logs" / "fineedu"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

VARIANTS = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s11", "s12", "s13"]
SEEDS = [42, 123, 2026]

# --- load data ---
data = {}
for v in VARIANTS:
    all_recs = []
    for s in SEEDS:
        path = LOGS / f"seed{s}" / f"{v}_grad_stats.jsonl"
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_recs.append(json.loads(line))
    data[v] = all_recs

def mean_by_step(recs, key, field, max_step=28000):
    """Average across seeds for each step bucket, return (steps, values)."""
    by_step = {}
    for r in recs:
        step = r["step"]
        metric = r.get(key, {})
        v = metric.get(field) if metric else None
        if v is None:
            continue
        by_step.setdefault(step, []).append(v)
    steps = sorted([s for s in by_step if s <= max_step])
    means = [np.mean(by_step[s]) for s in steps]
    return steps, means

# --- plot layout: 2 rows x 5 cols ---
fig, axes = plt.subplots(2, 5, figsize=(24, 10))
fig.suptitle("FineEdu — Embedding vs Unembedding gradient norm (3-seed mean)", fontsize=16, y=0.995)

for idx, variant in enumerate(VARIANTS):
    ax = axes[idx // 5][idx % 5]
    recs = data[variant]

    steps_e, embed = mean_by_step(recs, "embed_grad", "norm")
    steps_u, unemb = mean_by_step(recs, "unemb_grad", "norm")
    steps_s, shared = mean_by_step(recs, "shared_total_grad", "norm")

    ax.plot(steps_e, embed, color="#1f77b4", alpha=0.9, label="embed")
    ax.plot(steps_u, unemb, color="#ff7f0e", alpha=0.9, label="unemb")
    if steps_s:
        ax.plot(steps_s, shared, color="#2ca02c", alpha=0.7, ls="--", label="shared")

    ax.set_title(variant.upper(), fontsize=13, fontweight="bold")
    if idx >= 5:
        ax.set_xlabel("step")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUT / "fineedu-gradient-all-variants-combined.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved: {out_path}")

# --- also a single overlay plot for quick comparison ---
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
cmap = plt.cm.tab10
for idx, variant in enumerate(VARIANTS):
    color = cmap(idx)
    recs = data[variant]
    steps_e, embed = mean_by_step(recs, "embed_grad", "norm")
    steps_u, unemb = mean_by_step(recs, "unemb_grad", "norm")
    ax1.plot(steps_e, embed, color=color, alpha=0.8, label=variant.upper())
    ax2.plot(steps_u, unemb, color=color, alpha=0.8, label=variant.upper())

ax1.set_title("Embedding gradient norm (3-seed mean)")
ax2.set_title("Unembedding gradient norm (3-seed mean)")
for ax in (ax1, ax2):
    ax.set_xlabel("step")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path2 = OUT / "fineedu-gradient-all-variants-overlay.png"
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"saved: {out_path2}")
print("done")
