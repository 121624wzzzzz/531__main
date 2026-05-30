"""Plot gradient cosine similarity between embed/unemb for all tied variants."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

LOGS = Path(__file__).resolve().parent.parent / "logs" / "fineedu"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

TIED_VARIANTS = ["s1", "s3", "s4", "s6", "s7", "s11", "s12", "s13"]
SEEDS = [42, 123, 2026]
COLORS = {42: "#1f77b4", 123: "#ff7f0e", 2026: "#2ca02c"}

# --- load ---
data = {}
for v in TIED_VARIANTS:
    all_recs = []
    for s in SEEDS:
        path = LOGS / f"seed{s}" / f"{v}_grad_stats.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    rec["_seed"] = s
                    all_recs.append(rec)
    data[v] = all_recs

# --- combined grid: 2x4 ---
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("FineEdu — Embed/Unembed gradient cosine similarity (tied_split.embed_unemb_grad_cosine)", fontsize=14, y=0.995)

for idx, variant in enumerate(TIED_VARIANTS):
    ax = axes[idx // 4][idx % 4]
    for seed in SEEDS:
        recs = [r for r in data[variant] if r["_seed"] == seed]
        if not recs:
            continue
        steps = [r["step"] for r in recs]
        cos = [r.get("tied_split", {}).get("embed_unemb_grad_cosine", float("nan")) for r in recs]
        ax.plot(steps, cos, color=COLORS[seed], alpha=0.85, label=f"seed {seed}")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    ax.set_title(variant.upper(), fontsize=13, fontweight="bold")
    if idx >= 4:
        ax.set_xlabel("step")
    ax.set_ylabel("cosine")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out1 = OUT / "fineedu-gradient-cosine-all-variants.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved: {out1}")

# --- overlay comparison ---
fig2, ax = plt.subplots(figsize=(14, 7))
cmap = plt.cm.tab10
for idx, variant in enumerate(TIED_VARIANTS):
    # 3-seed mean
    by_step = {}
    for r in data[variant]:
        cos = r.get("tied_split", {}).get("embed_unemb_grad_cosine")
        if cos is None:
            continue
        by_step.setdefault(r["step"], []).append(cos)
    if not by_step:
        continue
    steps = sorted(by_step.keys())
    means = [np.mean(by_step[s]) for s in steps]
    ax.plot(steps, means, color=cmap(idx), alpha=0.85, label=variant.upper(), linewidth=1.5)

ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
ax.set_title("FineEdu — Embed/Unembed gradient cosine (3-seed mean)", fontsize=14)
ax.set_xlabel("step")
ax.set_ylabel("cosine similarity")
ax.legend(fontsize=10, ncol=4)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out2 = OUT / "fineedu-gradient-cosine-overlay.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"saved: {out2}")
print("done")
