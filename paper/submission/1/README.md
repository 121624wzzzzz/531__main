# Asymmetric Geometric Decoupling (AGD)

Code release for **Efficient Untying of Weight-Tied Language Models via Asymmetric Geometric Decoupling**.

This repository contains a self-contained reference implementation of AGD and the static geometry diagnostics used to motivate it. The full-scale MiniMind / FineWeb-Edu pretraining stack lives in the upstream project; here we provide the **core vocabulary-boundary mechanism**, a **minimal training/evaluation loop**, and **embedding/unembedding geometry audit** scripts.

## Overview

Weight tying shares one `V × d` matrix between input embeddings `E` and output unembedding `U`. AGD keeps the shared matrix on the **output side** (still tied) while adding lightweight structured freedom on the **input side**:

| Variant | Tying | Input-side freedom | Role |
|---------|-------|-------------------|------|
| `s1` | tied | none | tied baseline |
| `s2` | untied | full separate `U` | full-untie baseline |
| `s3` | tied | bias correction `E(token) − β` | ablation |
| `s6` | tied | hidden low-rank deformation | ablation |
| **`s12`** | tied | **bias + hidden low-rank** | **main AGD input-side variant** |

Parameter count for `s12` scales as `O(d + r·d)` instead of `O(V·d)` for full untying.

Forward path for the main input-side AGD variant (`s12`):

```
E_eff(token) = E(token) − β + B(A(E(token)))
logits = lm_head(h)
```

where `A: R^d → R^r`, `B: R^r → R^d`, and `β ∈ R^d`.

## Repository layout

```
.
├── main.pdf                 # Paper PDF
├── README.md
├── requirements.txt
├── src/agd/
│   ├── config.py            # AGDConfig
│   ├── embedding_head.py    # AGD variants on E/U boundary
│   └── model.py             # Minimal causal LM wrapper
├── scripts/
│   ├── train_pretrain.py    # Minimal pretraining loop
│   └── eval_variants.py     # Compare s1/s2/s3/s6/s12
└── analysis/
    ├── geometry/spectral.py # E/U spectral & GCorr metrics
    └── run_geometry_audit.py
```

## Quick start

### 1. Install

```bash
cd getcode/1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train an AGD variant

Prepare JSONL with token ids (`{"input_ids": [...]}`), then:

```bash
PYTHONPATH=src python scripts/train_pretrain.py \
  --train-data /path/to/train.jsonl \
  --output-dir outputs/agd_s12 \
  --variant s12 \
  --variant-rank 32 \
  --epochs 1
```

### 3. Evaluate variants

```bash
PYTHONPATH=src python scripts/eval_variants.py \
  --eval-data /path/to/eval.jsonl \
  --output-dir outputs/eval \
  --variants s1,s2,s3,s6,s12
```

### 4. Static geometry audit

Given a `.safetensors` extract with `model.embed_tokens.weight` and `model.lm_head.weight`:

```bash
python analysis/run_geometry_audit.py \
  --extract /path/to/model.safetensors \
  --info-json /path/to/model.info.json \
  --output-dir outputs/geometry
```

Metrics include mean-axis ratio (`μ-ratio`), rank-1 / rank-5 spectral energy, and (for untied models) generalized correlation between `E` and `U`.

## Expected results (qualitative)

On tied small LMs, the paper reports:

1. **Diagnostics**: shared tied matrices are spectrally closer to untied output unembeddings than to input embeddings.
2. **Pretraining**: input-side AGD (`s12`) improves over tied baseline (`s1`) with far lower overhead than full untying (`s2`).
3. **Asymmetry**: output-side AGD variants (`s7`, `s11`) are less stable than input-side counterparts.

Exact numbers depend on dataset, model size, and training budget; see `main.pdf`.

## Concurrent companion submission (EMNLP 2026)

This anonymous release is one of **two concurrent submissions** from the same author group on vocabulary-boundary asymmetry. The companion paper studies **post-training A-LoRA placement** on frozen models; **this paper** studies **pretraining-time AGD untying**.

For EMNLP/ARR compliance:

- Paste the Related Work paragraph from `supplementary/related_work_companion_alora.tex` (or `.md`) into the main paper.
- Upload `supplementary/` together with this code release as **anonymous supplementary material**.
- Read `supplementary/companion_submission_statement.md` for the side-by-side contribution comparison.

**Anonymous cross-citation:** Anonymous (2026b) — *Rethinking LoRA Placement at the Vocabulary Boundary with A-LoRA*.

## Citation

```bibtex
@inproceedings{agd2026,
  title={Efficient Untying of Weight-Tied Language Models via Asymmetric Geometric Decoupling},
  author={Anonymous},
  year={2026}
}
```

## License

Research code released for paper submission review. Adapt upstream MiniMind / Hugging Face dependencies under their respective licenses when scaling beyond this minimal reference.
