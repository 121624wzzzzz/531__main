# A-LoRA: Vocabulary-Boundary LoRA via Hidden-Space Affine Adaptation

Code release for **Rethinking LoRA Placement at the Vocabulary Boundary with A-LoRA**.

A-LoRA adapts vocabulary-boundary layers without learning a free `V × d` residual. Instead, for a frozen matrix `M_0 ∈ R^{V×d}` it learns hidden-dimensional factors:

```
M_eff ≈ M_0 (I + α/r · P Q) + 1_V β^T
```

Trainable parameters scale with hidden size `d` and rank `r`, not vocabulary size `V`.

## Overview

The paper studies four placement topologies:

| Topology | Config flags | Typical use |
|----------|--------------|-------------|
| input-only | `use_input=True` | strongest standalone small-budget adapter |
| output-only (`lm_head`) | `use_lm_head=True` | stable supplement when hidden LoRA is present |
| decoupled input + output | both sides, independent adapters | asymmetric capacity |
| shared tied | `tie_input_lm_head_adapters=True` | mergeable tied models |

Supported training variants in `scripts/train_alora.py`:

- `affine_input`
- `affine_lm_head`
- `affine_input_lm_head`
- `affine_shared_tied` (mergeable tied affine with transpose output path)

## Repository layout

```
.
├── main.pdf
├── README.md
├── requirements.txt
├── src/alora/
│   └── adapter.py           # A-LoRA core (config, apply/save/load)
├── scripts/
│   └── train_alora.py       # SFT smoke test / HF model fine-tune entry
└── analysis/
    ├── affine_fit.py        # Full-vocab Y ≈ XA + b diagnostics
    └── run_affine_fit.py    # CLI for Base→Instruct matrix pairs
```

## Quick start

### 1. Install

```bash
cd getcode/2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train A-LoRA (toy tied model smoke test)

```bash
PYTHONPATH=src python scripts/train_alora.py \
  --variant affine_input \
  --rank 16 \
  --alpha 32 \
  --output-dir outputs/alora_input \
  --steps 100
```

### 3. Train on a Hugging Face causal LM

```bash
PYTHONPATH=src python scripts/train_alora.py \
  --model-path /path/to/Qwen2.5-0.5B-Instruct \
  --variant affine_input_lm_head \
  --rank 16 \
  --output-dir outputs/qwen_alora
```

For full-scale SFT experiments (Alpaca-style data, hidden LoRA combinations, merge-equivalence checks), extend this reference trainer with your dataset pipeline and evaluation harness.

### 4. Analyze affine structure of Base→Instruct deltas

```bash
python analysis/run_affine_fit.py \
  --base-matrix /path/to/base_lm_head.safetensors \
  --instruct-matrix /path/to/instruct_lm_head.safetensors \
  --role lm_head \
  --output-dir outputs/affine_fit
```

Outputs include `R2`, `norm_A_minus_I`, and low-rank energy spectrum of `(A − I)`, supporting the claim that output-side / tied shared matrices match hidden affine structure more closely than untied input embeddings.

## Parameter budget

For rank `r`, hidden size `d`, one-sided adapter with bias:

```
params ≈ 2 r d + d
```

Compare with standard Vocab LoRA on one matrix: `r (V + d)`.

Use `trainable_parameter_count()` in `adapter.py` for topology-aware estimates (shared tied adapters count once).

## Expected findings (qualitative)

1. **Analysis (30 Base→Instruct pairs)**: output LM heads and tied shared matrices exhibit higher affine R² and lower `(A−I)` rank than untied input embeddings.
2. **Fine-tuning**: input-only A-LoRA is the strongest standalone small-budget adapter; with hidden LoRA already present, `lm_head` A-LoRA is a cheaper stable add-on.
3. **Tied models**: shared adapters with transpose output (`affine_shared_tied`) can be merged back into the tied embedding matrix for deployment.

See `main.pdf` for full tables and ablations.

## Concurrent companion submission (EMNLP 2026)

This anonymous release is one of **two concurrent submissions** from the same author group. The companion paper studies **pretraining-time AGD untying**; **this paper** studies **post-training A-LoRA placement** on frozen models.

For EMNLP/ARR compliance:

- Paste the Related Work paragraph from `supplementary/related_work_companion_agd.tex` (or `.md`) into the main paper.
- Upload `supplementary/` with this code release as **anonymous supplementary material**.
- Read `supplementary/companion_submission_statement.md` for contribution boundaries.

**Anonymous cross-citation:** Anonymous (2026a) — *Efficient Untying of Weight-Tied Language Models via Asymmetric Geometric Decoupling*.

## Citation

```bibtex
@inproceedings{alora2026,
  title={Rethinking LoRA Placement at the Vocabulary Boundary with A-LoRA},
  author={Anonymous},
  year={2026}
}
```

## License

Research code released for paper submission review. When running on Hugging Face checkpoints, follow the base model licenses.
