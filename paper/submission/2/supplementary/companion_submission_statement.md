# Statement on Concurrent Anonymous Companion Submission

**Paper:** Rethinking LoRA Placement at the Vocabulary Boundary with A-LoRA  
**Venue:** EMNLP 2026 (anonymous review)  
**Companion paper:** *Efficient Untying of Weight-Tied Language Models via Asymmetric Geometric Decoupling* (submitted concurrently, same author group, anonymous)

---

## Compliance statement

The author group is submitting **two concurrent anonymous papers** on vocabulary-boundary asymmetry, hidden-dimensional structure at the lexical interface, and parameter-efficient alternatives to full `V×d` updates.

We affirm that:

1. **No thin slicing:** The companion paper targets **pretraining-time architectural untying**; this paper targets **post-training adapter placement**. They are complementary, not duplicate slices.
2. **Mutual citation:** Both submissions cross-reference each other in **Related Work** with anonymous labels.
3. **Explicit difference discussion:** Each paper clarifies scope boundaries (see table below).
4. **Supplementary anonymity:** This folder and the companion's supplementary folder are uploaded as **anonymous supplementary material** per EMNLP/ARR policy.

---

## Side-by-side comparison

| Dimension | **Companion (AGD)** | **This paper (A-LoRA)** |
|-----------|---------------------|-------------------------|
| **Stage** | Pretraining / CPT | SFT / instruction tuning |
| **Problem** | Cheap alternative to full untying | Vocabulary-boundary LoRA placement |
| **Method** | Bias + hidden LR on input; keep tied output | Affine hidden adapter; 4 topologies + mergeable tied variant |
| **Baselines** | Tied S1 vs full untie S2 | Vocab LoRA, hidden LoRA, matched-param baselines |
| **Analysis** | Cross-model geometry + training dynamics | 30 Base→Instruct affine structure + placement sweep |
| **Unique claim** | Input-side decoupling beats tying in pretrain | Placement depends on residual type × existing capacity |

---

## Shared vs. distinct

**Shared:** E/U asymmetry diagnostics; hidden low-rank parameterization intuition.  
**Distinct:** AGD modifies the **training objective path** during pretrain; A-LoRA modifies **where to attach adapters** on frozen weights and how to merge them at deployment.

---

## Camera-ready plan

If both papers are accepted, we will add explicit non-anonymous citations and a short joint footnote clarifying the relationship between the two lines of work.
