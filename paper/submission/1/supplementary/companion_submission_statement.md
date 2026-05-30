# Statement on Concurrent Anonymous Companion Submission

**Paper:** Efficient Untying of Weight-Tied Language Models via Asymmetric Geometric Decoupling (AGD)  
**Venue:** EMNLP 2026 (anonymous review)  
**Companion paper:** *Rethinking LoRA Placement at the Vocabulary Boundary with A-LoRA* (submitted concurrently, same author group, anonymous)

---

## Compliance statement

The author group is submitting **two concurrent anonymous papers** on closely related themes: vocabulary-boundary geometry, input/output asymmetry between embeddings and LM heads, and hidden-dimensional low-rank structure at the lexical interface.

We affirm that:

1. **No thin slicing:** The two papers address **distinct research questions and experimental settings** (see comparison below). Neither paper is a subset of the other.
2. **Mutual awareness:** Both submissions explicitly cross-reference the companion work in **Related Work** using anonymous citations (`Anonymous, 2026a` / `Anonymous, 2026b`).
3. **Difference discussion:** Each paper states what the companion contributes, what remains unique to the present submission, and why both results are needed.
4. **Shared diagnostics, separate claims:** Some geometric analyses (E/U spectral structure; Base→Instruct affine fit on output-side matrices) inform both lines of work, but the **primary contributions and evaluations differ**.

---

## Side-by-side comparison (for reviewers / area chairs)

| Dimension | **This paper (AGD)** | **Companion (A-LoRA)** |
|-----------|----------------------|-------------------------|
| **Stage** | Pretraining / continued pretraining | Post-training PEFT / SFT |
| **Problem** | Relax weight tying without full `V×d` untying cost | Where to place LoRA at vocabulary boundary under budget |
| **Method** | Architectural decoupling: tied output + structured input freedom (bias + hidden LR) | Hidden-space affine adapter: `M_eff ≈ M_0(I+α/r·PQ)+β` |
| **Training regime** | MiniMind, FineWeb-Edu/GPT-2 from scratch or CPT | Frozen base LLMs + instruction tuning |
| **Main metric** | Pretrain loss / PPL vs tied & fully untied baselines | SFT eval loss; placement × hidden-LoRA interaction |
| **Unique claim** | Input-side AGD improves over tying at ~1% untie cost | Optimal boundary placement is **conditional** on existing adapter capacity |
| **Deployment focus** | Parameter-efficient untying at small scale | Mergeable tied A-LoRA for serving |

---

## Shared material

- Static geometry audit scripts (spectral / μ-ratio) appear in both code releases in simplified form.
- The observation that **output-side / tied matrices are more output-like** supports both papers but is **motivation** here and **structural applicability analysis** in the companion.

---

## Contact after acceptance

Author identities and non-anonymous cross-links will be added in the camera-ready version if both papers are accepted.
