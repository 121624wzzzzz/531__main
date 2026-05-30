## Concurrent work on vocabulary-boundary adaptation (paste into Related Work)

In concurrent anonymous work (Anonymous, 2026b), the authors study **post-training** placement of low-rank adapters at the vocabulary boundary and propose **A-LoRA**, a hidden-space affine parameterization

`M_eff ≈ M_0 (I + α/r · P Q) + 1_V β^T`

whose trainable cost scales with hidden dimension rather than vocabulary size. Their analysis of 30 Base→Instruct pairs similarly finds that output LM heads and tied shared matrices are better approximated by hidden affine structure than untied input embeddings, and that optimal adapter placement is **conditional** on existing hidden-layer LoRA capacity.

**Our work is complementary, not overlapping:** we focus on **pretraining-time** relaxation of hard weight tying via Asymmetric Geometric Decoupling (AGD), keeping a shared output-side matrix while adding structured input-side freedom (bias correction and hidden low-rank deformation). Where the companion paper asks *where to adapt a frozen model*, we ask *how to untie efficiently while training from scratch or continuing pretraining*. The two papers share geometric motivation but differ in **stage** (pretrain vs. fine-tune), **method** (architectural decoupling vs. PEFT placement), and **evaluation** (pretrain PPL vs. SFT loss and mergeability). We cite the companion submission per EMNLP/ARR concurrent-work policy.

**Suggested citation:** Anonymous. 2026b. *Rethinking LoRA Placement at the Vocabulary Boundary with A-LoRA.* Anonymous EMNLP 2026 submission.
