# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: AffLoRA

This project tests **AffLoRA** — low-rank affine adapters on the vocabulary-sized layers (`embed_tokens` and `lm_head`) that standard LoRA typically skips during SFT/post-training. The key insight (from `task6_base_instruct_full_vocab` analysis) is that base→instruct changes in these layers satisfy a hidden-dimensional affine relationship: `W' = W (I + s1·A·B) + s2·b`.

The primary task is `sft_t2t_mini_25k` (24k train + 1k eval), tested on Qwen2.5/Qwen3 base models. Three claims are evaluated: (1a) AffLoRA + hidden LoRA beats hidden LoRA alone; (1b) AffLoRA beats vocab-dim LoRA on emb/lm_head at matched params; (2) AffLoRA-only beats frozen base; (3) AffLoRA beats matched-budget single-layer LoRA.

## Environment

```bash
source /home/wz/projects/mypro/im_exp/set
cd /home/wz/projects/mypro/im_exp/lora
export PYTHONPATH=/home/wz/projects/mypro/im_exp/lora/src:${PYTHONPATH:-}
```

Qwen base models live at `/home/wz/projects/mypro/im_exp/models/`. Outputs go to `outputs/affine_vocab/`.

## Common Commands

**Smoke test** (2 steps on Qwen3-0.6B, validates toolchain and precision):
```bash
bash scripts/sh/run_affine_vocab_smoke.sh
```

**Syntax check** (verifies all Python scripts parse):
```bash
bash scripts/sh/run_syntax_check.sh
```

**Main training** (single GPU, headline hyperparams):
```bash
python scripts/train_affine_vocab_lora.py \
  --model-path /home/wz/projects/mypro/im_exp/models/Qwen2.5-1.5B-Base \
  --train-data data/sft_t2t_mini_25k/train.jsonl \
  --eval-data data/sft_t2t_mini_25k/eval.jsonl --eval-samples 1000 --eval-steps 250 \
  --output-dir outputs/affine_vocab/<run_name> \
  --variant affine_input_lm_head_plus_hidden_lora \
  --hidden-lora-rank 8 --hidden-lora-alpha 16 \
  --affine-rank 16 --affine-alpha 128 \
  --max-seq-len 1024 --per-device-train-batch-size 8 --gradient-accumulation-steps 2 \
  --learning-rate 2e-4 --lr-scheduler-type cosine --warmup-ratio 0.03 \
  --num-train-epochs 1 --save-strategy no \
  --master-dtype fp32 --base-dtype auto --bf16 --seed 42
```

**Frozen base eval** (Claim 2 zero-training reference):
```bash
python scripts/eval_base_loss.py \
  --model-path /home/wz/projects/mypro/im_exp/models/Qwen2.5-1.5B-Base \
  --eval-data data/sft_t2t_mini_25k/eval.jsonl \
  --report-file outputs/affine_vocab/claim2_base/qwen25_1_5b.json
```

## Architecture

### Core library (`src/affine_vocab_lora/adapter.py`)

The entire AffLoRA mechanism lives in one file:

- **`AffineVocabConfig`** — dataclass with all hyperparams: `rank`, `alpha` (LoRA-style scale = alpha/rank), `bias_scale`, `use_input`/`use_lm_head` flags, `tie_input_lm_head_adapters` for tied-embedding models, `intermediate_layer_idx` for position ablation, etc.

- **`LowRankAffineMap`** — the core module. A row-wise affine transform `x → x + s1·up(down(x)) + s2·bias` where down/up are linear projections through a rank bottleneck. Initialized with Kaiming uniform on `down` and zeros on `up` (so the map starts as identity).

- **`AffineEmbedding`** — wraps a frozen `nn.Embedding`, applies `LowRankAffineMap` after lookup.

- **`AffineLMHead`** — wraps a frozen `lm_head` (`nn.Linear`), applies `LowRankAffineMap` to hidden states before the linear projection. Optionally adds a per-token `vocab_logit_bias`.

- **`TiedTransposeAffineLMHead`** — for tied-embedding models sharing one affine adapter between input and output. Applies the transposed affine on the output side so raw logits match a model with the adapter merged into the single embedding matrix.

- **`apply_affine_vocab_adapters(model, cfg)`** — main entry point. Freezes the entire base model, then wraps `embed_tokens` and/or `lm_head` according to config. Handles tied-embedding sharing, position ablation (routing affine to a specific decoder layer instead of embedding), and after-norm placement.

- **`save_affine_vocab_adapter()` / `load_affine_vocab_adapter()`** — serialization via safetensors + JSON config.

### Training script (`scripts/train_affine_vocab_lora.py`)

Single-GPU training using HuggingFace `Trainer` + PEFT. Key design decisions:

1. **Precision**: Trainable parameters must be fp32 with `--master-dtype fp32`. The base model can stay bf16 (`--base-dtype auto --bf16`). bf16 master weights + bf16 AdamW silently rounds away lr=2e-4 updates. The script logs trainable parameter dtypes at start and Adam state dtypes at end — both should show fp32.

2. **Variants**: 8 training variants controlled by `--variant` — `full_finetune`, `hidden_lora`, and combinations of `affine_input`/`affine_lm_head` with/without `plus_hidden_lora`.

3. **Data format**: Supports JSON/JSONL files with `conversations` (list of role/content dicts) or simple `question`/`answer` (or `query`/`response`, etc.) fields. A prompt template wraps the question; labels mask out the prompt prefix with -100.

4. **Affine adapter checkpointing**: A `SaveAffineAdapterCallback` mirrors Trainer's PEFT checkpoint saves for affine adapters, since Trainer's `save_model` only serializes PEFT weights.

### Control scripts

- `scripts/eval_base_loss.py` — computes eval loss on a frozen base model (Claim 2 reference).
- `scripts/eval_merge_equivalence.py` — verifies tied affine adapter↔merged embedding equivalence. `--quick` does a single-text logit diff; without it, runs full PPL + generation comparison on an eval set.
- `data/prepare_ultrachat_100k.py` — prepares UltraChat 100k dataset (one-shot, kept alongside the data it produces).

## Key Design Rules

- **Never train without `--master-dtype fp32`**. bf16 master weights silently kill training at typical learning rates.
- **Model paths**: Primary models at `/home/wz/projects/mypro/im_exp/models/`.
- **Tied embeddings**: All Qwen models used here have tied `embed_tokens`/`lm_head` weights. The `tie_input_lm_head_adapters` + `mergeable_tied_affine_output` flags enable shared affine adapters that can be merged back into the single embedding matrix.
- **Data format**: Training data should have either `conversations` (list of `{role, content}` dicts with "user"/"assistant" roles) or flat `question`/`answer` (or `query`/`response`, `instruction`/`output`, `problem`/`solution`, etc.).
- **Evaluated by eval_loss only**: This is a pure language modeling loss study. There is no generation-based evaluation in the active workflow.

## Results & Experiment Tracking

- `docs/RESULTS_SO_FAR.md` — the canonical results ledger. All claim evidence is recorded here.
- `docs/AFFINE_VOCAB_MAIN_EXPERIMENT.md` — experiment design and the three claims.
- `docs/EXPERIMENT_SETTINGS.md` — full paths, model table, hyperparams, variant descriptions.
- `docs/CLEANUP_MANIFEST.md` — what was deleted/retained and why. Each evidence run must retain: `run_args.json`, `trainable_summary.json`, affine config/weights, and training log.
- `docs/` also contains per-phase result files (e.g., `phase5a_results.md`) with detailed numeric tables.
