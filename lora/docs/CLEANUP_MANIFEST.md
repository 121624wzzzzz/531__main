# 清理与归档记录

本文记录 `lora/outputs`、`lora/logs` 以及 retired code path 的保守清理策略。关键数值已经保存在 `docs/RESULTS_SO_FAR.md`。

## 保留策略

每个证据 run 至少保留：

- `run_args.json`
- `trainable_summary.json`
- affine run 的 `affine_vocab_config.json` 和 `affine_vocab_adapter.safetensors`
- 对应 `logs/affine_vocab/<phase>/*__train.log`

当指标已经写入 `docs/RESULTS_SO_FAR.md` 后，可以删除：

- `checkpoint-*`
- 大的 PEFT `adapter_model.safetensors`
- 确认不需要从输出目录直接 reload 时的重复 `tokenizer.json`

## 证据索引

| Phase | 路径 | 用途 |
|---|---|---|
| 2' | `outputs/affine_vocab/phase2_sweep_*/*/` | Claim 2 |
| 2'' | `outputs/affine_vocab/phase2b_*/*/` | Claim 1a 弱证据 |
| 2''' | `outputs/affine_vocab/phase2c_*/*/` | Claim 3 |
| 3a | `outputs/affine_vocab/phase3a_*/*/` | Claim 1a / 2 |
| 3b | `outputs/affine_vocab/phase3b_*/*/` | s1 超参 |
| 4a | `outputs/affine_vocab/claim2_base/` | Claim 2 frozen-base 控制 |
| 4b | `outputs/affine_vocab/phase4b_*/*/` | Claim 1b |
| 4c | `outputs/affine_vocab/phase4c_*/*/` | Claim 1a 多 seed |
| supp | `outputs/affine_vocab/phase3c_*`, `phase3d_*` | 补充实验 |

## 已执行清理

### Step 1：过时输出

执行时间：2026-05-19。

| 已删除 | 原因 |
|---|---|
| `outputs/affine_vocab/phase1_*` | math smoke，无区分信号 |
| `outputs/affine_vocab/phase1_5_*` | math 放大，无区分信号 |
| `outputs/affine_vocab/pilot/` | 已被 Phase 3a 取代 |
| `outputs/affine_vocab/smoke_data/` | 已过时 |
| `outputs/affine_vocab/overnight_*` | 中止 run |
| `outputs/affine_vocab/smoke/**/checkpoint-*` | smoke checkpoint 体积较大且非证据 |
| 对应 `logs/affine_vocab/` | 非证据日志 |

保留：`outputs/affine_vocab/smoke/` 和 `ddp_smoke/` 的最小 replay 产物。

### Step 2：大权重

执行时间：2026-05-19。

| 已删除 | 原因 |
|---|---|
| `phase2*` / `phase3*` / `phase4*` 下的 `adapter_model.safetensors` | 指标已记录，审计不要求 reload PEFT 权重 |

清理后 `outputs/` 约 740M，`logs/` 约 17M。

### Step 3：retired 代码路径

执行时间：2026-05-19。

| 已删除 | 原因 |
|---|---|
| `scripts/eval_affine_vocab_math.py`, `scripts/eval_math.py` | math generation eval 不再属于 active claim workflow |
| `scripts/run_phase1_validate.sh`, `scripts/run_phase1_5_validate.sh` | Phase 1 / 1.5 已归档 |
| `scripts/train_math_lora.py`, `scripts/train_glue_lora.py`, `scripts/train_e2e_lora.py`, `scripts/make_run_matrix.py`, `scripts/download_paper_models.py`, `scripts/prepare_lora_data.py` | LoRMA-paper baseline workflow 已归档 |
| `configs/lorma_lora_matrix.yaml`, `docs/EXPERIMENT_PLAN.md` | LoRMA-paper baseline 配置与计划已归档 |

## 暂不删除

- `data/sft_t2t_mini_25k/`：主任务数据，必须保留。
- `data/ultrachat_100k/`、`data/minimind_random_100k/`：活跃中的扩展实验数据。

## Step 4：graduation 旧依赖

执行时间：2026-05-30。

以下退役内容从目录中移除，不再保留本地审计副本（git 历史和远程仓库仍可回溯）：

| 已删除 | 原因 |
|---|---|
| `data/metamathqa_40k/`, `data/gsm8k/`, `data/math/`, `data/glue/`, `data/e2e_nlg/`, `data/raw/` | 历史 math / GLUE / E2E 数据，675 MB |
| `models/roberta-base/`, `models/roberta-large/`, `models/gpt2-medium/` | 历史 LoRMA-paper checkpoint，~6.6 GB |
| `scripts/train_math_lora.py`, `train_glue_lora.py`, `train_e2e_lora.py`, `download_paper_models.py`, `make_run_matrix.py`, `prepare_lora_data.py`, `eval_affine_vocab_math.py`, `eval_math.py` | 空壳文件，已在 Step 3 清空 |
| `configs/lorma_lora_matrix.yaml` | 空文件 |
| `docs/EXPERIMENT_PLAN.md` | 空文件 |

总计释放约 7.3 GB。

## Step 5：scripts 精简

执行时间：2026-05-30。

删除 48 个已完成的 phase launcher（run_phase*.sh、run_*_dynamic.sh、resume_*.sh）和 3 个一次性辅助脚本（prepare_sft_t2t_split.py、prepare_minimind_random_100k.py、list_cleanup_candidates.sh）。每个 run 的超参已保存在对应 `outputs/affine_vocab/<run>/run_args.json` 中，可随时复现。

清理后 scripts/ 保留 9 个文件：
`train_affine_vocab_lora.py`、`eval_base_loss.py`、`run_affine_vocab_smoke.sh`、`run_syntax_check.sh`、`check_tied_affine_merge_equivalence.py`、`eval_merge_equivalence.py`、`diagnose_affine_gradients.py`、`collect_afflora_results.py`、`prepare_ultrachat_100k.py`。

## Step 6：scripts 去重合并

执行时间：2026-05-30。

- `prepare_ultrachat_100k.py` 移至 `data/`（一次性数据准备，靠近数据本身）
- `check_tied_affine_merge_equivalence.py` 合并进 `eval_merge_equivalence.py`（加 `--quick` flag，消除 ~50 行重复的 merge math）
- `diagnose_affine_gradients.py` 删除（开发阶段工具，架构已稳定）
- `collect_afflora_results.py` 删除（被 `docs/RESULTS_SO_FAR.md` 取代）

最终 scripts/ 保留 Python 脚本 + sh/ 子目录：
`train_affine_vocab_lora.py`、`eval_base_loss.py`、`eval_merge_equivalence.py`、`sh/run_affine_vocab_smoke.sh`、`sh/run_syntax_check.sh`。
