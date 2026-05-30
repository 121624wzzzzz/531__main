# 实验运行设置

本文记录 AffLoRA 实验的运行路径、模型、数据、variant 和超参。实验设计见 `docs/AFFINE_VOCAB_MAIN_EXPERIMENT.md`，结果见 `docs/RESULTS_SO_FAR.md`。

AffLoRA 的依据是 `/home/wz/projects/mypro/get_useful/ijcai_clean/results/task6_base_instruct_full_vocab` 中的 base→instruct 全词表分析：`embed_tokens` / `lm_head` 的变化基本满足 hidden 维仿射关系。因此本目录关注的是在 SFT / post-training 中，让这些词表层以低参数量参与训练。

## 目录布局

```text
lora/
  data/
    sft_t2t_mini_25k/           # 主任务：24k train + 1k eval
    ultrachat_100k/             # UltraChat 扩展实验
    minimind_random_100k/       # MiniMind 扩展实验
  docs/
    AFFINE_VOCAB_MAIN_EXPERIMENT.md
    EXPERIMENT_SETTINGS.md
    RESULTS_SO_FAR.md
    CLEANUP_MANIFEST.md
  outputs/
    affine_vocab/
  scripts/
    train_affine_vocab_lora.py  # 主训练入口
    eval_base_loss.py           # Claim 2 frozen-base 参照
    eval_merge_equivalence.py   # merge 等价性验证
    run_affine_vocab_smoke.sh   # smoke test
    run_syntax_check.sh         # 语法检查
  src/
    affine_vocab_lora/adapter.py
```

## 主训练入口

```text
scripts/train_affine_vocab_lora.py
```

## 模型

主线使用 `/home/wz/projects/mypro/im_exp/models` 下的 Qwen base 模型。

| ID | 路径 | task6 R² | 角色 |
|---|---|---:|---|
| `qwen25_0_5b` | `/home/wz/projects/mypro/im_exp/models/Qwen2.5-0.5B-Base` | 0.9903 | smoke |
| `qwen3_0_6b` | `/home/wz/projects/mypro/im_exp/models/Qwen3-0.6B-Base` | 0.9877 | 快速 sweep |
| `qwen25_1_5b` | `/home/wz/projects/mypro/im_exp/models/Qwen2.5-1.5B-Base` | 0.9997 | headline 主模型 |
| `qwen3_1_7b` | `/home/wz/projects/mypro/im_exp/models/Qwen3-1.7B-Base` | 0.9938 | 后续主模型 |
| `qwen25_3b` | `/home/wz/projects/mypro/im_exp/models/Qwen2.5-3B-Base` | 0.9997 | 后续放大 |
| `qwen3_4b` | `/home/wz/projects/mypro/im_exp/models/Qwen3-4B-Base` | 0.9901 | 后续放大 |
| `qwen25_7b` | `/home/wz/projects/mypro/im_exp/models/Qwen2.5-7B-Base` | 未测 | 大模型后续 |
| `qwen3_8b` | `/home/wz/projects/mypro/im_exp/models/Qwen3-8B-Base` | 未测 | 大模型后续 |

## 数据

| 用途 | 路径 | 状态 |
|---|---|---|
| 主训练 | `data/sft_t2t_mini_25k/train.jsonl` | 主任务 |
| 主评估 | `data/sft_t2t_mini_25k/eval.jsonl` | headline `eval_loss` |
| 扩展实验 | `data/ultrachat_100k/`, `data/minimind_random_100k/` | 活跃 |

## Variant

| Variant | 训练内容 | 用途 |
|---|---|---|
| `hidden_lora` | 所有层 hidden LoRA | Claim 1a baseline |
| `affine_input_plus_hidden_lora` | hidden LoRA + `embed_tokens` AffLoRA | Claim 1a 输入侧处理组 |
| `affine_input_lm_head_plus_hidden_lora` | hidden LoRA + `embed_tokens` / `lm_head` AffLoRA | Claim 1a 完整处理组 |
| `affine_input` | 仅 `embed_tokens` AffLoRA | Claim 2 / 3 |
| `affine_input_lm_head` | `embed_tokens` + `lm_head` AffLoRA | Claim 2 / 3 |
| `affine_lm_head` | 仅 `lm_head` AffLoRA | 少量 sanity |

控制组：

| 控制组 | 用途 | 参数 |
|---|---|---|
| emb/lm_head vocab 维 LoRA | Claim 1b | `--include-emb-lmh-lora-rank` |
| 单层 LoRA | Claim 3 | `--hidden-lora-layers-to-transform` |
| frozen base eval_loss | Claim 2 | `scripts/eval_base_loss.py` |

## Headline 超参

| 参数 | 值 |
|---|---|
| 任务 | `sft_t2t_mini` |
| `max_seq_len` | 1024 |
| epoch | 1 |
| batch | `8 × 2`，有效 batch 16 |
| learning rate | `2e-4` |
| schedule | cosine，warmup 0.03 |
| hidden LoRA | rank 8，alpha 16 |
| AffLoRA | rank 16，alpha 128，s1=8 |
| eval | 1k holdout，每 250 step |
| trainable master dtype | fp32 |
| seed | 42；Phase 4c 增加 43、44 |

## 精度要求

训练时应保持：

```text
--master-dtype fp32 --base-dtype auto --bf16
```

原因是 trainable 参数若以 bf16 作为 master 权重，`lr=2e-4` 的小更新可能被舍入掉。训练脚本启动时会打印 trainable 参数 dtype，结束时会打印 Adam state dtype，两处都应显示 fp32。

## 验证命令

```bash
source /home/wz/projects/mypro/im_exp/set
cd /home/wz/projects/mypro/im_exp/lora
bash scripts/sh/run_syntax_check.sh
```

## 保留但非主线

- `phase3c_*`：位置消融，只回答 affine 放置位置。
- `phase3d_*`：DoRA / rsLoRA 对照，只作为 LoRA 超参背景。
