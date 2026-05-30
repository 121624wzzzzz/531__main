# AffLoRA 实验

本目录用于验证 **AffLoRA**：基于 `/home/wz/projects/mypro/get_useful/ijcai_clean/results/task6_base_instruct_full_vocab` 中 base→instruct 全词表仿射关系的分析结果，让 SFT / post-training 中通常被跳过的 `embed_tokens` 和 `lm_head` 以很小参数量参与训练。

AffLoRA 的定位不是替代 transformer block 上的普通 LoRA，而是补上后训练里词表层难以高效训练的问题。对于 tied 模型，`lm_head` 和 embedding 权重绑定，但仍可在输入侧与输出侧 hidden state 上分别放置适配；对于非 tied LLM，则对应适配 `embed_tokens` 和独立的 `lm_head` / `llm_head`。

核心文档：

```text
docs/AFFINE_VOCAB_MAIN_EXPERIMENT.md   # 实验设计与三条主张
docs/EXPERIMENT_SETTINGS.md            # 路径、模型、超参、运行方式
docs/RESULTS_SO_FAR.md                 # 当前结果审计
docs/CLEANUP_MANIFEST.md               # 清理与归档记录
```

## 当前主线

Headline 任务是 `sft_t2t_mini_25k`，单卡训练，hidden LoRA r=8，AffLoRA rank=16 / alpha=128。三条主张是：

1. 完整架构有效：`hidden_lora + affine_emb + affine_lm_head` 优于纯 `hidden_lora`。
2. AffLoRA 模块本身有效：只训练 emb/lm_head 侧 AffLoRA 也明显优于 frozen base。
3. 参数效率更高：同等小参数预算下，AffLoRA 优于 vocab 维 LoRA 或少量层 LoRA。

旧的 math generation 评测、LoRMA 论文设置、GLUE/E2E 对照已经从 active workflow 移除。历史数据和结果只作为审计/背景保留。

## 目录结构

```text
data/      主任务 sft_t2t_mini_25k，以及 UltraChat / MiniMind 扩展数据
docs/      实验设计、运行设置、结果审计、清理记录
outputs/   训练和评估产物
scripts/   主训练、base eval、phase launcher
src/       affine_vocab_lora / AffLoRA 核心实现
```

## 环境

```bash
source /home/wz/projects/mypro/im_exp/set
cd /home/wz/projects/mypro/im_exp/lora
export PYTHONPATH=/home/wz/projects/mypro/im_exp/lora/src:${PYTHONPATH:-}
```

Qwen base 模型位于 `/home/wz/projects/mypro/im_exp/models`。

## Smoke

```bash
bash scripts/sh/run_affine_vocab_smoke.sh
```

该命令在 Qwen3-0.6B 上跑 2 step，用于验证工具链和精度打印。训练日志中应看到 trainable 参数与 Adam state 为 fp32，避免 bf16 master 权重导致小更新被舍入。

## 主训练示例

```bash
python scripts/train_affine_vocab_lora.py \
  --model-path /home/wz/projects/mypro/im_exp/models/Qwen2.5-1.5B-Base \
  --train-data data/sft_t2t_mini_25k/train.jsonl \
  --eval-data data/sft_t2t_mini_25k/eval.jsonl --eval-samples 1000 --eval-steps 250 \
  --output-dir outputs/affine_vocab/headline/qwen25_1_5b/affine_input_lm_head_plus_hidden_lora \
  --variant affine_input_lm_head_plus_hidden_lora \
  --hidden-lora-rank 8 --hidden-lora-alpha 16 \
  --affine-rank 16 --affine-alpha 128 \
  --max-seq-len 1024 --per-device-train-batch-size 8 --gradient-accumulation-steps 2 \
  --learning-rate 2e-4 --lr-scheduler-type cosine --warmup-ratio 0.03 \
  --num-train-epochs 1 --save-strategy no \
  --master-dtype fp32 --base-dtype auto --bf16 --seed 42
```

常用控制组：

```bash
# Claim 2：frozen base 参照
python scripts/eval_base_loss.py \
  --model-path /home/wz/projects/mypro/im_exp/models/Qwen2.5-1.5B-Base \
  --eval-data data/sft_t2t_mini_25k/eval.jsonl \
  --report-file outputs/affine_vocab/claim2_base/qwen25_1_5b.json

# Claim 1b：emb/lm_head 上的 vocab 维 LoRA
bash scripts/run_phase4b_claim1b.sh

# Claim 1a：1.5B 多 seed 稳定性
bash scripts/run_phase4c_multiseed_15b.sh
```

## 当前结果

截至 2026-05-19，三条主张均已有支持性证据：

- Claim 1a：Qwen2.5-1.5B 三 seed 上 `combined_inlmh` 平均 `eval_loss` 1.048，baseline 1.054。
- Claim 1b：1.5B 上 100k AffLoRA 略优于 307k vocab-dim LoRA r=1。
- Claim 2：AffLoRA-only 相比 frozen base 降低约 0.5+ `eval_loss`。
- Claim 3：33k AffLoRA-only 明显优于 33k-49k single-layer LoRA。

完整表格见 `docs/RESULTS_SO_FAR.md`。
