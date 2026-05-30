# AffLoRA 主实验设计

本文是 `lora` 目录的实验设计主文档。项目核心是 **AffLoRA：基于 base→instruct 全词表仿射关系，让 SFT / post-training 中的 `embed_tokens` 和 `lm_head` 低成本参与训练**。日常运行参数见 `docs/EXPERIMENT_SETTINGS.md`，已完成结果见 `docs/RESULTS_SO_FAR.md`。

## 研究动机

在 SFT、instruction tuning、domain adaptation 等后训练流程中，标准 LoRA 通常只训练 transformer block 内部的投影层，`embed_tokens` 和 `lm_head` 往往被跳过。原因很直接：这两个矩阵是 `vocab × hidden` 规模，直接训练或在词表维做 LoRA 参数开销很高。

但 `/home/wz/projects/mypro/get_useful/ijcai_clean/results/task6_base_instruct_full_vocab` 的 base/instruct 全词表分析显示，Qwen2.5 / Qwen3 多数组合中，base 模型到 instruct 模型的 embedding / 输出矩阵变化基本符合 hidden 维仿射关系：

```text
W_instruct ≈ W_base · A + b
```

这说明后训练中的词表侧变化未必需要为每个 token 独立学习大规模更新；可以用一个 hidden-space affine map 捕捉主要变化。AffLoRA 的目标就是把这个分析结果变成可训练模块，使 `embed_tokens` 以及 `lm_head` / `llm_head` 在后训练中也能低成本参与优化。

本项目采用 hidden 维低秩仿射适配器：

```text
W' = W (I + s1 · A B) + s2 · b
```

其中 `AB` 是 `hidden × hidden` 低秩矩阵，`b` 是 hidden 维偏置。参数量约为 `O(2 · hidden · rank + hidden)`，与词表大小无关。

实际实现不直接修改冻结的权重矩阵，而是在 hidden state 上加适配：

```text
# 输入侧：包装 embed_tokens 输出
e' = e + s1 · U_in(D_in(e)) + s2 · b_in

# 输出侧：包装进入 lm_head 前的 hidden state
h' = h + s1 · U_out(D_out(h)) + s2 · b_out
```

基座 `embed_tokens.weight` 和 `lm_head.weight` 保持冻结，只训练 AffLoRA 参数。对于 tied 模型，`lm_head` 和 embedding 权重本身绑定，但输入侧和输出侧仍然可以分别在 hidden state 上放置适配器；对于非 tied LLM，则对应适配 `embed_tokens` 和独立的 `lm_head` / `llm_head`。默认不启用 `lm_head` hidden 偏置，因为进入 `lm_head` 前的统一 hidden 偏移只会对所有词表 logit 产生共同平移，softmax 后会抵消；如需真正的词表 logit 偏置，可用 `--vocab-logit-bias` 做消融。

## 三条核心主张

目录中的主实验、基线和控制组都应服务以下三条主张。不能服务这些主张的内容应标为补充或归档，不进入 headline 结论。

### Claim 1：完整架构有效

完整架构是标准 transformer-block hidden LoRA 加上 `embed_tokens` / `lm_head` 的 AffLoRA。它要证明的不是“替代普通 LoRA”，而是补上普通 LoRA 后训练里词表层通常无法高效参与训练的问题。

1a 关注是否优于纯 hidden LoRA：

```text
hidden_lora + affine_emb + affine_lm_head  >  hidden_lora
```

1b 关注 hidden 维 affine 是否是更合适的 emb/lm_head 参数化方式。对同样的 emb/lm_head 适配范围，如果改用传统 vocab 维 LoRA，它应当被 AffLoRA 追平或超过，同时 AffLoRA 使用更少参数：

```text
hidden_lora + affine_emb + affine_lm_head
  > hidden_lora + LoRA_emb + LoRA_lm_head
```

### Claim 2：Affine 模块本身有用

即使不训练 hidden-layer LoRA，只训练 emb/lm_head 侧 AffLoRA，也应明显优于完全冻结的 base model：

```text
affine_emb + affine_lm_head  >  base, no training
```

这用于证明来自 task6 仿射假设的词表层适配本身在工作，而不是只搭了 hidden LoRA 的便车。

### Claim 3：小参数预算下更高效

在约 `30k-100k` 的小参数预算下，AffLoRA 应优于把相同预算投到少量 transformer 层的 LoRA：

```text
affine_emb (~33k params)  >  single-layer LoRA (~33k-50k params)
```

## 主线 Variant

| Variant | 训练内容 | 用途 |
|---|---|---|
| `hidden_lora` | 所有层的 `q/k/v/o/up/down/gate` LoRA | Claim 1a 基线 |
| `affine_input_plus_hidden_lora` | hidden LoRA + `embed_tokens` AffLoRA | Claim 1a 输入侧处理组 |
| `affine_input_lm_head_plus_hidden_lora` | hidden LoRA + `embed_tokens` / `lm_head` AffLoRA | Claim 1a 完整处理组 |
| `affine_input` | 仅 `embed_tokens` AffLoRA | Claim 2 / 3 |
| `affine_input_lm_head` | `embed_tokens` + `lm_head` AffLoRA，无 hidden LoRA | Claim 2 / 3 |

控制组：

| 控制组 | 服务主张 | 实现方式 |
|---|---|---|
| emb/lm_head 上的 vocab 维 LoRA | Claim 1b | `--include-emb-lmh-lora-rank` |
| 单层或少量层 LoRA | Claim 3 | `--hidden-lora-layers-to-transform` |
| 冻结 base eval_loss | Claim 2 | `scripts/eval_base_loss.py` |

## 主模型

| ID | 路径 | task6 R² | 角色 |
|---|---|---:|---|
| `qwen25_0_5b` | `/home/wz/projects/mypro/im_exp/models/Qwen2.5-0.5B-Base` | 0.9903 | smoke |
| `qwen3_0_6b` | `/home/wz/projects/mypro/im_exp/models/Qwen3-0.6B-Base` | 0.9877 | 快速 sweep |
| `qwen25_1_5b` | `/home/wz/projects/mypro/im_exp/models/Qwen2.5-1.5B-Base` | 0.9997 | headline 主模型 |
| `qwen3_1_7b` | `/home/wz/projects/mypro/im_exp/models/Qwen3-1.7B-Base` | 0.9938 | 后续主模型 |
| `qwen25_3b` | `/home/wz/projects/mypro/im_exp/models/Qwen2.5-3B-Base` | 0.9997 | 后续放大 |
| `qwen3_4b` | `/home/wz/projects/mypro/im_exp/models/Qwen3-4B-Base` | 0.9901 | 后续放大 |

当前最强证据来自 `qwen25_1_5b`。

## 主任务

| 任务 | 状态 | 说明 | 切分 |
|---|---|---|---|
| `sft_t2t_mini` | 主任务 | 通用中文 SFT，能体现 base 到 instruct 的词表侧变化，是 AffLoRA 的目标场景 | `data/sft_t2t_mini_25k/train.jsonl` 24k train，`data/sft_t2t_mini_25k/eval.jsonl` 1k eval |

所有 claim 结论都只基于 `sft_t2t_mini` 的 `eval_loss`。旧的 MetaMathQA/GSM8K/MATH 生成评测已经退出主流程，因为 Phase 1 / 1.5 显示其在当前预算下几乎不能区分 variant。

## Headline 超参

| 参数 | 值 |
|---|---|
| `max_seq_len` | 1024 |
| batch | `per_device_train_batch_size=8`，`gradient_accumulation_steps=2`，有效 batch 16 |
| learning rate | `2e-4` |
| schedule | cosine，`warmup_ratio=0.03` |
| epoch | 1 |
| affine | rank 16，alpha 128，`s1=8` |
| hidden LoRA | rank 8，alpha 16 |
| precision | frozen base bf16/auto，trainable 参数 fp32 master |
| seed | 42；Phase 4c 额外使用 43、44 |

headline 实验通过 `scripts/run_phase*.sh` 直接调用 `train_affine_vocab_lora.py`。

## 补充和归档边界

以下内容不进入三条 claim 的核心结论：

| 内容 | 当前状态 |
|---|---|
| position ablation：layer 3/10/17/24/final norm | 补充实验，代码仍在 `AffineLayerWrapper` / `AffineAfterNorm`，输出在 `phase3c_*` |
| DoRA / rsLoRA 对照 | 补充实验，输出在 `phase3d_*` |
| LoRMA 论文设置、GLUE/E2E、math generation eval | 已从 active scripts/configs 移除，只保留历史数据和结果摘要 |

## 常用命令

Smoke：

```bash
source /home/wz/projects/mypro/im_exp/set
cd /home/wz/projects/mypro/im_exp/lora
bash scripts/run_affine_vocab_smoke.sh
```

1.5B headline 单 run：

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
