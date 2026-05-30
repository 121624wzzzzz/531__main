# Phase 9 后续补强实验总结

快照时间：2026-05-21。除特别说明外，所有实验均使用 `data/sft_t2t_mini_25k/train.jsonl` 训练 1 epoch，在 `data/sft_t2t_mini_25k/eval.jsonl` 上取最终 `eval_loss`，越低越好。

## 实验内容与设置

| phase | 目的 | 模型 | 设置 |
|---|---|---|---|
| Phase 9a | 补齐 7B / 8B vocab-dim LoRA 多 seed 对照 | Qwen2.5-7B、Qwen3-8B | seed 43/44，hidden r1/r2 + `embed_tokens`/`lm_head` vocab LoRA r1 |
| Phase 9b / 9e | 补 Qwen2.5-1.5B low-rank 三方对照并做多 seed | Qwen2.5-1.5B | seed 42/43/44，hidden r1/r2、hidden+AffLoRA、hidden+vocab LoRA r1 |
| Phase 9c | AffLoRA alpha 鲁棒性 | Qwen3-1.7B、Qwen2.5-7B | hidden r1/r2，AffLoRA rank 16，`affine_alpha` 64/128/256 |
| Phase 9d | input-side / lm_head-side 位置归因 | Qwen2.5-7B | hidden r1/r2，对比 hidden-only、input-only、lm_head-only、input+lm_head |
| Phase 9f | AffLoRA bias scale 鲁棒性 | Qwen3-1.7B | hidden r1/r2，AffLoRA rank 16，alpha 128，`affine_bias_scale` 0/0.5/1/2 |

## 关键结果

### 7B / 8B Claim 1b 多 seed

Phase 9a 将 Phase 8a 的 seed42 vocab LoRA 对照扩展到 seed43/44，并与 Phase 7c 的 AffLoRA 三 seed 结果对齐。

| model | rank | hidden + vocab LoRA r1 mean | hidden + AffLoRA mean | AffLoRA advantage |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | 1 | 0.9639 | 0.9576 | -0.0063 |
| Qwen2.5-7B | 2 | 0.9503 | 0.9457 | -0.0046 |
| Qwen3-8B | 1 | 0.9062 | 0.9001 | -0.0061 |
| Qwen3-8B | 2 | 0.8963 | 0.8907 | -0.0056 |

结论：传统 vocab-dim LoRA r1 在 7B / 8B 上只带来小幅增益，且 3/3 seed 均弱于 AffLoRA。AffLoRA 参数还略少，因此 Claim 1b 的低 rank 大模型证据已经闭合。

### Qwen2.5-1.5B 三方多 seed

Phase 9b 是 seed42，Phase 9e 补 seed43/44。

| rank | hidden mean | hidden + vocab LoRA r1 mean | hidden + AffLoRA mean | AffLoRA vs hidden | AffLoRA vs vocab |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.1267 | 1.1200 | 1.1133 | -0.0134 | -0.0067 |
| 2 | 1.1043 | 1.0987 | 1.0953 | -0.0090 | -0.0034 |

结论：Qwen2.5-1.5B 在 r1/r2、3 个 seed 上均表现为 AffLoRA 最优，补上了 Qwen2.5 系列 0.5B -> 1.5B -> 7B 的尺寸链。

### Alpha / Scale 鲁棒性

Phase 9c 固定 AffLoRA rank 16，扫 `affine_alpha`。

| model | rank | alpha 64 | alpha 128 | alpha 256 | 结论 |
|---|---:|---:|---:|---:|---|
| Qwen3-1.7B | 1 | 0.779 | 0.777 | 0.776 | 256 略优，差距很小 |
| Qwen3-1.7B | 2 | 0.766 | 0.764 | 0.763 | 256 略优，差距很小 |
| Qwen2.5-7B | 1 | 0.959 | 0.958 | 0.956 | 256 略优，差距很小 |
| Qwen2.5-7B | 2 | 0.947 | 0.946 | 0.944 | 256 略优，差距很小 |

Phase 9f 固定 alpha 128，扫 `affine_bias_scale`。

| model | rank | scale 0 | scale 0.5 | scale 1 | scale 2 |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 1 | 0.7776 | 0.7773 | 0.7767 | 0.7774 |
| Qwen3-1.7B | 2 | 0.7642 | 0.7641 | 0.7636 | 0.7637 |

结论：alpha 64/128/256 均有效，256 略优但不是质变；bias scale 在 0-2 范围内几乎不敏感。默认 `rank=16, alpha=128, bias_scale=1` 不是偶然甜点。

### 位置归因

Phase 9d 在 Qwen2.5-7B 上对比 input-side 与 lm_head/output-side。

| rank | hidden-only | input-only | lm_head-only | input+lm_head |
|---:|---:|---:|---:|---:|
| 1 | 0.966 | 0.965 | 0.959 | 0.957 |
| 2 | 0.952 | 0.951 | 0.946 | 0.946 |

结论：低 rank 下收益主要来自输出侧 `lm_head` affine；input-only 增益很小，input+lm_head 与 lm_head-only 接近但略好或打平。这与 1.5B / 3B / 7B 既有观察一致。

## 总体判断

Phase 9 后，核心结论已经比较充分：

- Claim 1a：完整 `input+lm_head` AffLoRA 在 0.5B、1.5B、1.7B、3B、4B、7B、8B 上均有正向证据；低 rank 大模型三 seed 也稳定。
- Claim 1b：AffLoRA 不只是优于 hidden-only，也在 1.5B、1.7B、7B、8B 上优于同口径 vocab-dim LoRA；0.5B 与 vocab LoRA 打平但参数更少。
- 低 rank 场景是 AffLoRA 的最强叙事：hidden LoRA rank 越低，AffLoRA 的边际收益越明显。
- 机制上，主要收益来自 output-side / `lm_head`，input-side 是次要或不稳定贡献。
- 超参上，alpha 与 bias scale 都不敏感，默认设置可作为稳健默认。

后续如果继续做，边际价值最高的是非 loss 生成质量抽样；大模型 full fine-tuning 成本高且不直接增强 AffLoRA 参数效率主张，不建议优先。
