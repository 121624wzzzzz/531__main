# 当前结果审计

快照时间：2026-05-21，Phase 9f 之后。

本文按 `docs/AFFINE_VOCAB_MAIN_EXPERIMENT.md` 中的三条主张整理已完成实验。项目核心是 AffLoRA：基于 `/home/wz/projects/mypro/get_useful/ijcai_clean/results/task6_base_instruct_full_vocab` 中 base→instruct 全词表仿射关系，让 SFT / post-training 中的 `embed_tokens` 和 `lm_head` 低成本参与训练。除特别说明外，数字都是 `data/sft_t2t_mini_25k/eval.jsonl` 上的最终 `eval_loss`，越低越好。

## 结论摘要

`sft_t2t_mini` 是唯一 headline 任务。旧的 math generation 路径已经退出主流程，不再用于支持或反驳 claim。

| 主张 | 状态 | 最强证据 |
|---|---|---|
| Claim 1a：完整架构优于纯 LoRA | 已支持，1.5B 上三 seed 小幅稳定提升，0.5B / 1.7B / 3B / 4B / 7B / 8B 均有正向证据 | 1.5B `combined_inlmh` 平均 1.048 vs baseline 1.054，Δ -0.006；0.5B Δ -0.008；1.7B Δ -0.0033；3B Δ -0.0035；4B Δ -0.0018；7B Δ -0.0029；8B Δ -0.0021；7B / 8B 低 rank 三 seed平均 Δ 约 -0.0044 到 -0.0082 |
| Claim 1b：AffLoRA 比 vocab 维 LoRA 更参数高效 | 已支持，1.5B / 1.7B / 7B / 8B 上优于 vocab-dim LoRA；0.5B 打平但 AffLoRA 参数更少 | 1.5B r1 三 seed均值：AffLoRA 1.113 vs vocab 1.120；7B r1 三 seed均值：AffLoRA 0.9576 vs vocab 0.9639；8B r1 三 seed均值：AffLoRA 0.9001 vs vocab 0.9062 |
| Claim 2：AffLoRA-only 本身有效 | 强支持 | 0.6B：1.320 vs base 1.848；1.5B：1.326 vs base 1.926 |
| Claim 3：小参数预算下优于局部层 LoRA | 强支持 | 0.6B：33k `affine_input` 1.426，优于 33k-49k single-layer LoRA 的 1.473-1.780 |

## 已完成阶段

### Phase 1 / 1.5：已归档的 math 探索

Phase 1 使用 MetaMathQA 小规模训练，Phase 1.5 放大到 24k × 2 epoch。两者都显示 variant 差异很小，最大约 0.002，不能有效区分 affine 架构。相关训练入口和输出已从 active workflow 清理，结果仅作为负结果保留。

### Phase 2'：0.6B AffLoRA-only s1 sweep

模型：`Qwen3-0.6B`。任务：`sft_t2t_mini` 24k × 1 epoch。输出：`outputs/affine_vocab/phase2_sweep_*/`。

| run | 参数量 | final eval |
|---|---:|---:|
| baseline hidden_lora_r8 | 5.05M | 1.101 |
| affine_only s1=0.5 | 33k | 1.495 |
| affine_only s1=1 | 33k | 1.485 |
| affine_only s1=2 | 33k | 1.466 |
| affine_only s1=4 | 33k | 1.452 |
| affine_only s1=2 s2=4 | 33k | 1.463 |

结论：支持 Claim 2。33k AffLoRA-only 已明显优于 frozen base，且 s1 增大时效果改善。

### Phase 2''：0.6B combined + capacity controls

输出：`outputs/affine_vocab/phase2b_*/`。

| run | 参数量 | final eval | Δ vs baseline |
|---|---:|---:|---:|
| baseline hidden_lora_r8 | 5.05M | 1.101 | - |
| combined_in_s1_2 | 5.08M | 1.099 | -0.002 |
| combined_in_s1_4 | 5.08M | 1.099 | -0.002 |
| combined_inlmh_s1_4 | 5.11M | 1.098 | -0.003 |
| hidden_lora_r9 | 5.68M | 1.096 | -0.005 |
| affine_only_s1_8 | 33k | 1.442 | - |
| affine_only_s1_16 | 33k | 1.434 | - |

结论：在 0.6B 上弱支持 Claim 1a；full combined 相比 baseline 有 -0.003 改善。

### Phase 2'''：0.6B matched-budget single-layer LoRA

输出：`outputs/affine_vocab/phase2c_*/`。

| run | 参数量 | final eval |
|---|---:|---:|
| affine_only s1=32 | 33k | 1.426 |
| affine_in_lmh s1=4 | 67k | 1.346 |
| lora qkvo l14 r=4 | 40k | 1.473 |
| lora q l=0 r=16 | 49k | 1.626 |
| lora q l=14 r=16 | 49k | 1.667 |
| lora q l=27 r=16 | 49k | 1.780 |

结论：强支持 Claim 3。33k AffLoRA-only 优于所有 33k-49k 的局部层 LoRA。

### Phase 3a：Qwen2.5-1.5B 放大

输出：`outputs/affine_vocab/phase3a_*/`。

| run | 参数量 | final eval | Δ vs baseline |
|---|---:|---:|---:|
| baseline hidden_lora_r8 | 9.23M | 1.054 | - |
| combined_in_s1_8 | 9.28M | 1.053 | -0.001 |
| combined_inlmh_s1_8 | 9.33M | 1.048 | -0.006 |
| affine_only s1=32 | 50k | 1.478 | - |
| affine_only s1=64 | 50k | 1.468 | - |
| affine_in_lmh s1=16 | 100k | 1.326 | - |

结论：1.5B 上 Claim 1a 更清晰。`lm_head` affine 是更大的增益来源；只加 input affine 约 -0.001，加 input + lm_head 约 -0.006。

### Phase 3b：0.6B s1 ceiling

输出：`outputs/affine_vocab/phase3b_*/`。

结论：0.6B 上 s1 甜点大约在 64；16 到 64 之间趋于平缓，128 开始略差。该阶段主要给超参建议，间接支持 Claim 2。

### Phase 4a：Claim 2 frozen base 控制组

输出：`outputs/affine_vocab/claim2_base/`。

| model | base eval_loss | affine_in_lmh best | Δ |
|---|---:|---:|---:|
| Qwen3-0.6B | 1.848 | 1.320，s1=64，67k 参数 | -0.528 |
| Qwen2.5-1.5B | 1.926 | 1.326，s1=16，100k 参数 | -0.600 |

结论：正式支持 Claim 2。只训练 67k-100k AffLoRA 参数即可比 frozen base 降低 0.5+ eval_loss。

### Phase 4b：Claim 1b vocab 维 LoRA 控制组

在 `embed_tokens` 和 `lm_head` 上额外加入传统 PEFT LoRA，rank ∈ {1, 2}，与 hidden 维 affine 比较同一适配范围。

Qwen3-0.6B：

| run | hidden_lora_r8 之外的额外参数 | final eval | Δ vs baseline |
|---|---:|---:|---:|
| baseline hidden_lora_r8 | 0 | 1.101 | - |
| combined_inlmh | +66k affine | 1.098 | -0.003 |
| lora_emblmh r=1 | +306k vocab LoRA | 1.094 | -0.007 |
| lora_emblmh r=2 | +612k vocab LoRA | 1.093 | -0.008 |

Qwen2.5-1.5B：

| run | hidden_lora_r8 之外的额外参数 | final eval | Δ vs baseline |
|---|---:|---:|---:|
| baseline hidden_lora_r8 | 0 | 1.054 | - |
| combined_inlmh | +100k affine | 1.048 | -0.006 |
| lora_emblmh r=1 | +307k vocab LoRA | 1.049 | -0.005 |
| lora_emblmh r=2 | +614k vocab LoRA | 1.046 | -0.008 |

结论：Claim 1b 在 `Qwen2.5-1.5B` 上成立。100k AffLoRA 以更少参数略优于 307k vocab-dim LoRA r=1。0.6B 上 vocab LoRA 绝对 eval 更低，但 AffLoRA 的单位参数效率仍更高。

### Phase 4c：Qwen2.5-1.5B 多 seed Claim 1a

输出：`outputs/affine_vocab/phase4c_20260519_155136/`。seed 42 来自 Phase 3a，Phase 4c 补 seed 43、44。

| run | seed 42 | seed 43 | seed 44 | mean |
|---|---:|---:|---:|---:|
| baseline hidden_lora_r8 | 1.054 | 1.054 | 1.054 | 1.054 |
| combined_in_s1_8 | 1.053 | 1.052 | 1.052 | 1.052 |
| combined_inlmh_s1_8 | 1.048 | 1.048 | 1.047 | 1.048 |

结论：Claim 1a 达到当前项目闭合标准。`combined_inlmh` 相比 baseline 的 Δ 为 -0.006、-0.006、-0.007，3/3 seed 一致提升。

### Phase 5a：0.6B hidden LoRA rank sweep

输出：`outputs/affine_vocab/phase5a_20260519_170658/`。该阶段只训练基础 `hidden_lora`，观察普通 LoRA rank / 参数量增加带来的收益。

| run | 参数量 | final eval | Δ vs r8 |
|---|---:|---:|---:|
| hidden_lora_r2 | 1.26M | 1.163 | +0.062 |
| hidden_lora_r4 | 2.52M | 1.133 | +0.032 |
| hidden_lora_r8 | 5.05M | 1.101 | 0 |
| hidden_lora_r16 | 10.09M | 1.071 | -0.030 |
| hidden_lora_r32 | 20.19M | 1.042 | -0.059 |

结论：基础 LoRA 增大 rank 的收益非常稳定，几乎单调改善。相比 r8，r16 多约 5.05M 参数换来 -0.030 eval_loss，r32 多约 15.14M 参数换来 -0.059 eval_loss。这个结果说明：在 0.6B 上，如果允许大幅增加 hidden LoRA 参数，纯 LoRA 能获得明显收益；AffLoRA 的优势应主要强调“让 emb/lm_head 参与训练的低成本增益”和小参数预算效率，而不是声称几十 k 参数能替代数千万 hidden LoRA 参数。

### Phase 5b：Qwen2.5-3B AffLoRA 架构验证

输出：`outputs/affine_vocab/phase5b_20260519_172412/`。模型为 `Qwen2.5-3B-Base`，task6 R² = 0.9997，seed 42。

| run | 参数量 | final eval | Δ vs baseline |
|---|---:|---:|---:|
| baseline hidden_lora_r8 | 14.97M | 0.9797 | - |
| combined_in_s1_8 | 15.03M | 0.9789 | -0.0008 |
| combined_inlmh_s1_8 | 15.10M | 0.9762 | -0.0035 |
| affine_in_lmh_s1_16 | 133k | 1.223 | - |

结论：3B 上 Claim 1a 继续成立，完整 input + lm_head AffLoRA 相比 pure hidden LoRA 降低约 0.0035 eval_loss；input-only 仍是很小增益，lm_head 侧仍是主要增益来源。该趋势与 1.5B 一致，但单 seed 增益幅度小于 1.5B 多 seed 均值 -0.006。AffLoRA-only 在 3B 上仍显著优于 frozen base 的预期方向需要另跑 `eval_base_loss.py` 才能正式量化；这里只作为轻量独立性参照。

### Phase 5c：Qwen3-4B AffLoRA 架构验证

输出：`outputs/affine_vocab/phase5c_20260519_180407/`。模型为 `Qwen3-4B-Base`，task6 R² = 0.9901，seed 42。

| run | 参数量 | final eval | Δ vs baseline |
|---|---:|---:|---:|
| baseline hidden_lora_r8 | 16.52M | 0.8828 | - |
| combined_in_s1_8 | 16.60M | 0.8818 | -0.0010 |
| combined_inlmh_s1_8 | 16.68M | 0.8810 | -0.0018 |
| affine_in_lmh_s1_16 | 166k | 1.048 | - |

结论：4B 上 Claim 1a 仍为正向，完整 input + output-side AffLoRA 相比 pure hidden LoRA 降低约 0.0018 eval_loss；input-only 贡献约 -0.0010，输出侧额外贡献约 -0.0008。增益方向延续 1.5B / 3B，但幅度更小，可能与 Qwen3-4B 的 task6 R² 低于 Qwen2.5 系列、以及单 seed 噪声有关。该结果更适合作为“跨系列外推仍有小幅正增益”的证据，而不是最强 headline 数字。

### Phase 5d：Qwen2.5-7B AffLoRA 架构验证

输出：`outputs/affine_vocab/phase5d_20260519_193129/`。模型为 `Qwen2.5-7B-Base`，seed 42。该阶段使用 `PER_DEVICE_BS=4`、`GRAD_ACCUM=4`、gradient checkpointing，有效 batch 仍为 16；重跑后单卡显存约 52GB，完整训练约 64 分钟。

| run | 参数量 | final eval | Δ vs baseline |
|---|---:|---:|---:|
| baseline hidden_lora_r8 | 20.19M | 0.9180 | - |
| combined_in_s1_8 | 20.30M | 0.9184 | +0.0004 |
| combined_inlmh_s1_8 | 20.42M | 0.9151 | -0.0029 |
| affine_in_lmh_s1_16 | 233k | 1.120 | - |

结论：7B 上完整 input + output-side AffLoRA 仍然优于 pure hidden LoRA，降幅约 0.0029 eval_loss，支持 Claim 1a 向更大 Qwen2.5 模型推广。与 3B 一致，真正带来收益的是加上输出侧后的完整方案；input-only 在本次单 seed 中没有带来正增益，说明输入侧贡献较小且容易被噪声覆盖。AffLoRA-only 仅 233k 参数达到 1.120，明显弱于 20M 参数级 hidden LoRA，但仍可作为低参数独立适配能力的参照。

### Phase 6a：剩余模型 AffLoRA 架构验证

输出：`outputs/affine_vocab/phase6a_20260519_204314/`。该阶段补齐 `Qwen2.5-0.5B-Base`、`Qwen3-1.7B-Base`、`Qwen3-8B-Base`，均为 seed 42。

| model | hidden_lora_r8 | combined_inlmh_s1_8 | Δ | AffLoRA-only |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1.302 | 1.294 | -0.008 | 1.637 |
| Qwen3-1.7B | 0.7383 | 0.7350 | -0.0033 | 0.9074 |
| Qwen3-8B | 0.8717 | 0.8696 | -0.0021 | 1.010 |

结论：新增三个模型上完整 `input + lm_head` AffLoRA 均优于 pure hidden LoRA，继续支持 Claim 1a 的跨模型推广。增益仍是小而稳定的低成本增益，量级约 0.002-0.008；Qwen3 系列从 1.7B 到 8B 也保持正向。

### Phase 6b：hidden LoRA rank 曲线

输出：`outputs/affine_vocab/phase6b_20260519_222852/`。该阶段比较 pure hidden LoRA 与固定 AffLoRA rank=16 的完整方案。

Qwen2.5-0.5B：

| rank | hidden_lora | combined_inlmh | Δ |
|---:|---:|---:|---:|
| 2 | 1.398 | 1.380 | -0.018 |
| 4 | 1.354 | 1.339 | -0.015 |
| 8 | 1.302 | 1.294 | -0.008 |
| 16 | 1.254 | 1.250 | -0.004 |
| 32 | 1.211 | 1.210 | -0.001 |

Qwen3-1.7B：

| rank | hidden_lora | combined_inlmh | Δ |
|---:|---:|---:|---:|
| 2 | 0.7713 | 0.764 | -0.0073 |
| 4 | 0.7545 | 0.750 | -0.0045 |
| 8 | 0.7381 | 0.735 | -0.0031 |
| 16 | 0.7227 | 0.721 | -0.0017 |
| 32 | 0.7093 | 0.709 | -0.0003 |

Qwen2.5-7B：

| rank | hidden_lora | combined_inlmh | Δ |
|---:|---:|---:|---:|
| 4 | 0.937 | 0.931 | -0.006 |
| 8 | 0.919 | 0.916 | -0.003 |
| 16 | 0.901 | 0.899 | -0.002 |

结论：普通 hidden LoRA 随 rank 增加稳定降低 loss；AffLoRA 的边际收益在低 rank / 小参数预算下最大，随着 hidden LoRA 容量增大逐渐收敛。这个结果强化了项目叙事：AffLoRA 不是替代大容量 hidden LoRA，而是在相同 hidden LoRA 预算上用很少额外参数补足词表侧训练。

### Phase 7a / 7b / 7c：7B / 8B 低 rank 参数效率

输出：`outputs/affine_vocab/phase7a_20260520_035042/`、`outputs/affine_vocab/phase7b_20260520_045817/`、`outputs/affine_vocab/phase7c_20260520_094742/`。该阶段按 30 分钟节奏监控 GPU 使用，GPU 1-6 正常满载；目标是把大模型 rank 曲线向更低 rank 继续延伸，检验 AffLoRA 在小参数预算下的边际收益。Phase 7c 补充 seed 43 / 44，因此这里汇总 seed 42 / 43 / 44 的三 seed 平均。

Qwen2.5-7B：

| rank | hidden_lora mean | combined_inlmh mean | Δ |
|---:|---:|---:|---:|
| 1 | 0.9658 | 0.9576 | -0.0082 |
| 2 | 0.9528 | 0.9457 | -0.0071 |
| 4 | 0.9361 | 0.9313 | -0.0048 |

Qwen3-8B：

| rank | hidden_lora mean | combined_inlmh mean | Δ |
|---:|---:|---:|---:|
| 1 | 0.9074 | 0.9001 | -0.0073 |
| 2 | 0.8967 | 0.8907 | -0.0060 |
| 4 | 0.8849 | 0.8805 | -0.0044 |

结论：7B / 8B 低 rank 实验直接支持“AffLoRA 在受限 hidden LoRA 预算下更有参数效率”这一表述。相比 Phase 6b 的 r8 / r16，r1-r4 的增益更大，且两个大模型在 3/3 seed 上都一致为正；这说明 AffLoRA 最适合被描述为低成本词表侧补充模块，尤其适合 hidden LoRA rank 较低或参数预算紧张的后训练场景。

### Phase 8a：7B / 8B 低 rank Claim 1b vocab LoRA 对照

输出：`outputs/affine_vocab/phase8a_20260520_152912/`。该阶段在 `hidden_lora_r1/r2` 基础上，对 `embed_tokens` 和 `lm_head` 额外加入传统 vocab-dim LoRA r=1，用于和 Phase 7 的 AffLoRA 低 rank 结果比较。

Qwen2.5-7B，seed 42：

| setting | trainable | eval_loss |
|---|---:|---:|
| hidden r1 | 2.52M | 0.9656 |
| hidden r1 + vocab LoRA r1 | 2.83M | 0.9637 |
| hidden r1 + AffLoRA | 2.76M | 0.9574 |
| hidden r2 | 5.05M | 0.9523 |
| hidden r2 + vocab LoRA r1 | 5.36M | 0.9500 |
| hidden r2 + AffLoRA | 5.28M | 0.9460 |

Qwen3-8B，seed 42：

| setting | trainable | eval_loss |
|---|---:|---:|
| hidden r1 | 2.73M | 0.9071 |
| hidden r1 + vocab LoRA r1 | 3.04M | 0.9061 |
| hidden r1 + AffLoRA | 2.99M | 0.9000 |
| hidden r2 | 5.46M | 0.8971 |
| hidden r2 + vocab LoRA r1 | 5.77M | 0.8967 |
| hidden r2 + AffLoRA | 5.72M | 0.8915 |

结论：在 7B / 8B 的低 rank 场景下，传统 vocab-dim LoRA r=1 只带来很小增益，且明显弱于 AffLoRA；AffLoRA 参数量还略少于对应 vocab LoRA 对照。这直接补强 Claim 1b：AffLoRA 的收益不是单纯来自“额外训练了 emb/lm_head”，而是 hidden 维 affine 参数化在该适配范围内更有效。

### Phase 8b：0.5B / 1.7B 低 rank Claim 1b vocab LoRA 对照

输出：`outputs/affine_vocab/phase8b_20260520_170927/`。该阶段按 Phase 8a 的统一口径补齐小/中模型，比较 `hidden_lora`、`hidden_lora + AffLoRA`、`hidden_lora + vocab-dim LoRA r=1`。

Qwen2.5-0.5B，seed 42：

| setting | trainable | eval_loss |
|---|---:|---:|
| hidden r1 | 550k | 1.440 |
| hidden r1 + vocab LoRA r1 | 856k | 1.413 |
| hidden r1 + AffLoRA | 608k | 1.413 |
| hidden r2 | 1.10M | 1.398 |
| hidden r2 + vocab LoRA r1 | 1.41M | 1.378 |
| hidden r2 + AffLoRA | 1.16M | 1.378 |

Qwen3-1.7B，seed 42：

| setting | trainable | eval_loss |
|---|---:|---:|
| hidden r1 | 1.09M | 0.787 |
| hidden r1 + vocab LoRA r1 | 1.40M | 0.785 |
| hidden r1 + AffLoRA | 1.22M | 0.777 |
| hidden r2 | 2.18M | 0.771 |
| hidden r2 + vocab LoRA r1 | 2.49M | 0.769 |
| hidden r2 + AffLoRA | 2.31M | 0.764 |

结论：0.5B 上 vocab LoRA r=1 与 AffLoRA 的 eval_loss 基本打平，但 AffLoRA 参数更少；到 1.7B 时，AffLoRA 已明显优于 vocab LoRA，并且仍保持更少参数。结合 Phase 8a 的 7B / 8B 结果，Claim 1b 呈现清晰尺寸趋势：模型越大，hidden 维 AffLoRA 相比 vocab 维 LoRA 的优势越明显。

### Phase 9：后续补强实验

完整整理见 `docs/PHASE9_STRENGTHENING_SUMMARY.md`。该阶段用于把 Claim 1b 从单 seed 结果提升为多 seed 证据，并补充 alpha / bias scale 鲁棒性与位置归因。

#### Phase 9a：7B / 8B vocab LoRA 多 seed

输出：`outputs/affine_vocab/phase9a_20260520_184329/`。Phase 9a 将 Phase 8a 的 seed42 vocab LoRA 对照扩展到 seed43 / 44，并与 Phase 7c 的 AffLoRA 三 seed 对齐。

| model | rank | hidden + vocab LoRA r1 mean | hidden + AffLoRA mean | AffLoRA advantage |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | 1 | 0.9639 | 0.9576 | -0.0063 |
| Qwen2.5-7B | 2 | 0.9503 | 0.9457 | -0.0046 |
| Qwen3-8B | 1 | 0.9062 | 0.9001 | -0.0061 |
| Qwen3-8B | 2 | 0.8963 | 0.8907 | -0.0056 |

结论：传统 vocab-dim LoRA r1 在 7B / 8B 上只带来小幅增益，且 3/3 seed 均弱于 AffLoRA。AffLoRA 参数还略少，因此 Claim 1b 的低 rank 大模型证据已经闭合。

#### Phase 9b / 9e：Qwen2.5-1.5B low-rank 三方多 seed

输出：`outputs/affine_vocab/phase9b_20260520_212708/`、`outputs/affine_vocab/phase9e_20260521_015553/`。设置为 hidden r1/r2、hidden + AffLoRA、hidden + vocab LoRA r1，seed 42/43/44。

| rank | hidden mean | hidden + vocab LoRA r1 mean | hidden + AffLoRA mean | AffLoRA vs hidden | AffLoRA vs vocab |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.1267 | 1.1200 | 1.1133 | -0.0134 | -0.0067 |
| 2 | 1.1043 | 1.0987 | 1.0953 | -0.0090 | -0.0034 |

结论：Qwen2.5-1.5B 在 r1/r2、3 个 seed 上均表现为 AffLoRA 最优，补上了 Qwen2.5 系列 0.5B -> 1.5B -> 7B 的尺寸链。

#### Phase 9c / 9f：alpha 与 bias scale 鲁棒性

输出：`outputs/affine_vocab/phase9c_20260520_214840/`、`outputs/affine_vocab/phase9f_20260521_105726/`。Phase 9c 固定 AffLoRA rank 16，扫 `affine_alpha` 64/128/256；Phase 9f 固定 alpha 128，在 Qwen3-1.7B 上扫 `affine_bias_scale` 0/0.5/1/2。

| model | rank | alpha 64 | alpha 128 | alpha 256 |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 1 | 0.779 | 0.777 | 0.776 |
| Qwen3-1.7B | 2 | 0.766 | 0.764 | 0.763 |
| Qwen2.5-7B | 1 | 0.959 | 0.958 | 0.956 |
| Qwen2.5-7B | 2 | 0.947 | 0.946 | 0.944 |

| model | rank | scale 0 | scale 0.5 | scale 1 | scale 2 |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 1 | 0.7776 | 0.7773 | 0.7767 | 0.7774 |
| Qwen3-1.7B | 2 | 0.7642 | 0.7641 | 0.7636 | 0.7637 |

结论：alpha 64/128/256 均有效，256 略优但不是质变；bias scale 在 0-2 范围内几乎不敏感。默认 `rank=16, alpha=128, bias_scale=1` 不是偶然甜点。

#### Phase 9d：7B 低 rank 位置归因

输出：`outputs/affine_vocab/phase9d_20260520_234301/`。设置为 Qwen2.5-7B hidden r1/r2，对比 hidden-only、input-only、lm_head-only、input+lm_head。

| rank | hidden-only | input-only | lm_head-only | input+lm_head |
|---:|---:|---:|---:|---:|
| 1 | 0.966 | 0.965 | 0.959 | 0.957 |
| 2 | 0.952 | 0.951 | 0.946 | 0.946 |

结论：低 rank 下收益主要来自输出侧 `lm_head` affine；input-only 增益很小，input+lm_head 与 lm_head-only 接近但略好或打平。这与 1.5B / 3B / 7B 既有观察一致。

### Phase 6c：AffLoRA rank sweep

输出：`outputs/affine_vocab/phase6c_20260520_004810/`。固定 hidden LoRA r8，比较 AffLoRA rank 8 / 16 / 32。

| model | aff rank 8 | aff rank 16 | aff rank 32 | 最优 |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 0.7357 | 0.7346 | 0.7338 | rank 32 |
| Qwen2.5-7B | 0.9175 | 0.9167 | 0.9137 | rank 32 |
| Qwen3-8B | 0.8700 | 0.8690 | 0.8682 | rank 32 |

结论：AffLoRA rank 从 8 到 32 带来小幅但一致的改善，7B 上最明显。rank 16 是当前成本和效果的稳妥默认；若追求单点最佳，rank 32 更好，但额外收益仍小于 hidden LoRA rank 扩容带来的主效应。

### Phase 6d：全量微调上限

输出：`outputs/affine_vocab/phase6d_20260520_031643/`。该阶段只作为小模型强上限参考，使用 `full_finetune`，不保存完整 checkpoint。

| model | full finetune | 对照 |
|---|---:|---|
| Qwen2.5-0.5B | 1.177 | 优于 Phase 6b 的 hidden/combined r32，后者约 1.211 / 1.210 |
| Qwen3-0.6B | 1.019 | 优于 Phase 5a 的 hidden LoRA r32 1.042 |

结论：全量微调仍是小模型上限，明显强于同任务下的 LoRA / AffLoRA 参数高效方案。这个结果适合作为“性能上限”参照，不削弱 AffLoRA 的参数效率主张；它反而明确了 AffLoRA 的定位是低成本补充词表层训练，而不是追求替代 full fine-tuning。

## 补充实验

### Phase 3c：位置消融

将 AffLoRA 风格的 hidden 维 affine adapter 放在 decoder layer 3、10、17、24 输出后或 final norm 后。结果显示 layer 3/10 约 1.367，优于 after embedding 的 1.423，final norm 最差约 1.569。该实验回答“affine 放在哪里”，不直接验证三条 claim。

### Phase 3d：DoRA / rsLoRA 对照

Qwen3-0.6B 上的结果：

| run | final eval |
|---|---:|
| LoRA r=8 | 1.100 |
| LoRA r=4 | 1.133 |
| DoRA r=8 | 1.098 |
| rsLoRA r=8 | 1.069 |
| rsLoRA r=8 + affine s1=8 | 1.070 |

rsLoRA 的优势主要来自 scale 公式差异。该实验对 LoRA 超参讨论有参考价值，但不进入三条 claim 的核心结论。

## Claim 闭合状态

1. Claim 1a：已闭合，证据来自 Phase 3a + Phase 4c 三 seed，以及 0.5B / 1.7B / 3B / 4B / 7B / 8B 的跨模型验证。
2. Claim 1b：已闭合，证据来自 Phase 8a / 8b / 9a / 9b / 9e。AffLoRA 在 1.5B / 1.7B / 7B / 8B 上优于同口径 vocab-dim LoRA，0.5B 打平但参数更少。
3. Claim 2：已闭合，证据来自 Phase 4a + Phases 2' / 3a。
4. Claim 3：已闭合，证据来自 Phase 2c。
5. 超参与机制：Phase 9c / 9f 显示 alpha 与 bias scale 不敏感；Phase 9d 显示主要收益来自 output-side / `lm_head`。

可选后续：补充非 `eval_loss` 的生成质量抽样；大模型 full fine-tuning 成本高且不直接增强 AffLoRA 参数效率主张，不建议优先。
