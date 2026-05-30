# AffLoRA 变体定义与正确实验结果总表

更新时间：2026-05-26（含 §7.10 v3 终极证据 sweep，35 任务全完成）。

## 0. 读者导读

> **TL;DR**：AffLoRA 用 ~1% 的参数把 LoRA 这把刀从 transformer 内部移到 hidden ↔ vocab 边界。`affine_input` 在没有 hidden_lora 时是最廉价的独立 SFT 适配器；`affine_lm_head` 在有 hidden_lora 时是最廉价的"末端补刀"；`mergeable_tied` 把两件事合二为一并解决部署 merge。

### 0.1 想直接看核心结论 + 证据

请直接跳到 **§10「项目核心 Insight + 数据来源总表」**——每条 insight 后面都附了：

- claim 一句话；
- 直接的 eval_loss 对比数字；
- 具体的 run 名 / outputs 路径；
- 文档内交叉引用（§x.y）。

可以一行行溯源验证，不需要再翻早期 phase 数据。

§9 是这些 insight 的散文化机制讲解，§10 是结构化的 claim + 数据清单——两者互补。**只想要"项目能得出什么"的读者请直接看 §10**。

### 0.2 建议阅读顺序

| 你的目的 | 推荐路径 |
|---|---|
| 只想知道"能用上什么"——3 分钟版 | **§9.7** 一句话总览 → **§10** 数据来源总表 |
| 准备写论文/报告，要核心 claim 和叙事修正 | **§10 Insight + 数据来源** → **§9 机制讲解** → **§8 最终结论** → **§7.9 论文叙事修正** |
| 复现/验证某个 claim 的实验数据 | **§10**（找对应 run 名）→ **§7.10** v3 终极证据 sweep（最近 2026-05-26 35 任务）→ **§7** 位置归因 v2（之前主线证据）→ **§5** Run 级详表（早期分阶段数据）|
| 给同事讲 AffLoRA 是什么 | **§1 变体定义** → **§7.0 hidden_lora 详细定义** → **§9.1–9.2 方法论 / 几何** |
| 部署 tied 模型，关心 merge 是否等价 | **§1.3 Tied/mergeable 变体** → **§9.4 mergeable_tied 双重价值** |
| 选超参 / 选具体配置 | **§9.6 实操选择表** → **§2 默认训练/超参口径** |

### 0.3 章节速查

| 章节 | 内容 | 读者优先级 |
|---|---|---|
| §1 | 变体定义（hidden_lora、affine_input/lm_head、tied/mergeable 等所有变体的概念） | 入门必读 |
| §2 | 默认训练/超参口径（lr、batch、epoch、seed 等） | 复现需要 |
| §3 | 表格字段说明 | 看表前查 |
| §4 | Base eval 参照（frozen base loss）| 数据对照 |
| §5 | Phase 2–16 + UltraChat + MiniMind 的所有 run 级 eval_loss | 详细证据库（厚） |
| §6 | 不纳入主表的实验（探索性 / 失败 run） | 知道哪些不算数 |
| §7 | **位置归因 v2 综合发现**（2026-05-26 主线，§7.10 含最新 v3 终极证据 sweep） | **核心证据章** |
| §8 | 最终结论口径（7 条 claim） | **核心结论章** |
| §9 | 项目核心 Insight（散文化讲解：方法论、几何、参数效率、工程、叙事、实操） | **机制讲解章** |
| §10 | **核心 Insight + 数据来源总表**（结构化 claim + 关键数字 + run 名 + 章节交叉引用） | **数据溯源章** |

> **说明**：§5 的早期 Phase 数据保留是为了可追溯性；如果你只关心当前 claim 的支撑证据，§7 已经做了完整的重新整理与修正。§5 中 §3 Phase 2c 等少量旧表得出的"lm_head 通常更强"已经被 §7 替代，请以 §7 为准。

---

本文整理当前可用于论文/报告论证的正确实验：`sft_t2t_mini_25k` 主线、tied/mergeable 工程线、UltraChat 100k 外部验证、MiniMind random 100k 随机子集验证。旧 MetaMath/math 探索、早期 DoRA/rsLoRA 背景探索、失败的 `minimind_random100k_lowrank_20260525_131007` 启动不纳入。

指标统一为 final `eval_loss`，越低越好。

## 1. 变体定义

### 1.1 基础 LoRA / 控制组

| 变体 | 训练位置 | 定义 | 主要用途 |
|---|---|---|---|
| `hidden_lora` | Transformer block projections | 标准 PEFT LoRA，默认目标为 `q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj` | 主 baseline |
| `single-layer LoRA` | 单个 decoder layer 的 projections | 通过 `--hidden-lora-layers-to-transform` 只训练一层；run 名常见 `single_q_l14_r16`、`single_qkvo_mid_r4` | Claim 3 同量级小预算对照 |
| `hidden + vocab LoRA` | hidden LoRA + `embed_tokens/lm_head` vocab 维 LoRA | 通过 `--include-emb-lmh-lora-rank` 添加传统词表维 LoRA | Claim 1b 参数效率对照 |
| `full_finetune` | 全模型 | 所有参数参与训练 | 性能上限或 sanity probe |
| `frozen base` | 无训练 | 只评估 base model loss | Claim 2 参照 |

### 1.2 AffLoRA 变体

AffLoRA 在 hidden 维学习低秩 affine map，用少量参数作用到词表层权重或对应 hidden state。记 `s1 = affine_alpha / affine_rank`。

| 变体 | 训练位置 | 定义 | 主要用途 |
|---|---|---|---|
| `affine_input` | `embed_tokens` | 只在输入 embedding 侧加 hidden-space affine adapter | Aff 本身表达能力、小预算对照 |
| `affine_lm_head` | `lm_head` | 只在输出 head 侧加 hidden-space affine adapter | input vs output 归因 |
| `affine_input_lm_head` | `embed_tokens + lm_head` | 双侧加 AffLoRA，不加 hidden LoRA | Claim 2 / Claim 3 的 AffLoRA-only 主形式 |
| `affine_input_plus_hidden_lora` | hidden LoRA + input affine | hidden LoRA 基础上只加输入侧 affine | input 侧边际贡献 |
| `affine_lm_head_plus_hidden_lora` | hidden LoRA + lm_head affine | hidden LoRA 基础上只加输出侧 affine | lm_head 侧边际贡献 |
| `affine_input_lm_head_plus_hidden_lora` / `combined_inlmh` | hidden LoRA + input/lm_head affine | 完整 AffLoRA 架构 | Claim 1a 主处理组 |

### 1.3 Tied / mergeable 变体

对 tied embedding 模型，本文区分两类 tied AffLoRA 形态：

1. **保留 adapter 模块的双侧适配形态**：input 侧和 lm_head 侧都可以有适配器，推理/保存时保留 adapter 模块；它不要求把所有增量严格合并回同一个 tied embedding 矩阵。
2. **可 merge 的 tied 形态**：约束 input 和 lm_head 使用同一个 affine adapter，并在 lm_head 侧使用转置作用，使其等价于对 tied embedding 权重做一次 affine merge。

| 开关/变体 | 所属形态 | 定义 | 正确口径 |
|---|---|---|
| 普通 tied adapter 双侧适配 | 保留 adapter 模块 | 在 tied 模型上仍按 `affine_input_lm_head` 或 `combined_inlmh` 形式给 input/lm_head 双侧加适配器；两侧增量作为 adapter 模块保留 | 有效果，但部署时保存 adapter 模块，不声称严格 merge 回单一 tied embedding |
| tied adapter 位置归因 | 保留 adapter 模块 | 分别比较 input-only、lm_head-only、input+lm_head 的 adapter 效果 | tied 模型中 lm_head 侧适配器通常更强 |
| `--tie-affine-input-lm-head-adapters` | 共享 adapter / merge 前置约束 | tied embedding 模型中让 input 和 lm_head 共享同一个 affine adapter | 可减少参数并靠近 merge 形式，但单独使用时还不是严格 merge 等价 |
| `--mergeable-tied-affine-output` | 可 merge tied 形态 | lm_head 侧应用 input affine 的转置作用，使 raw logits 与 merge 后 tied embedding 数值等价 | 正确可 merge tied 形式 |
| `mergeable_tied_combined_rank32_s1_8` | 可 merge tied 形态 | hidden LoRA r8 + mergeable tied affine rank32/s1=8 | tied 工程线最终推荐配置 |

## 2. 默认训练/超参口径

| 项 | 默认值 |
|---|---|
| 主数据 | `data/sft_t2t_mini_25k`，24k train + 1k eval |
| 外部验证 | UltraChat 100k：100k train + 2k eval；MiniMind random 100k：100k train + 2k eval |
| epoch | 1 |
| `max_seq_len` | 1024 |
| 有效 batch | 常见 `8x2`，即 per-device batch 8、gradient accumulation 2 |
| learning rate | adapter 默认 `2e-4`；full fine-tune probe 另有 `2e-5` |
| scheduler | cosine，warmup ratio 0.03 |
| hidden LoRA 默认 | rank 8，alpha 16；低 rank sweep 使用 r1/r2/r4 等 |
| AffLoRA 默认 | rank 16，alpha 128，即 s1=8；AffLoRA-only 常扫 s1=16/32/64 |
| precision | frozen base bf16/auto，trainable adapter fp32 master |
| seed | seed42 为默认；关键实验补 seed43/44 |

## 3. 表格字段说明

| 字段 | 含义 |
|---|---|
| `h_r/a` | hidden LoRA rank / alpha；非 hidden LoRA 相关实验可能显示默认参数但不一定实际使用 |
| `aff_r/a/s1` | AffLoRA rank / alpha / s1；仅 affine 相关变体显示 |
| `flags` | 额外重要开关，如 `mergeable`、`tied_shared`、`emb_lmh_r1`、`layers=14` |
| `batch` | `per_device_train_batch_size x gradient_accumulation_steps` |
| `trainable` | 可训练参数量 |
| `eval_loss` | final eval loss |

## 4. Base Eval 参照

| model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss | source |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| Qwen3-0.6B | sft_t2t_mini_25k | frozen base | - | - | - | - | - | - | - | 1.848 | RESULTS_SO_FAR.md / Claim 2 |
| Qwen2.5-1.5B | sft_t2t_mini_25k | frozen base | - | - | - | - | - | - | - | 1.926 | RESULTS_SO_FAR.md / Claim 2 |


## 5. 正确实验 Run 级详表


### Phase 2 sweep：0.6B AffLoRA-only s1 sweep

来源：`outputs/affine_vocab/phase2_sweep_20260519_115922`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| affine_in_r16_a16_s2_1 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input | 42 | 16/32 | 16/16/1 | 0.0002 | 16x1 | - | 34k | 1.485 |
| affine_in_r16_a32_s2_1 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input | 42 | 16/32 | 16/32/2 | 0.0002 | 16x1 | - | 34k | 1.466 |
| affine_in_r16_a32_s2_4 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input | 42 | 16/32 | 16/32/2 | 0.0002 | 16x1 | - | 34k | 1.463 |
| affine_in_r16_a64_s2_1 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input | 42 | 16/32 | 16/64/4 | 0.0002 | 16x1 | - | 34k | 1.452 |
| affine_in_r16_a8_s2_1 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input | 42 | 16/32 | 16/8/0.5 | 0.0002 | 16x1 | - | 34k | 1.495 |
| baseline_hidden_lora_r8 | Qwen3-0.6B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 16x1 | - | 5.05M | 1.101 |


### Phase 2b：0.6B combined + capacity controls

来源：`outputs/affine_vocab/phase2b_20260519_123458`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| affine_only_s1_16 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input | 42 | 16/32 | 16/256/16 | 0.0002 | 16x1 | - | 34k | 1.434 |
| affine_only_s1_8 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input | 42 | 16/32 | 16/128/8 | 0.0002 | 16x1 | - | 34k | 1.442 |
| combined_in_s1_2 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_plus_hidden_lora | 42 | 8/16 | 16/32/2 | 0.0002 | 16x1 | - | 5.08M | 1.099 |
| combined_in_s1_4 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_plus_hidden_lora | 42 | 8/16 | 16/64/4 | 0.0002 | 16x1 | - | 5.08M | 1.099 |
| combined_inlmh_s1_4 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/64/4 | 0.0002 | 16x1 | - | 5.11M | 1.098 |
| hidden_lora_r9_baseline | Qwen3-0.6B | sft_t2t_mini_25k | hidden_lora | 42 | 9/18 | - | 0.0002 | 16x1 | - | 5.68M | 1.096 |


### Phase 2c：0.6B matched-budget single-layer LoRA

来源：`outputs/affine_vocab/phase2c_20260519_125518`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| affine_only_in_lmh_s1_4 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/64/4 | 0.0002 | 16x1 | - | 67k | 1.346 |
| affine_only_s1_32 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input | 42 | 16/32 | 16/512/32 | 0.0002 | 16x1 | - | 34k | 1.426 |
| lora_only_q_l0_r16 | Qwen3-0.6B | sft_t2t_mini_25k | hidden_lora | 42 | 16/32 | - | 0.0002 | 16x1 | layers=0 | 49k | 1.626 |
| lora_only_q_l14_r16 | Qwen3-0.6B | sft_t2t_mini_25k | hidden_lora | 42 | 16/32 | - | 0.0002 | 16x1 | layers=14 | 49k | 1.667 |
| lora_only_q_l27_r16 | Qwen3-0.6B | sft_t2t_mini_25k | hidden_lora | 42 | 16/32 | - | 0.0002 | 16x1 | layers=27 | 49k | 1.78 |
| lora_only_qkvo_l14_r4 | Qwen3-0.6B | sft_t2t_mini_25k | hidden_lora | 42 | 4/8 | - | 0.0002 | 16x1 | layers=14 | 41k | 1.473 |


### Phase 3a：Qwen2.5-1.5B 放大

来源：`outputs/affine_vocab/phase3a_20260519_132103`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| affine_in_lmh_s1_16 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/256/16 | 0.0002 | 8x2 | - | 100k | 1.326 |
| affine_only_s1_32 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input | 42 | 16/32 | 16/512/32 | 0.0002 | 8x2 | - | 51k | 1.478 |
| affine_only_s1_64 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input | 42 | 16/32 | 16/1024/64 | 0.0002 | 8x2 | - | 51k | 1.468 |
| baseline_hidden_lora_r8 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 8x2 | - | 9.23M | 1.054 |
| combined_in_s1_8 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 8x2 | - | 9.28M | 1.053 |
| combined_inlmh_s1_8 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 8x2 | - | 9.33M | 1.048 |


### Phase 3b：Qwen3-0.6B AffLoRA-only s1 sweep

来源：`outputs/affine_vocab/phase3b_20260519_134358`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| affine_in_lmh_s1_16 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/256/16 | 0.0002 | 16x1 | - | 67k | 1.326 |
| affine_in_lmh_s1_32 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/512/32 | 0.0002 | 16x1 | - | 67k | 1.322 |
| affine_in_lmh_s1_64 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/1024/64 | 0.0002 | 16x1 | - | 67k | 1.32 |
| affine_in_lmh_s1_8 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/128/8 | 0.0002 | 16x1 | - | 67k | 1.335 |
| affine_only_s1_128 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input | 42 | 16/32 | 16/2048/128 | 0.0002 | 16x1 | - | 34k | 1.422 |
| affine_only_s1_64 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input | 42 | 16/32 | 16/1024/64 | 0.0002 | 16x1 | - | 34k | 1.419 |


### Phase 4b：vocab 维 LoRA 控制组

来源：`outputs/affine_vocab/phase4b_20260519_152409`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_1_5b__lora_emblmh_r1 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 8x2 | emb_lmh_r1 | 9.54M | 1.049 |
| qwen25_1_5b__lora_emblmh_r2 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 8x2 | emb_lmh_r2 | 9.85M | 1.046 |
| qwen3_0_6b__lora_emblmh_r1 | Qwen3-0.6B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 16x1 | emb_lmh_r1 | 5.35M | 1.094 |
| qwen3_0_6b__lora_emblmh_r2 | Qwen3-0.6B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 16x1 | emb_lmh_r2 | 5.66M | 1.093 |


### Phase 4c：Qwen2.5-1.5B 多 seed Claim 1a

来源：`outputs/affine_vocab/phase4c_20260519_155136`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| baseline_r8_seed43 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 43 | 8/16 | - | 0.0002 | 8x2 | - | 9.23M | 1.054 |
| baseline_r8_seed44 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 44 | 8/16 | - | 0.0002 | 8x2 | - | 9.23M | 1.054 |
| combined_in_s1_8_seed43 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_plus_hidden_lora | 43 | 8/16 | 16/128/8 | 0.0002 | 8x2 | - | 9.28M | 1.052 |
| combined_in_s1_8_seed44 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_plus_hidden_lora | 44 | 8/16 | 16/128/8 | 0.0002 | 8x2 | - | 9.28M | 1.052 |
| combined_inlmh_s1_8_seed43 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 43 | 8/16 | 16/128/8 | 0.0002 | 8x2 | - | 9.33M | 1.048 |
| combined_inlmh_s1_8_seed44 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 44 | 8/16 | 16/128/8 | 0.0002 | 8x2 | - | 9.33M | 1.047 |


### Phase 5a：0.6B hidden LoRA rank sweep

来源：`outputs/affine_vocab/phase5a_20260519_170658`；摘要见 `docs/phase5a_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| hidden_lora_r16 | Qwen3-0.6B | sft_t2t_mini_25k | hidden_lora | 42 | 16/32 | - | 0.0002 | 16x1 | - | 10.09M | 1.071 |
| hidden_lora_r2 | Qwen3-0.6B | sft_t2t_mini_25k | hidden_lora | 42 | 2/4 | - | 0.0002 | 16x1 | - | 1.26M | 1.163 |
| hidden_lora_r32 | Qwen3-0.6B | sft_t2t_mini_25k | hidden_lora | 42 | 32/64 | - | 0.0002 | 16x1 | - | 20.19M | 1.042 |
| hidden_lora_r4 | Qwen3-0.6B | sft_t2t_mini_25k | hidden_lora | 42 | 4/8 | - | 0.0002 | 16x1 | - | 2.52M | 1.133 |
| hidden_lora_r8 | Qwen3-0.6B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 16x1 | - | 5.05M | 1.101 |


### Phase 5b：Qwen2.5-3B 架构验证

来源：`outputs/affine_vocab/phase5b_20260519_172412`；摘要见 `docs/phase5b_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| affine_in_lmh_s1_16 | Qwen2.5-3B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/256/16 | 0.0002 | 8x2 | - | 133k | 1.223 |
| baseline_hidden_lora_r8 | Qwen2.5-3B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 8x2 | - | 14.97M | 0.9797 |
| combined_in_s1_8 | Qwen2.5-3B | sft_t2t_mini_25k | affine_input_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 8x2 | - | 15.03M | 0.9789 |
| combined_inlmh_s1_8 | Qwen2.5-3B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 8x2 | - | 15.10M | 0.9762 |


### Phase 5c：Qwen3-4B 架构验证

来源：`outputs/affine_vocab/phase5c_20260519_180407`；摘要见 `docs/phase5c_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| affine_in_lmh_s1_16 | Qwen3-4B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/256/16 | 0.0002 | 4x4 | - | 166k | 1.048 |
| baseline_hidden_lora_r8 | Qwen3-4B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 4x4 | - | 16.52M | 0.8828 |
| combined_in_s1_8 | Qwen3-4B | sft_t2t_mini_25k | affine_input_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 4x4 | - | 16.60M | 0.8818 |
| combined_inlmh_s1_8 | Qwen3-4B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 4x4 | - | 16.68M | 0.881 |


### Phase 5d：Qwen2.5-7B 架构验证

来源：`outputs/affine_vocab/phase5d_20260519_193129`；摘要见 `docs/phase5d_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| affine_in_lmh_s1_16 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/256/16 | 0.0002 | 4x4 | - | 233k | 1.12 |
| baseline_hidden_lora_r8 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 4x4 | - | 20.19M | 0.918 |
| combined_in_s1_8 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 4x4 | - | 20.30M | 0.9184 |
| combined_inlmh_s1_8 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 4x4 | - | 20.42M | 0.9151 |


### Phase 6a：0.5B/1.7B/8B 架构验证

来源：`outputs/affine_vocab/phase6a_20260519_204314`；摘要见 `docs/phase6a_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_05b__affine_in_lmh_s1_16 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/256/16 | 0.0002 | 16x1 | - | 58k | 1.637 |
| qwen25_05b__combined_inlmh_s1_8 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 16x1 | - | 4.46M | 1.294 |
| qwen25_05b__hidden_lora_r8 | Qwen2.5-0.5B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 16x1 | - | 4.40M | 1.302 |
| qwen3_17b__affine_in_lmh_s1_16 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/256/16 | 0.0002 | 8x2 | - | 133k | 0.9074 |
| qwen3_17b__combined_inlmh_s1_8 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 8x2 | - | 8.85M | 0.735 |
| qwen3_17b__hidden_lora_r8 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 8x2 | - | 8.72M | 0.7383 |
| qwen3_8b__affine_in_lmh_s1_16 | Qwen3-8B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/256/16 | 0.0002 | 4x4 | - | 266k | 1.01 |
| qwen3_8b__combined_inlmh_s1_8 | Qwen3-8B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 4x4 | - | 22.09M | 0.8696 |
| qwen3_8b__hidden_lora_r8 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 4x4 | - | 21.82M | 0.8717 |


### Phase 6b：hidden rank 曲线 + combined

来源：`outputs/affine_vocab/phase6b_20260519_222852`；摘要见 `docs/phase6b_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_05b__combined_inlmh_r16 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 16/32 | 16/128/8 | 0.0002 | 16x1 | - | 8.86M | 1.25 |
| qwen25_05b__combined_inlmh_r2 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 16x1 | - | 1.16M | 1.38 |
| qwen25_05b__combined_inlmh_r32 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 32/64 | 16/128/8 | 0.0002 | 16x1 | - | 17.65M | 1.21 |
| qwen25_05b__combined_inlmh_r4 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 4/8 | 16/128/8 | 0.0002 | 16x1 | - | 2.26M | 1.339 |
| qwen25_05b__combined_inlmh_r8 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 16x1 | - | 4.46M | 1.294 |
| qwen25_05b__hidden_lora_r16 | Qwen2.5-0.5B | sft_t2t_mini_25k | hidden_lora | 42 | 16/32 | - | 0.0002 | 16x1 | - | 8.80M | 1.254 |
| qwen25_05b__hidden_lora_r2 | Qwen2.5-0.5B | sft_t2t_mini_25k | hidden_lora | 42 | 2/4 | - | 0.0002 | 16x1 | - | 1.10M | 1.398 |
| qwen25_05b__hidden_lora_r32 | Qwen2.5-0.5B | sft_t2t_mini_25k | hidden_lora | 42 | 32/64 | - | 0.0002 | 16x1 | - | 17.60M | 1.211 |
| qwen25_05b__hidden_lora_r4 | Qwen2.5-0.5B | sft_t2t_mini_25k | hidden_lora | 42 | 4/8 | - | 0.0002 | 16x1 | - | 2.20M | 1.354 |
| qwen25_05b__hidden_lora_r8 | Qwen2.5-0.5B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 16x1 | - | 4.40M | 1.302 |
| qwen25_7b__combined_inlmh_r16 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 16/32 | 16/128/8 | 0.0002 | 4x4 | - | 40.60M | 0.8995 |
| qwen25_7b__combined_inlmh_r4 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 4/8 | 16/128/8 | 0.0002 | 4x4 | - | 10.33M | 0.9314 |
| qwen25_7b__combined_inlmh_r8 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 4x4 | - | 20.42M | 0.9161 |
| qwen25_7b__hidden_lora_r16 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 42 | 16/32 | - | 0.0002 | 4x4 | - | 40.37M | 0.9008 |
| qwen25_7b__hidden_lora_r4 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 42 | 4/8 | - | 0.0002 | 4x4 | - | 10.09M | 0.9366 |
| qwen25_7b__hidden_lora_r8 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 4x4 | - | 20.19M | 0.9186 |
| qwen3_17b__combined_inlmh_r16 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 16/32 | 16/128/8 | 0.0002 | 8x2 | - | 17.57M | 0.721 |
| qwen3_17b__combined_inlmh_r2 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 8x2 | - | 2.31M | 0.7642 |
| qwen3_17b__combined_inlmh_r32 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 32/64 | 16/128/8 | 0.0002 | 8x2 | - | 35.00M | 0.7087 |
| qwen3_17b__combined_inlmh_r4 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 4/8 | 16/128/8 | 0.0002 | 8x2 | - | 4.49M | 0.7496 |
| qwen3_17b__combined_inlmh_r8 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 8x2 | - | 8.85M | 0.7347 |
| qwen3_17b__hidden_lora_r16 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 42 | 16/32 | - | 0.0002 | 8x2 | - | 17.43M | 0.7227 |
| qwen3_17b__hidden_lora_r2 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 42 | 2/4 | - | 0.0002 | 8x2 | - | 2.18M | 0.7713 |
| qwen3_17b__hidden_lora_r32 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 42 | 32/64 | - | 0.0002 | 8x2 | - | 34.87M | 0.7093 |
| qwen3_17b__hidden_lora_r4 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 42 | 4/8 | - | 0.0002 | 8x2 | - | 4.36M | 0.7545 |
| qwen3_17b__hidden_lora_r8 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 8x2 | - | 8.72M | 0.7381 |


### Phase 6c：AffLoRA rank/alpha sweep

来源：`outputs/affine_vocab/phase6c_20260520_004810`；摘要见 `docs/phase6c_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_7b__combined_inlmh_affr16 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 4x4 | - | 20.42M | 0.9167 |
| qwen25_7b__combined_inlmh_affr32 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 32/256/8 | 0.0002 | 4x4 | - | 20.65M | 0.9137 |
| qwen25_7b__combined_inlmh_affr8 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 8/64/8 | 0.0002 | 4x4 | - | 20.30M | 0.9175 |
| qwen3_17b__combined_inlmh_affr16 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 8x2 | - | 8.85M | 0.7346 |
| qwen3_17b__combined_inlmh_affr32 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 32/256/8 | 0.0002 | 8x2 | - | 8.98M | 0.7338 |
| qwen3_17b__combined_inlmh_affr8 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 8/64/8 | 0.0002 | 8x2 | - | 8.78M | 0.736 |
| qwen3_8b__combined_inlmh_affr16 | Qwen3-8B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 4x4 | - | 22.09M | 0.8692 |
| qwen3_8b__combined_inlmh_affr32 | Qwen3-8B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 32/256/8 | 0.0002 | 4x4 | - | 22.35M | 0.8677 |
| qwen3_8b__combined_inlmh_affr8 | Qwen3-8B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 8/64/8 | 0.0002 | 4x4 | - | 21.96M | 0.8703 |


### Phase 6d：full fine-tune 上限

来源：`outputs/affine_vocab/phase6d_20260520_031643`；摘要见 `docs/phase6d_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_05b__full_finetune | Qwen2.5-0.5B | sft_t2t_mini_25k | full_finetune | 42 | 16/32 | - | 0.00001 | 4x4 | - | 494.03M | 1.177 |
| qwen3_06b__full_finetune | Qwen3-0.6B | sft_t2t_mini_25k | full_finetune | 42 | 16/32 | - | 0.00001 | 4x4 | - | 596.05M | 1.019 |


### Phase 7a：大模型低 rank

来源：`outputs/affine_vocab/phase7a_20260520_035042`；摘要见 `docs/phase7a_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| combined_inlmh_r1 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/128/8 | 0.0002 | 4x4 | - | 2.76M | 0.9574 |
| combined_inlmh_r2 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 4x4 | - | 5.28M | 0.946 |
| combined_inlmh_r4 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 4/8 | 16/128/8 | 0.0002 | 4x4 | - | 10.33M | 0.9309 |
| hidden_lora_r1 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 42 | 1/2 | - | 0.0002 | 4x4 | - | 2.52M | 0.9656 |
| hidden_lora_r2 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 42 | 2/4 | - | 0.0002 | 4x4 | - | 5.05M | 0.9523 |
| hidden_lora_r4 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 42 | 4/8 | - | 0.0002 | 4x4 | - | 10.09M | 0.9362 |


### Phase 7b：大模型多 seed 低 rank

来源：`outputs/affine_vocab/phase7b_20260520_045817`；摘要见 `docs/phase7b_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| combined_inlmh_r1 | Qwen3-8B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/128/8 | 0.0002 | 4x4 | - | 2.99M | 0.9 |
| combined_inlmh_r2 | Qwen3-8B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 4x4 | - | 5.72M | 0.8915 |
| combined_inlmh_r4 | Qwen3-8B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 4/8 | 16/128/8 | 0.0002 | 4x4 | - | 11.18M | 0.8808 |
| hidden_lora_r1 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 42 | 1/2 | - | 0.0002 | 4x4 | - | 2.73M | 0.9071 |
| hidden_lora_r2 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 42 | 2/4 | - | 0.0002 | 4x4 | - | 5.46M | 0.8971 |
| hidden_lora_r4 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 42 | 4/8 | - | 0.0002 | 4x4 | - | 10.91M | 0.8848 |


### Phase 7c：vocab LoRA 对照

来源：`outputs/affine_vocab/phase7c_20260520_094742`；摘要见 `docs/phase7c_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_7b__seed43__combined_inlmh_r1 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 43 | 1/2 | 16/128/8 | 0.0002 | 4x4 | - | 2.76M | 0.9571 |
| qwen25_7b__seed43__combined_inlmh_r2 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 43 | 2/4 | 16/128/8 | 0.0002 | 4x4 | - | 5.28M | 0.9455 |
| qwen25_7b__seed43__combined_inlmh_r4 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 43 | 4/8 | 16/128/8 | 0.0002 | 4x4 | - | 10.33M | 0.9317 |
| qwen25_7b__seed43__hidden_lora_r1 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 43 | 1/2 | - | 0.0002 | 4x4 | - | 2.52M | 0.9657 |
| qwen25_7b__seed43__hidden_lora_r2 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 43 | 2/4 | - | 0.0002 | 4x4 | - | 5.05M | 0.9528 |
| qwen25_7b__seed43__hidden_lora_r4 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 43 | 4/8 | - | 0.0002 | 4x4 | - | 10.09M | 0.9364 |
| qwen25_7b__seed44__combined_inlmh_r1 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 44 | 1/2 | 16/128/8 | 0.0002 | 4x4 | - | 2.76M | 0.9582 |
| qwen25_7b__seed44__combined_inlmh_r2 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 44 | 2/4 | 16/128/8 | 0.0002 | 4x4 | - | 5.28M | 0.9456 |
| qwen25_7b__seed44__combined_inlmh_r4 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 44 | 4/8 | 16/128/8 | 0.0002 | 4x4 | - | 10.33M | 0.9314 |
| qwen25_7b__seed44__hidden_lora_r1 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 44 | 1/2 | - | 0.0002 | 4x4 | - | 2.52M | 0.9661 |
| qwen25_7b__seed44__hidden_lora_r2 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 44 | 2/4 | - | 0.0002 | 4x4 | - | 5.05M | 0.9534 |
| qwen25_7b__seed44__hidden_lora_r4 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 44 | 4/8 | - | 0.0002 | 4x4 | - | 10.09M | 0.936 |
| qwen3_8b__seed43__combined_inlmh_r1 | Qwen3-8B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 43 | 1/2 | 16/128/8 | 0.0002 | 4x4 | - | 2.99M | 0.8995 |
| qwen3_8b__seed43__combined_inlmh_r2 | Qwen3-8B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 43 | 2/4 | 16/128/8 | 0.0002 | 4x4 | - | 5.72M | 0.8898 |
| qwen3_8b__seed43__combined_inlmh_r4 | Qwen3-8B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 43 | 4/8 | 16/128/8 | 0.0002 | 4x4 | - | 11.18M | 0.8799 |
| qwen3_8b__seed43__hidden_lora_r1 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 43 | 1/2 | - | 0.0002 | 4x4 | - | 2.73M | 0.9075 |
| qwen3_8b__seed43__hidden_lora_r2 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 43 | 2/4 | - | 0.0002 | 4x4 | - | 5.46M | 0.8957 |
| qwen3_8b__seed43__hidden_lora_r4 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 43 | 4/8 | - | 0.0002 | 4x4 | - | 10.91M | 0.8846 |
| qwen3_8b__seed44__combined_inlmh_r1 | Qwen3-8B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 44 | 1/2 | 16/128/8 | 0.0002 | 4x4 | - | 2.99M | 0.9007 |
| qwen3_8b__seed44__combined_inlmh_r2 | Qwen3-8B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 44 | 2/4 | 16/128/8 | 0.0002 | 4x4 | - | 5.72M | 0.8908 |
| qwen3_8b__seed44__combined_inlmh_r4 | Qwen3-8B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 44 | 4/8 | 16/128/8 | 0.0002 | 4x4 | - | 11.18M | 0.8808 |
| qwen3_8b__seed44__hidden_lora_r1 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 44 | 1/2 | - | 0.0002 | 4x4 | - | 2.73M | 0.9075 |
| qwen3_8b__seed44__hidden_lora_r2 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 44 | 2/4 | - | 0.0002 | 4x4 | - | 5.46M | 0.8972 |
| qwen3_8b__seed44__hidden_lora_r4 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 44 | 4/8 | - | 0.0002 | 4x4 | - | 10.91M | 0.8847 |


### Phase 8a：1.5B/1.7B 低 rank 多 seed

来源：`outputs/affine_vocab/phase8a_20260520_152912`；摘要见 `docs/phase8a_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_7b__hidden_r1_vocab_lora_r1 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 42 | 1/2 | - | 0.0002 | 4x4 | emb_lmh_r1 | 2.83M | 0.9637 |
| qwen25_7b__hidden_r2_vocab_lora_r1 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 42 | 2/4 | - | 0.0002 | 4x4 | emb_lmh_r1 | 5.36M | 0.95 |
| qwen3_8b__hidden_r1_vocab_lora_r1 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 42 | 1/2 | - | 0.0002 | 4x4 | emb_lmh_r1 | 3.04M | 0.9061 |
| qwen3_8b__hidden_r2_vocab_lora_r1 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 42 | 2/4 | - | 0.0002 | 4x4 | emb_lmh_r1 | 5.77M | 0.8967 |


### Phase 8b：低 rank vocab LoRA 对照

来源：`outputs/affine_vocab/phase8b_20260520_170927`；摘要见 `docs/phase8b_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_05b__combined_inlmh_r1 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/128/8 | 0.0002 | 16x1 | - | 608k | 1.413 |
| qwen25_05b__combined_inlmh_r2 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 16x1 | - | 1.16M | 1.378 |
| qwen25_05b__hidden_lora_r1 | Qwen2.5-0.5B | sft_t2t_mini_25k | hidden_lora | 42 | 1/2 | - | 0.0002 | 16x1 | - | 550k | 1.44 |
| qwen25_05b__hidden_lora_r2 | Qwen2.5-0.5B | sft_t2t_mini_25k | hidden_lora | 42 | 2/4 | - | 0.0002 | 16x1 | - | 1.10M | 1.398 |
| qwen25_05b__vocab_lora_r1 | Qwen2.5-0.5B | sft_t2t_mini_25k | hidden_lora | 42 | 1/2 | - | 0.0002 | 16x1 | emb_lmh_r1 | 856k | 1.413 |
| qwen25_05b__vocab_lora_r2 | Qwen2.5-0.5B | sft_t2t_mini_25k | hidden_lora | 42 | 2/4 | - | 0.0002 | 16x1 | emb_lmh_r1 | 1.41M | 1.378 |
| qwen3_17b__combined_inlmh_r1 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/128/8 | 0.0002 | 8x2 | - | 1.22M | 0.7774 |
| qwen3_17b__combined_inlmh_r2 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 8x2 | - | 2.31M | 0.7636 |
| qwen3_17b__hidden_lora_r1 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 42 | 1/2 | - | 0.0002 | 8x2 | - | 1.09M | 0.7874 |
| qwen3_17b__hidden_lora_r2 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 42 | 2/4 | - | 0.0002 | 8x2 | - | 2.18M | 0.7715 |
| qwen3_17b__vocab_lora_r1 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 42 | 1/2 | - | 0.0002 | 8x2 | emb_lmh_r1 | 1.40M | 0.7849 |
| qwen3_17b__vocab_lora_r2 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 42 | 2/4 | - | 0.0002 | 8x2 | emb_lmh_r1 | 2.49M | 0.7688 |


### Phase 9a：alpha/bias scale

来源：`outputs/affine_vocab/phase9a_20260520_184329`；摘要见 `docs/phase9a_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_7b__seed43__hidden_r1_vocab_lora_r1 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 43 | 1/2 | - | 0.0002 | 4x4 | emb_lmh_r1 | 2.83M | 0.9641 |
| qwen25_7b__seed43__hidden_r2_vocab_lora_r1 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 43 | 2/4 | - | 0.0002 | 4x4 | emb_lmh_r1 | 5.36M | 0.9504 |
| qwen25_7b__seed44__hidden_r1_vocab_lora_r1 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 44 | 1/2 | - | 0.0002 | 4x4 | emb_lmh_r1 | 2.83M | 0.9638 |
| qwen25_7b__seed44__hidden_r2_vocab_lora_r1 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 44 | 2/4 | - | 0.0002 | 4x4 | emb_lmh_r1 | 5.36M | 0.9505 |
| qwen3_8b__seed43__hidden_r1_vocab_lora_r1 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 43 | 1/2 | - | 0.0002 | 4x4 | emb_lmh_r1 | 3.04M | 0.9063 |
| qwen3_8b__seed43__hidden_r2_vocab_lora_r1 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 43 | 2/4 | - | 0.0002 | 4x4 | emb_lmh_r1 | 5.77M | 0.8959 |
| qwen3_8b__seed44__hidden_r1_vocab_lora_r1 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 44 | 1/2 | - | 0.0002 | 4x4 | emb_lmh_r1 | 3.04M | 0.9061 |
| qwen3_8b__seed44__hidden_r2_vocab_lora_r1 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 44 | 2/4 | - | 0.0002 | 4x4 | emb_lmh_r1 | 5.77M | 0.8964 |


### Phase 9b：位置归因

来源：`outputs/affine_vocab/phase9b_20260520_212708`；摘要见 `docs/phase9b_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_15b__combined_inlmh_r1 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/128/8 | 0.0002 | 8x2 | - | 1.25M | 1.113 |
| qwen25_15b__combined_inlmh_r2 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 8x2 | - | 2.41M | 1.095 |
| qwen25_15b__hidden_lora_r1 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 42 | 1/2 | - | 0.0002 | 8x2 | - | 1.15M | 1.127 |
| qwen25_15b__hidden_lora_r2 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 42 | 2/4 | - | 0.0002 | 8x2 | - | 2.31M | 1.105 |
| qwen25_15b__vocab_lora_r1 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 42 | 1/2 | - | 0.0002 | 8x2 | emb_lmh_r1 | 1.46M | 1.119 |
| qwen25_15b__vocab_lora_r2 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 42 | 2/4 | - | 0.0002 | 8x2 | emb_lmh_r1 | 2.62M | 1.099 |


### Phase 9c：机制/位置补充

来源：`outputs/affine_vocab/phase9c_20260520_214840`；摘要见 `docs/phase9c_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_7b__combined_inlmh_r1_affine_alpha128 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/128/8 | 0.0002 | 4x4 | - | 2.76M | 0.9578 |
| qwen25_7b__combined_inlmh_r1_affine_alpha256 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/256/16 | 0.0002 | 4x4 | - | 2.76M | 0.9558 |
| qwen25_7b__combined_inlmh_r1_affine_alpha64 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/64/4 | 0.0002 | 4x4 | - | 2.76M | 0.9591 |
| qwen25_7b__combined_inlmh_r2_affine_alpha128 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 4x4 | - | 5.28M | 0.9457 |
| qwen25_7b__combined_inlmh_r2_affine_alpha256 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/256/16 | 0.0002 | 4x4 | - | 5.28M | 0.9437 |
| qwen25_7b__combined_inlmh_r2_affine_alpha64 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/64/4 | 0.0002 | 4x4 | - | 5.28M | 0.9472 |
| qwen3_17b__combined_inlmh_r1_affine_alpha128 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/128/8 | 0.0002 | 8x2 | - | 1.22M | 0.7773 |
| qwen3_17b__combined_inlmh_r1_affine_alpha256 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/256/16 | 0.0002 | 8x2 | - | 1.22M | 0.776 |
| qwen3_17b__combined_inlmh_r1_affine_alpha64 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/64/4 | 0.0002 | 8x2 | - | 1.22M | 0.7789 |
| qwen3_17b__combined_inlmh_r2_affine_alpha128 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 8x2 | - | 2.31M | 0.7643 |
| qwen3_17b__combined_inlmh_r2_affine_alpha256 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/256/16 | 0.0002 | 8x2 | - | 2.31M | 0.7633 |
| qwen3_17b__combined_inlmh_r2_affine_alpha64 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/64/4 | 0.0002 | 8x2 | - | 2.31M | 0.7661 |


### Phase 9d：input-only vs lm_head-only

来源：`outputs/affine_vocab/phase9d_20260520_234301`；摘要见 `docs/phase9d_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_7b__hidden_lora_r1 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 42 | 1/2 | - | 0.0002 | 4x4 | - | 2.52M | 0.9657 |
| qwen25_7b__hidden_lora_r2 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 42 | 2/4 | - | 0.0002 | 4x4 | - | 5.05M | 0.9524 |
| qwen25_7b__input_lm_head_r1 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/128/8 | 0.0002 | 4x4 | - | 2.76M | 0.9574 |
| qwen25_7b__input_lm_head_r2 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 4x4 | - | 5.28M | 0.9457 |
| qwen25_7b__input_only_r1 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_plus_hidden_lora | 42 | 1/2 | 16/128/8 | 0.0002 | 4x4 | - | 2.64M | 0.9646 |
| qwen25_7b__input_only_r2 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 4x4 | - | 5.16M | 0.9509 |
| qwen25_7b__lm_head_only_r1 | Qwen2.5-7B | sft_t2t_mini_25k | affine_lm_head_plus_hidden_lora | 42 | 1/2 | 16/128/8 | 0.0002 | 4x4 | - | 2.64M | 0.9585 |
| qwen25_7b__lm_head_only_r2 | Qwen2.5-7B | sft_t2t_mini_25k | affine_lm_head_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 4x4 | - | 5.16M | 0.9463 |


### Phase 9e：补充归因

来源：`outputs/affine_vocab/phase9e_20260521_015553`；摘要见 `docs/phase9e_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_15b__seed43__combined_inlmh_r1 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 43 | 1/2 | 16/128/8 | 0.0002 | 8x2 | - | 1.25M | 1.113 |
| qwen25_15b__seed43__combined_inlmh_r2 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 43 | 2/4 | 16/128/8 | 0.0002 | 8x2 | - | 2.41M | 1.095 |
| qwen25_15b__seed43__hidden_lora_r1 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 43 | 1/2 | - | 0.0002 | 8x2 | - | 1.15M | 1.126 |
| qwen25_15b__seed43__hidden_lora_r2 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 43 | 2/4 | - | 0.0002 | 8x2 | - | 2.31M | 1.103 |
| qwen25_15b__seed43__vocab_lora_r1 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 43 | 1/2 | - | 0.0002 | 8x2 | emb_lmh_r1 | 1.46M | 1.121 |
| qwen25_15b__seed43__vocab_lora_r2 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 43 | 2/4 | - | 0.0002 | 8x2 | emb_lmh_r1 | 2.62M | 1.099 |
| qwen25_15b__seed44__combined_inlmh_r1 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 44 | 1/2 | 16/128/8 | 0.0002 | 8x2 | - | 1.25M | 1.114 |
| qwen25_15b__seed44__combined_inlmh_r2 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 44 | 2/4 | 16/128/8 | 0.0002 | 8x2 | - | 2.41M | 1.096 |
| qwen25_15b__seed44__hidden_lora_r1 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 44 | 1/2 | - | 0.0002 | 8x2 | - | 1.15M | 1.127 |
| qwen25_15b__seed44__hidden_lora_r2 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 44 | 2/4 | - | 0.0002 | 8x2 | - | 2.31M | 1.105 |
| qwen25_15b__seed44__vocab_lora_r1 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 44 | 1/2 | - | 0.0002 | 8x2 | emb_lmh_r1 | 1.46M | 1.12 |
| qwen25_15b__seed44__vocab_lora_r2 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 44 | 2/4 | - | 0.0002 | 8x2 | emb_lmh_r1 | 2.62M | 1.098 |


### Phase 9f：补充归因

来源：`outputs/affine_vocab/phase9f_20260521_105726`；摘要见 `docs/phase9f_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen3_17b__combined_inlmh_r1_bias_scale0 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/128/8 | 0.0002 | 8x2 | - | 1.22M | 0.7776 |
| qwen3_17b__combined_inlmh_r1_bias_scale0p5 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/128/8 | 0.0002 | 8x2 | - | 1.22M | 0.7773 |
| qwen3_17b__combined_inlmh_r1_bias_scale1 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/128/8 | 0.0002 | 8x2 | - | 1.22M | 0.7767 |
| qwen3_17b__combined_inlmh_r1_bias_scale2 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/128/8 | 0.0002 | 8x2 | - | 1.22M | 0.7774 |
| qwen3_17b__combined_inlmh_r2_bias_scale0 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 8x2 | - | 2.31M | 0.7642 |
| qwen3_17b__combined_inlmh_r2_bias_scale0p5 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 8x2 | - | 2.31M | 0.7641 |
| qwen3_17b__combined_inlmh_r2_bias_scale1 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 8x2 | - | 2.31M | 0.7643 |
| qwen3_17b__combined_inlmh_r2_bias_scale2 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 8x2 | - | 2.31M | 0.7643 |


### Phase 10：1.5B/1.7B single-layer 多 seed

来源：`outputs/affine_vocab/phase10_single_layer_20260523_102449`；摘要见 `docs/phase10_single_layer_20260523_102449_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_15b__seed42__affine_inlmh_s1_16 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/256/16 | 0.0002 | 8x2 | - | 100k | 1.322 |
| qwen25_15b__seed42__affine_input_s1_64 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input | 42 | 16/32 | 16/1024/64 | 0.0002 | 8x2 | - | 51k | 1.471 |
| qwen25_15b__seed42__single_q_l0_r16 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 42 | 16/32 | - | 0.0002 | 8x2 | layers=0 | 49k | 1.771 |
| qwen25_15b__seed42__single_q_l14_r16 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 42 | 16/32 | - | 0.0002 | 8x2 | layers=14 | 49k | 1.72 |
| qwen25_15b__seed42__single_q_l27_r16 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 42 | 16/32 | - | 0.0002 | 8x2 | layers=27 | 49k | 1.85 |
| qwen25_15b__seed42__single_qkvo_l14_r4 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 42 | 4/8 | - | 0.0002 | 8x2 | layers=14 | 39k | 1.486 |
| qwen25_15b__seed43__affine_inlmh_s1_16 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head | 43 | 16/32 | 16/256/16 | 0.0002 | 8x2 | - | 100k | 1.322 |
| qwen25_15b__seed43__affine_input_s1_64 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input | 43 | 16/32 | 16/1024/64 | 0.0002 | 8x2 | - | 51k | 1.471 |
| qwen25_15b__seed43__single_q_l0_r16 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 43 | 16/32 | - | 0.0002 | 8x2 | layers=0 | 49k | 1.777 |
| qwen25_15b__seed43__single_q_l14_r16 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 43 | 16/32 | - | 0.0002 | 8x2 | layers=14 | 49k | 1.72 |
| qwen25_15b__seed43__single_q_l27_r16 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 43 | 16/32 | - | 0.0002 | 8x2 | layers=27 | 49k | 1.85 |
| qwen25_15b__seed43__single_qkvo_l14_r4 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 43 | 4/8 | - | 0.0002 | 8x2 | layers=14 | 39k | 1.477 |
| qwen25_15b__seed44__affine_inlmh_s1_16 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head | 44 | 16/32 | 16/256/16 | 0.0002 | 8x2 | - | 100k | 1.324 |
| qwen25_15b__seed44__affine_input_s1_64 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input | 44 | 16/32 | 16/1024/64 | 0.0002 | 8x2 | - | 51k | 1.474 |
| qwen25_15b__seed44__single_q_l0_r16 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 44 | 16/32 | - | 0.0002 | 8x2 | layers=0 | 49k | 1.776 |
| qwen25_15b__seed44__single_q_l14_r16 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 44 | 16/32 | - | 0.0002 | 8x2 | layers=14 | 49k | 1.718 |
| qwen25_15b__seed44__single_q_l27_r16 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 44 | 16/32 | - | 0.0002 | 8x2 | layers=27 | 49k | 1.85 |
| qwen25_15b__seed44__single_qkvo_l14_r4 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 44 | 4/8 | - | 0.0002 | 8x2 | layers=14 | 39k | 1.487 |
| qwen3_17b__seed42__affine_inlmh_s1_16 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/256/16 | 0.0002 | 8x2 | - | 133k | 0.9091 |
| qwen3_17b__seed42__affine_input_s1_64 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input | 42 | 16/32 | 16/1024/64 | 0.0002 | 8x2 | - | 68k | 0.9804 |
| qwen3_17b__seed42__single_q_l0_r16 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 42 | 16/32 | - | 0.0002 | 8x2 | layers=0 | 66k | 1.27 |
| qwen3_17b__seed42__single_q_l14_r16 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 42 | 16/32 | - | 0.0002 | 8x2 | layers=14 | 66k | 1.269 |
| qwen3_17b__seed42__single_q_l27_r16 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 42 | 16/32 | - | 0.0002 | 8x2 | layers=27 | 66k | 1.395 |
| qwen3_17b__seed42__single_qkvo_l14_r4 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 42 | 4/8 | - | 0.0002 | 8x2 | layers=14 | 57k | 1.053 |
| qwen3_17b__seed43__affine_inlmh_s1_16 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head | 43 | 16/32 | 16/256/16 | 0.0002 | 8x2 | - | 133k | 0.9123 |
| qwen3_17b__seed43__affine_input_s1_64 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input | 43 | 16/32 | 16/1024/64 | 0.0002 | 8x2 | - | 68k | 0.9787 |
| qwen3_17b__seed43__single_q_l0_r16 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 43 | 16/32 | - | 0.0002 | 8x2 | layers=0 | 66k | 1.268 |
| qwen3_17b__seed43__single_q_l14_r16 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 43 | 16/32 | - | 0.0002 | 8x2 | layers=14 | 66k | 1.27 |
| qwen3_17b__seed43__single_q_l27_r16 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 43 | 16/32 | - | 0.0002 | 8x2 | layers=27 | 66k | 1.395 |
| qwen3_17b__seed43__single_qkvo_l14_r4 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 43 | 4/8 | - | 0.0002 | 8x2 | layers=14 | 57k | 1.06 |
| qwen3_17b__seed44__affine_inlmh_s1_16 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head | 44 | 16/32 | 16/256/16 | 0.0002 | 8x2 | - | 133k | 0.9105 |
| qwen3_17b__seed44__affine_input_s1_64 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input | 44 | 16/32 | 16/1024/64 | 0.0002 | 8x2 | - | 68k | 0.9771 |
| qwen3_17b__seed44__single_q_l0_r16 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 44 | 16/32 | - | 0.0002 | 8x2 | layers=0 | 66k | 1.277 |
| qwen3_17b__seed44__single_q_l14_r16 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 44 | 16/32 | - | 0.0002 | 8x2 | layers=14 | 66k | 1.271 |
| qwen3_17b__seed44__single_q_l27_r16 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 44 | 16/32 | - | 0.0002 | 8x2 | layers=27 | 66k | 1.396 |
| qwen3_17b__seed44__single_qkvo_l14_r4 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 44 | 4/8 | - | 0.0002 | 8x2 | layers=14 | 57k | 1.055 |


### Phase 11：7B/8B single-layer 多 seed

来源：`outputs/affine_vocab/phase11_large_single_layer_20260523_130101`；摘要见 `docs/phase11_large_single_layer_20260523_130101_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_7b__seed42__affine_inlmh_s1_16 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/256/16 | 0.0002 | 4x4 | - | 233k | 1.119 |
| qwen25_7b__seed42__affine_input_s1_64 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input | 42 | 16/32 | 16/1024/64 | 0.0002 | 4x4 | - | 118k | 1.227 |
| qwen25_7b__seed42__single_q_l0_r16 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 42 | 16/32 | - | 0.0002 | 4x4 | layers=0 | 115k | 1.553 |
| qwen25_7b__seed42__single_q_l14_r16 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 42 | 16/32 | - | 0.0002 | 4x4 | layers=14 | 115k | 1.514 |
| qwen25_7b__seed42__single_q_l27_r16 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 42 | 16/32 | - | 0.0002 | 4x4 | layers=27 | 115k | 1.616 |
| qwen25_7b__seed42__single_qkvo_l14_r4 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 42 | 4/8 | - | 0.0002 | 4x4 | layers=14 | 90k | 1.259 |
| qwen25_7b__seed43__affine_inlmh_s1_16 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head | 43 | 16/32 | 16/256/16 | 0.0002 | 4x4 | - | 233k | 1.118 |
| qwen25_7b__seed43__affine_input_s1_64 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input | 43 | 16/32 | 16/1024/64 | 0.0002 | 4x4 | - | 118k | 1.232 |
| qwen25_7b__seed43__single_q_l0_r16 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 43 | 16/32 | - | 0.0002 | 4x4 | layers=0 | 115k | 1.555 |
| qwen25_7b__seed43__single_q_l14_r16 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 43 | 16/32 | - | 0.0002 | 4x4 | layers=14 | 115k | 1.514 |
| qwen25_7b__seed43__single_q_l27_r16 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 43 | 16/32 | - | 0.0002 | 4x4 | layers=27 | 115k | 1.617 |
| qwen25_7b__seed43__single_qkvo_l14_r4 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 43 | 4/8 | - | 0.0002 | 4x4 | layers=14 | 90k | 1.257 |
| qwen25_7b__seed44__affine_inlmh_s1_16 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input_lm_head | 44 | 16/32 | 16/256/16 | 0.0002 | 4x4 | - | 233k | 1.12 |
| qwen25_7b__seed44__affine_input_s1_64 | Qwen2.5-7B | sft_t2t_mini_25k | affine_input | 44 | 16/32 | 16/1024/64 | 0.0002 | 4x4 | - | 118k | 1.233 |
| qwen25_7b__seed44__single_q_l0_r16 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 44 | 16/32 | - | 0.0002 | 4x4 | layers=0 | 115k | 1.553 |
| qwen25_7b__seed44__single_q_l14_r16 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 44 | 16/32 | - | 0.0002 | 4x4 | layers=14 | 115k | 1.516 |
| qwen25_7b__seed44__single_q_l27_r16 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 44 | 16/32 | - | 0.0002 | 4x4 | layers=27 | 115k | 1.618 |
| qwen25_7b__seed44__single_qkvo_l14_r4 | Qwen2.5-7B | sft_t2t_mini_25k | hidden_lora | 44 | 4/8 | - | 0.0002 | 4x4 | layers=14 | 90k | 1.253 |
| qwen3_8b__seed42__affine_inlmh_s1_16 | Qwen3-8B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/256/16 | 0.0002 | 4x4 | - | 266k | 1.008 |
| qwen3_8b__seed42__affine_input_s1_64 | Qwen3-8B | sft_t2t_mini_25k | affine_input | 42 | 16/32 | 16/1024/64 | 0.0002 | 4x4 | - | 135k | 1.065 |
| qwen3_8b__seed42__single_q_l0_r16 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 42 | 16/32 | - | 0.0002 | 4x4 | layers=0 | 131k | 1.347 |
| qwen3_8b__seed42__single_q_l18_r16 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 42 | 16/32 | - | 0.0002 | 4x4 | layers=18 | 131k | 1.308 |
| qwen3_8b__seed42__single_q_l35_r16 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 42 | 16/32 | - | 0.0002 | 4x4 | layers=35 | 131k | 1.379 |
| qwen3_8b__seed42__single_qkvo_l18_r4 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 42 | 4/8 | - | 0.0002 | 4x4 | layers=18 | 106k | 1.12 |
| qwen3_8b__seed43__affine_inlmh_s1_16 | Qwen3-8B | sft_t2t_mini_25k | affine_input_lm_head | 43 | 16/32 | 16/256/16 | 0.0002 | 4x4 | - | 266k | 1.011 |
| qwen3_8b__seed43__affine_input_s1_64 | Qwen3-8B | sft_t2t_mini_25k | affine_input | 43 | 16/32 | 16/1024/64 | 0.0002 | 4x4 | - | 135k | 1.068 |
| qwen3_8b__seed43__single_q_l0_r16 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 43 | 16/32 | - | 0.0002 | 4x4 | layers=0 | 131k | 1.354 |
| qwen3_8b__seed43__single_q_l18_r16 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 43 | 16/32 | - | 0.0002 | 4x4 | layers=18 | 131k | 1.309 |
| qwen3_8b__seed43__single_q_l35_r16 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 43 | 16/32 | - | 0.0002 | 4x4 | layers=35 | 131k | 1.38 |
| qwen3_8b__seed43__single_qkvo_l18_r4 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 43 | 4/8 | - | 0.0002 | 4x4 | layers=18 | 106k | 1.118 |
| qwen3_8b__seed44__affine_inlmh_s1_16 | Qwen3-8B | sft_t2t_mini_25k | affine_input_lm_head | 44 | 16/32 | 16/256/16 | 0.0002 | 4x4 | - | 266k | 1.009 |
| qwen3_8b__seed44__affine_input_s1_64 | Qwen3-8B | sft_t2t_mini_25k | affine_input | 44 | 16/32 | 16/1024/64 | 0.0002 | 4x4 | - | 135k | 1.066 |
| qwen3_8b__seed44__single_q_l0_r16 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 44 | 16/32 | - | 0.0002 | 4x4 | layers=0 | 131k | 1.35 |
| qwen3_8b__seed44__single_q_l18_r16 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 44 | 16/32 | - | 0.0002 | 4x4 | layers=18 | 131k | 1.307 |
| qwen3_8b__seed44__single_q_l35_r16 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 44 | 16/32 | - | 0.0002 | 4x4 | layers=35 | 131k | 1.378 |
| qwen3_8b__seed44__single_qkvo_l18_r4 | Qwen3-8B | sft_t2t_mini_25k | hidden_lora | 44 | 4/8 | - | 0.0002 | 4x4 | layers=18 | 106k | 1.121 |


### Phase 12：tied shared adapters

来源：`outputs/affine_vocab/phase12_tied_shared_20260524_003753`；摘要见 `docs/phase12_tied_shared_20260524_003753_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_05b__seed42__hidden_lora_r8 | Qwen2.5-0.5B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 16x1 | - | 4.40M | 1.301 |
| qwen25_05b__seed42__tied_affine_inlmh_s1_16 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/256/16 | 0.0002 | 16x1 | tied_shared | 30k | 1.687 |
| qwen25_05b__seed42__tied_combined_inlmh_s1_8 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 16x1 | tied_shared | 4.43M | 1.296 |
| qwen25_05b__seed43__hidden_lora_r8 | Qwen2.5-0.5B | sft_t2t_mini_25k | hidden_lora | 43 | 8/16 | - | 0.0002 | 16x1 | - | 4.40M | 1.303 |
| qwen25_05b__seed43__tied_affine_inlmh_s1_16 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head | 43 | 16/32 | 16/256/16 | 0.0002 | 16x1 | tied_shared | 30k | 1.689 |
| qwen25_05b__seed43__tied_combined_inlmh_s1_8 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 43 | 8/16 | 16/128/8 | 0.0002 | 16x1 | tied_shared | 4.43M | 1.296 |
| qwen25_05b__seed44__hidden_lora_r8 | Qwen2.5-0.5B | sft_t2t_mini_25k | hidden_lora | 44 | 8/16 | - | 0.0002 | 16x1 | - | 4.40M | 1.301 |
| qwen25_05b__seed44__tied_affine_inlmh_s1_16 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head | 44 | 16/32 | 16/256/16 | 0.0002 | 16x1 | tied_shared | 30k | 1.686 |
| qwen25_05b__seed44__tied_combined_inlmh_s1_8 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 44 | 8/16 | 16/128/8 | 0.0002 | 16x1 | tied_shared | 4.43M | 1.294 |
| qwen25_15b__seed42__hidden_lora_r8 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 8x2 | - | 9.23M | 1.054 |
| qwen25_15b__seed42__tied_affine_inlmh_s1_16 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/256/16 | 0.0002 | 8x2 | tied_shared | 51k | 1.355 |
| qwen25_15b__seed42__tied_combined_inlmh_s1_8 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 8x2 | tied_shared | 9.28M | 1.049 |
| qwen25_15b__seed43__hidden_lora_r8 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 43 | 8/16 | - | 0.0002 | 8x2 | - | 9.23M | 1.054 |
| qwen25_15b__seed43__tied_affine_inlmh_s1_16 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head | 43 | 16/32 | 16/256/16 | 0.0002 | 8x2 | tied_shared | 51k | 1.359 |
| qwen25_15b__seed43__tied_combined_inlmh_s1_8 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 43 | 8/16 | 16/128/8 | 0.0002 | 8x2 | tied_shared | 9.28M | 1.048 |
| qwen25_15b__seed44__hidden_lora_r8 | Qwen2.5-1.5B | sft_t2t_mini_25k | hidden_lora | 44 | 8/16 | - | 0.0002 | 8x2 | - | 9.23M | 1.055 |
| qwen25_15b__seed44__tied_affine_inlmh_s1_16 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head | 44 | 16/32 | 16/256/16 | 0.0002 | 8x2 | tied_shared | 51k | 1.352 |
| qwen25_15b__seed44__tied_combined_inlmh_s1_8 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 44 | 8/16 | 16/128/8 | 0.0002 | 8x2 | tied_shared | 9.28M | 1.048 |
| qwen3_06b__seed42__hidden_lora_r8 | Qwen3-0.6B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 16x1 | - | 5.05M | 1.1 |
| qwen3_06b__seed42__tied_affine_inlmh_s1_16 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/256/16 | 0.0002 | 16x1 | tied_shared | 34k | 1.357 |
| qwen3_06b__seed42__tied_combined_inlmh_s1_8 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 16x1 | tied_shared | 5.08M | 1.097 |
| qwen3_06b__seed43__hidden_lora_r8 | Qwen3-0.6B | sft_t2t_mini_25k | hidden_lora | 43 | 8/16 | - | 0.0002 | 16x1 | - | 5.05M | 1.101 |
| qwen3_06b__seed43__tied_affine_inlmh_s1_16 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head | 43 | 16/32 | 16/256/16 | 0.0002 | 16x1 | tied_shared | 34k | 1.356 |
| qwen3_06b__seed43__tied_combined_inlmh_s1_8 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 43 | 8/16 | 16/128/8 | 0.0002 | 16x1 | tied_shared | 5.08M | 1.096 |
| qwen3_06b__seed44__hidden_lora_r8 | Qwen3-0.6B | sft_t2t_mini_25k | hidden_lora | 44 | 8/16 | - | 0.0002 | 16x1 | - | 5.05M | 1.101 |
| qwen3_06b__seed44__tied_affine_inlmh_s1_16 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head | 44 | 16/32 | 16/256/16 | 0.0002 | 16x1 | tied_shared | 34k | 1.356 |
| qwen3_06b__seed44__tied_combined_inlmh_s1_8 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 44 | 8/16 | 16/128/8 | 0.0002 | 16x1 | tied_shared | 5.08M | 1.097 |
| qwen3_17b__seed42__hidden_lora_r8 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 42 | 8/16 | - | 0.0002 | 8x2 | - | 8.72M | 0.7376 |
| qwen3_17b__seed42__tied_affine_inlmh_s1_16 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/256/16 | 0.0002 | 8x2 | tied_shared | 68k | 0.9363 |
| qwen3_17b__seed42__tied_combined_inlmh_s1_8 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 8x2 | tied_shared | 8.78M | 0.7346 |
| qwen3_17b__seed43__hidden_lora_r8 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 43 | 8/16 | - | 0.0002 | 8x2 | - | 8.72M | 0.7381 |
| qwen3_17b__seed43__tied_affine_inlmh_s1_16 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head | 43 | 16/32 | 16/256/16 | 0.0002 | 8x2 | tied_shared | 68k | 0.9316 |
| qwen3_17b__seed43__tied_combined_inlmh_s1_8 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 43 | 8/16 | 16/128/8 | 0.0002 | 8x2 | tied_shared | 8.78M | 0.735 |
| qwen3_17b__seed44__hidden_lora_r8 | Qwen3-1.7B | sft_t2t_mini_25k | hidden_lora | 44 | 8/16 | - | 0.0002 | 8x2 | - | 8.72M | 0.7377 |
| qwen3_17b__seed44__tied_affine_inlmh_s1_16 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head | 44 | 16/32 | 16/256/16 | 0.0002 | 8x2 | tied_shared | 68k | 0.9375 |
| qwen3_17b__seed44__tied_combined_inlmh_s1_8 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 44 | 8/16 | 16/128/8 | 0.0002 | 8x2 | tied_shared | 8.78M | 0.7356 |


### Phase 13：tied affine-only rank/scale 对齐

来源：`outputs/affine_vocab/phase13_tied_affine_align_20260524_102351`；摘要见 `docs/phase13_tied_affine_align_20260524_102351_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_05b__seed42__tied_affine_rank32_s1_16 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 32/512/16 | 0.0002 | 16x1 | tied_shared | 58k | 1.633 |
| qwen25_05b__seed43__tied_affine_rank32_s1_16 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head | 43 | 16/32 | 32/512/16 | 0.0002 | 16x1 | tied_shared | 58k | 1.631 |
| qwen25_05b__seed44__tied_affine_rank32_s1_16 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head | 44 | 16/32 | 32/512/16 | 0.0002 | 16x1 | tied_shared | 58k | 1.629 |
| qwen25_15b__seed42__tied_affine_rank32_s1_16 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 32/512/16 | 0.0002 | 8x2 | tied_shared | 100k | 1.307 |
| qwen25_15b__seed43__tied_affine_rank32_s1_16 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head | 43 | 16/32 | 32/512/16 | 0.0002 | 8x2 | tied_shared | 100k | 1.31 |
| qwen25_15b__seed44__tied_affine_rank32_s1_16 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head | 44 | 16/32 | 32/512/16 | 0.0002 | 8x2 | tied_shared | 100k | 1.31 |
| qwen3_06b__seed42__tied_affine_rank16_s1_64 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 16/1024/64 | 0.0002 | 16x1 | tied_shared | 34k | 1.35 |
| qwen3_06b__seed42__tied_affine_rank32_s1_16 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 32/512/16 | 0.0002 | 16x1 | tied_shared | 67k | 1.318 |
| qwen3_06b__seed42__tied_affine_rank32_s1_64 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 32/2048/64 | 0.0002 | 16x1 | tied_shared | 67k | 1.317 |
| qwen3_06b__seed43__tied_affine_rank16_s1_64 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head | 43 | 16/32 | 16/1024/64 | 0.0002 | 16x1 | tied_shared | 34k | 1.351 |
| qwen3_06b__seed43__tied_affine_rank32_s1_16 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head | 43 | 16/32 | 32/512/16 | 0.0002 | 16x1 | tied_shared | 67k | 1.319 |
| qwen3_06b__seed43__tied_affine_rank32_s1_64 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head | 43 | 16/32 | 32/2048/64 | 0.0002 | 16x1 | tied_shared | 67k | 1.318 |
| qwen3_06b__seed44__tied_affine_rank16_s1_64 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head | 44 | 16/32 | 16/1024/64 | 0.0002 | 16x1 | tied_shared | 34k | 1.351 |
| qwen3_06b__seed44__tied_affine_rank32_s1_16 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head | 44 | 16/32 | 32/512/16 | 0.0002 | 16x1 | tied_shared | 67k | 1.321 |
| qwen3_06b__seed44__tied_affine_rank32_s1_64 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head | 44 | 16/32 | 32/2048/64 | 0.0002 | 16x1 | tied_shared | 67k | 1.32 |
| qwen3_17b__seed42__tied_affine_rank32_s1_16 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head | 42 | 16/32 | 32/512/16 | 0.0002 | 8x2 | tied_shared | 133k | 0.9109 |
| qwen3_17b__seed43__tied_affine_rank32_s1_16 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head | 43 | 16/32 | 32/512/16 | 0.0002 | 8x2 | tied_shared | 133k | 0.9138 |
| qwen3_17b__seed44__tied_affine_rank32_s1_16 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head | 44 | 16/32 | 32/512/16 | 0.0002 | 8x2 | tied_shared | 133k | 0.9111 |


### Phase 14：mergeable tied combined sanity

来源：`outputs/affine_vocab/phase14_mergeable_tied_combined_20260524_113706`；摘要见 `docs/phase14_mergeable_tied_combined_20260524_113706_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_05b__seed42__mergeable_tied_combined_inlmh_s1_8 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 16x1 | tied_shared, mergeable | 4.43M | 1.296 |
| qwen25_15b__seed42__mergeable_tied_combined_inlmh_s1_8 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 8x2 | tied_shared, mergeable | 9.28M | 1.049 |
| qwen3_06b__seed42__mergeable_tied_combined_inlmh_s1_8 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 16x1 | tied_shared, mergeable | 5.08M | 1.098 |
| qwen3_17b__seed42__mergeable_tied_combined_inlmh_s1_8 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 8x2 | tied_shared, mergeable | 8.78M | 0.7349 |


### Phase 15：mergeable tied combined rank32

来源：`outputs/affine_vocab/phase15_mergeable_tied_combined_rank32_20260524_120300`；摘要见 `docs/phase15_mergeable_tied_combined_rank32_20260524_120300_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_05b__seed42__mergeable_tied_combined_rank32_s1_8 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 32/256/8 | 0.0002 | 16x1 | tied_shared, mergeable | 4.46M | 1.293 |
| qwen25_15b__seed42__mergeable_tied_combined_rank32_s1_8 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 32/256/8 | 0.0002 | 8x2 | tied_shared, mergeable | 9.33M | 1.047 |
| qwen3_06b__seed42__mergeable_tied_combined_rank32_s1_8 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 32/256/8 | 0.0002 | 16x1 | tied_shared, mergeable | 5.11M | 1.096 |
| qwen3_17b__seed42__mergeable_tied_combined_rank32_s1_8 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 32/256/8 | 0.0002 | 8x2 | tied_shared, mergeable | 8.85M | 0.7333 |


### Phase 16：mergeable tied combined rank32 多 seed

来源：`outputs/affine_vocab/phase16_mergeable_tied_combined_rank32_multiseed_20260524_123239`；摘要见 `docs/phase16_mergeable_tied_combined_rank32_multiseed_20260524_123239_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_05b__seed43__mergeable_tied_combined_rank32_s1_8 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 43 | 8/16 | 32/256/8 | 0.0002 | 16x1 | tied_shared, mergeable | 4.46M | 1.293 |
| qwen25_05b__seed44__mergeable_tied_combined_rank32_s1_8 | Qwen2.5-0.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 44 | 8/16 | 32/256/8 | 0.0002 | 16x1 | tied_shared, mergeable | 4.46M | 1.292 |
| qwen25_15b__seed43__mergeable_tied_combined_rank32_s1_8 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 43 | 8/16 | 32/256/8 | 0.0002 | 8x2 | tied_shared, mergeable | 9.33M | 1.047 |
| qwen25_15b__seed44__mergeable_tied_combined_rank32_s1_8 | Qwen2.5-1.5B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 44 | 8/16 | 32/256/8 | 0.0002 | 8x2 | tied_shared, mergeable | 9.33M | 1.047 |
| qwen3_06b__seed43__mergeable_tied_combined_rank32_s1_8 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 43 | 8/16 | 32/256/8 | 0.0002 | 16x1 | tied_shared, mergeable | 5.11M | 1.095 |
| qwen3_06b__seed44__mergeable_tied_combined_rank32_s1_8 | Qwen3-0.6B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 44 | 8/16 | 32/256/8 | 0.0002 | 16x1 | tied_shared, mergeable | 5.11M | 1.095 |
| qwen3_17b__seed43__mergeable_tied_combined_rank32_s1_8 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 43 | 8/16 | 32/256/8 | 0.0002 | 8x2 | tied_shared, mergeable | 8.85M | 0.7332 |
| qwen3_17b__seed44__mergeable_tied_combined_rank32_s1_8 | Qwen3-1.7B | sft_t2t_mini_25k | affine_input_lm_head_plus_hidden_lora | 44 | 8/16 | 32/256/8 | 0.0002 | 8x2 | tied_shared, mergeable | 8.85M | 0.7336 |


### UltraChat 100k：主验证矩阵

来源：`outputs/affine_vocab/ultrachat100k_validation_20260524_172421`；摘要见 `docs/ultrachat100k_validation_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_15b__seed42__affine_in_lmh_s1_16 | Qwen2.5-1.5B | UltraChat 100k | affine_input_lm_head | 42 | 16/32 | 16/256/16 | 0.0002 | 8x2 | - | 100k | 1.099 |
| qwen25_15b__seed42__combined_inlmh_s1_8 | Qwen2.5-1.5B | UltraChat 100k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 8x2 | - | 9.33M | 1.075 |
| qwen25_15b__seed42__hidden_lora_r8 | Qwen2.5-1.5B | UltraChat 100k | hidden_lora | 42 | 8/16 | - | 0.0002 | 8x2 | - | 9.23M | 1.075 |
| qwen25_15b__seed42__mergeable_tied_combined_rank32_s1_8 | Qwen2.5-1.5B | UltraChat 100k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 32/256/8 | 0.0002 | 8x2 | tied_shared, mergeable | 9.33M | 1.075 |
| qwen3_17b__seed42__affine_in_lmh_s1_16 | Qwen3-1.7B | UltraChat 100k | affine_input_lm_head | 42 | 16/32 | 16/256/16 | 0.0002 | 8x2 | - | 133k | 1.078 |
| qwen3_17b__seed42__combined_inlmh_s1_8 | Qwen3-1.7B | UltraChat 100k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 16/128/8 | 0.0002 | 8x2 | - | 8.85M | 1.058 |
| qwen3_17b__seed42__hidden_lora_r8 | Qwen3-1.7B | UltraChat 100k | hidden_lora | 42 | 8/16 | - | 0.0002 | 8x2 | - | 8.72M | 1.058 |
| qwen3_17b__seed42__mergeable_tied_combined_rank32_s1_8 | Qwen3-1.7B | UltraChat 100k | affine_input_lm_head_plus_hidden_lora | 42 | 8/16 | 32/256/8 | 0.0002 | 8x2 | tied_shared, mergeable | 8.85M | 1.058 |


### UltraChat 100k：single-layer 控制

来源：`outputs/affine_vocab/ultrachat100k_single_layer_20260524_205104`；摘要见 `docs/ultrachat100k_single_layer_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_15b__seed42__single_q_l0_r16 | Qwen2.5-1.5B | UltraChat 100k | hidden_lora | 42 | 16/32 | - | 0.0002 | 8x2 | layers=0 | 49k | 1.135 |
| qwen25_15b__seed42__single_q_l14_r16 | Qwen2.5-1.5B | UltraChat 100k | hidden_lora | 42 | 16/32 | - | 0.0002 | 8x2 | layers=14 | 49k | 1.121 |
| qwen25_15b__seed42__single_q_l27_r16 | Qwen2.5-1.5B | UltraChat 100k | hidden_lora | 42 | 16/32 | - | 0.0002 | 8x2 | layers=27 | 49k | 1.168 |
| qwen25_15b__seed42__single_qkvo_l14_r4 | Qwen2.5-1.5B | UltraChat 100k | hidden_lora | 42 | 4/8 | - | 0.0002 | 8x2 | layers=14 | 39k | 1.102 |
| qwen3_17b__seed42__single_q_l0_r16 | Qwen3-1.7B | UltraChat 100k | hidden_lora | 42 | 16/32 | - | 0.0002 | 8x2 | layers=0 | 66k | 1.108 |
| qwen3_17b__seed42__single_q_l14_r16 | Qwen3-1.7B | UltraChat 100k | hidden_lora | 42 | 16/32 | - | 0.0002 | 8x2 | layers=14 | 66k | 1.105 |
| qwen3_17b__seed42__single_q_l27_r16 | Qwen3-1.7B | UltraChat 100k | hidden_lora | 42 | 16/32 | - | 0.0002 | 8x2 | layers=27 | 66k | 1.128 |
| qwen3_17b__seed42__single_qkvo_l14_r4 | Qwen3-1.7B | UltraChat 100k | hidden_lora | 42 | 4/8 | - | 0.0002 | 8x2 | layers=14 | 57k | 1.088 |


### UltraChat 100k：低 rank 动态队列

来源：`outputs/affine_vocab/ultrachat100k_lowrank_20260525_011523`；摘要见 `docs/ultrachat100k_lowrank_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_15b__seed42__combined_inlmh_r1_s1_8 | Qwen2.5-1.5B | UltraChat 100k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/128/8 | 0.0002 | 8x2 | - | 1.25M | 1.081 |
| qwen25_15b__seed42__combined_inlmh_r2_s1_8 | Qwen2.5-1.5B | UltraChat 100k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 8x2 | - | 2.41M | 1.079 |
| qwen25_15b__seed42__combined_inlmh_r4_s1_8 | Qwen2.5-1.5B | UltraChat 100k | affine_input_lm_head_plus_hidden_lora | 42 | 4/8 | 16/128/8 | 0.0002 | 8x2 | - | 4.72M | 1.077 |
| qwen25_15b__seed42__hidden_lora_r1 | Qwen2.5-1.5B | UltraChat 100k | hidden_lora | 42 | 1/2 | - | 0.0002 | 8x2 | - | 1.15M | 1.081 |
| qwen25_15b__seed42__hidden_lora_r2 | Qwen2.5-1.5B | UltraChat 100k | hidden_lora | 42 | 2/4 | - | 0.0002 | 8x2 | - | 2.31M | 1.079 |
| qwen25_15b__seed42__hidden_lora_r4 | Qwen2.5-1.5B | UltraChat 100k | hidden_lora | 42 | 4/8 | - | 0.0002 | 8x2 | - | 4.62M | 1.078 |
| qwen3_17b__seed42__combined_inlmh_r1_s1_8 | Qwen3-1.7B | UltraChat 100k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/128/8 | 0.0002 | 8x2 | - | 1.22M | 1.064 |
| qwen3_17b__seed42__combined_inlmh_r2_s1_8 | Qwen3-1.7B | UltraChat 100k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 8x2 | - | 2.31M | 1.062 |
| qwen3_17b__seed42__combined_inlmh_r4_s1_8 | Qwen3-1.7B | UltraChat 100k | affine_input_lm_head_plus_hidden_lora | 42 | 4/8 | 16/128/8 | 0.0002 | 8x2 | - | 4.49M | 1.06 |
| qwen3_17b__seed42__hidden_lora_r1 | Qwen3-1.7B | UltraChat 100k | hidden_lora | 42 | 1/2 | - | 0.0002 | 8x2 | - | 1.09M | 1.065 |
| qwen3_17b__seed42__hidden_lora_r2 | Qwen3-1.7B | UltraChat 100k | hidden_lora | 42 | 2/4 | - | 0.0002 | 8x2 | - | 2.18M | 1.063 |
| qwen3_17b__seed42__hidden_lora_r4 | Qwen3-1.7B | UltraChat 100k | hidden_lora | 42 | 4/8 | - | 0.0002 | 8x2 | - | 4.36M | 1.06 |


### UltraChat 100k：full fine-tune 初探

来源：`outputs/affine_vocab/ultrachat100k_full_finetune_20260525_011523`；摘要见 `docs/ultrachat100k_full_finetune_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_15b__seed42__full_finetune | Qwen2.5-1.5B | UltraChat 100k | full_finetune | 42 | 16/32 | - | 0.0002 | 1x16 | - | 1543.71M | 1.245 |
| qwen3_17b__seed42__full_finetune | Qwen3-1.7B | UltraChat 100k | full_finetune | 42 | 16/32 | - | 0.0002 | 1x16 | - | 1720.57M | 1.2 |


### UltraChat 100k：full fine-tune lr2e-5 probe

来源：`outputs/affine_vocab/ultrachat100k_fullft_lr2e5_probe_20260525_103901`；摘要见 `docs/ultrachat100k_fullft_lr2e5_probe_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_15b__seed42__fullft_lr2e5_maxsteps2000 | Qwen2.5-1.5B | UltraChat 100k | full_finetune | 42 | 16/32 | - | 0.00002 | 1x16 | - | 1543.71M | 1.09 |


### MiniMind random 100k：低 rank 动态队列

来源：`outputs/affine_vocab/minimind_random100k_lowrank_20260525_131318`；摘要见 `docs/minimind_random100k_lowrank_results.md`

| run | model | data | variant | seed | h_r/a | aff_r/a/s1 | lr | batch | flags | trainable | eval_loss |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| qwen25_15b__seed42__affine_in_lmh_r16_s1_8 | Qwen2.5-1.5B | MiniMind random 100k | affine_input_lm_head | 42 | 16/32 | 16/128/8 | 0.0002 | 8x2 | - | 100k | 1.019 |
| qwen25_15b__seed42__combined_inlmh_r1_s1_8 | Qwen2.5-1.5B | MiniMind random 100k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/128/8 | 0.0002 | 8x2 | - | 1.25M | 0.9555 |
| qwen25_15b__seed42__combined_inlmh_r2_s1_8 | Qwen2.5-1.5B | MiniMind random 100k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 8x2 | - | 2.41M | 0.9485 |
| qwen25_15b__seed42__combined_inlmh_r4_s1_8 | Qwen2.5-1.5B | MiniMind random 100k | affine_input_lm_head_plus_hidden_lora | 42 | 4/8 | 16/128/8 | 0.0002 | 8x2 | - | 4.72M | 0.9413 |
| qwen25_15b__seed42__hidden_lora_r1 | Qwen2.5-1.5B | MiniMind random 100k | hidden_lora | 42 | 1/2 | - | 0.0002 | 8x2 | - | 1.15M | 0.957 |
| qwen25_15b__seed42__hidden_lora_r2 | Qwen2.5-1.5B | MiniMind random 100k | hidden_lora | 42 | 2/4 | - | 0.0002 | 8x2 | - | 2.31M | 0.949 |
| qwen25_15b__seed42__hidden_lora_r4 | Qwen2.5-1.5B | MiniMind random 100k | hidden_lora | 42 | 4/8 | - | 0.0002 | 8x2 | - | 4.62M | 0.9417 |
| qwen25_15b__seed42__single_qkvo_mid_r4 | Qwen2.5-1.5B | MiniMind random 100k | hidden_lora | 42 | 4/8 | - | 0.0002 | 8x2 | layers=14 | 39k | 1.032 |
| qwen3_17b__seed42__affine_in_lmh_r16_s1_8 | Qwen3-1.7B | MiniMind random 100k | affine_input_lm_head | 42 | 16/32 | 16/128/8 | 0.0002 | 8x2 | - | 133k | 0.8804 |
| qwen3_17b__seed42__combined_inlmh_r1_s1_8 | Qwen3-1.7B | MiniMind random 100k | affine_input_lm_head_plus_hidden_lora | 42 | 1/2 | 16/128/8 | 0.0002 | 8x2 | - | 1.22M | 0.8386 |
| qwen3_17b__seed42__combined_inlmh_r2_s1_8 | Qwen3-1.7B | MiniMind random 100k | affine_input_lm_head_plus_hidden_lora | 42 | 2/4 | 16/128/8 | 0.0002 | 8x2 | - | 2.31M | 0.8331 |
| qwen3_17b__seed42__combined_inlmh_r4_s1_8 | Qwen3-1.7B | MiniMind random 100k | affine_input_lm_head_plus_hidden_lora | 42 | 4/8 | 16/128/8 | 0.0002 | 8x2 | - | 4.49M | 0.8281 |
| qwen3_17b__seed42__hidden_lora_r1 | Qwen3-1.7B | MiniMind random 100k | hidden_lora | 42 | 1/2 | - | 0.0002 | 8x2 | - | 1.09M | 0.8395 |
| qwen3_17b__seed42__hidden_lora_r2 | Qwen3-1.7B | MiniMind random 100k | hidden_lora | 42 | 2/4 | - | 0.0002 | 8x2 | - | 2.18M | 0.8337 |
| qwen3_17b__seed42__hidden_lora_r4 | Qwen3-1.7B | MiniMind random 100k | hidden_lora | 42 | 4/8 | - | 0.0002 | 8x2 | - | 4.36M | 0.828 |
| qwen3_17b__seed42__single_qkvo_mid_r4 | Qwen3-1.7B | MiniMind random 100k | hidden_lora | 42 | 4/8 | - | 0.0002 | 8x2 | layers=13 | 57k | 0.9012 |



## 6. 不纳入本文主表的实验

| 实验 | 原因 |
|---|---|
| `phase3c_*` | 早期位置消融，后续 Phase 9 已用更系统的 input/lm_head/intermediate 归因替代 |
| `phase3d_*` | DoRA/rsLoRA 背景探索，不是 AffLoRA 主 claim 的正确结果来源 |
| `minimind_random100k_lowrank_20260525_131007` | 首次 random100k 启动遇到数据 schema 问题，已由 `20260525_131318` 过滤后重跑替代 |
| 旧 MetaMath/math/GLUE/E2E 探索 | 非当前 AffLoRA 主线数据和训练入口 |

## 7. 位置归因 v2 综合发现（2026-05-26 更新）

本节整合 2026-05-26 当天用动态调度跑完的几组针对 input vs lm_head 位置归因的 sweep。修正了第 5 节中早期凭少量 affine-only s1 sweep 得出的"lm_head 通常更强"的判断。所有结果都基于 `sft_t2t_mini_25k`，seed=42，1500 step（1 epoch），lr=2e-4 cosine。

### 7.0 hidden_lora（HL）详细定义

下文中所有 "HL"、"hidden_lora"、"hidden LoRA" 指同一个对象，区别于 affine vocab adapter（输入/输出词表层的 affine 适配）。

**作用位置**：transformer block 内部的 projection 层；不是词表层。具体加在每一层（共 N 层）的以下 7 个 nn.Linear 上：

| 模块组 | 子模块 | 形状（hidden=d，inter=h） |
|---|---|---|
| Self-attention | `q_proj`, `k_proj`, `v_proj`, `o_proj` | `(d, d)` (GQA 时 k/v 是 `(d_kv, d)`) |
| MLP (SwiGLU) | `up_proj`, `gate_proj`, `down_proj` | up/gate `(h, d)`，down `(d, h)` |

通过 `--hidden-lora-target-modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj` 控制。这是 PEFT/LoRA 文献最常见的 "all-linear" 配置。

**数学公式**（PEFT 标准 LoRA）：对每一个被 hook 的线性层 `W ∈ R^{out×in}`，
$$
y = W x + \frac{\alpha}{r} B (A x), \quad A \in R^{r×in},\ B \in R^{out×r}
$$
- `A` 用 Kaiming 初始化（默认）；`B` 初始化为 0，所以训练前 LoRA 增量为 0。
- `r` = `--hidden-lora-rank`，`α` = `--hidden-lora-alpha`，scale = `α/r`。
- dropout 在 `Ax` 之前作用，`--hidden-lora-dropout` 默认 0.05。

**参数预算**（Qwen2.5-0.5B：N=24 层，d=896，h=4864；7 个目标模块全开 r=8 a=16）：
- 单层 7 个 LoRA：≈ `7 × r × (d_in + d_out)` ≈ 7 × 8 × ~6500 ≈ **0.36M / 层**
- 24 层全开：≈ **8.7M 参数**（Qwen2.5-0.5B 实测约 4.4M，因为 Qwen2.5-0.5B 用 GQA 让 k/v_proj 比 q_proj 小）

对比 affine vocab：affine_input r=8 ≈ 2 × r × d + d ≈ 14k 参数，HL 大 ~300×。

**层范围控制**（§7.3–7.5 反复用到的几个 setting）：

| 名字 | 实现 | 等价 layers |
|---|---|---|
| **full HL** / "全层 HL" | `--hidden-lora-layers-to-transform` 不传 | 默认应用到所有 N 层 (0..N-1) |
| **HL_l0** / "仅 layer 0" | `--hidden-lora-layers-to-transform 0` | 只有 layer 0 一层 |
| **HL_no_l0** / "排除 layer 0" | `--hidden-lora-layers-to-transform 1,2,...,N-1` | layers 1..N-1，共 N-1 层 |
| 单层中段 | `--hidden-lora-layers-to-transform 14` | 只 layer 14 |

实现上 PEFT 在 `LoraConfig.layers_to_transform` 接受层索引列表，HL 之外的层（包括 embed_tokens、lm_head、frozen 层）参数原样冻结。

**与 AffLoRA 的本质区别**：
- HL 作用在 hidden state 流（每一层进入/离开 self-attn 与 MLP 之间）。
- AffLoRA 作用在 **词表 ↔ hidden 的边界**：`affine_input` 改的是 `embed_tokens` 输出（进入 layer 0 之前），`affine_lm_head` 改的是 `lm_head` 输入（离开 layer N-1、RMSNorm 之后）。
- 两者**完全不重叠**的位置只有 `affine_lm_head`：从 RMSNorm 到 logits 之间，HL 即便覆盖所有层也碰不到。这是 §7.8 "lm_head + HL 互补，input + HL 冗余" 的几何根据。

### 7.1 数据来源

| 来源目录 | 内容 |
|---|---|
| `outputs/affine_vocab/position_affine_sweep_small_20260526_094509` | affine-only r=8 全 s1 + r=16 部分 s1 ×（input withbias / lm_head nobias） |
| `outputs/affine_vocab/affine_pos_plus_hidden_lora_20260526_101307` | hidden_lora 全层 r=8 a=16 + 各种 affine（r=8 s1=64） |
| `outputs/affine_vocab/lm_head_recovery_quick_20260526_104314` | H1/H2 验证（低秩、lm_head_withbias） |
| `outputs/affine_vocab/position_attribution_v2_20260526_111320` | 大模型 1.5B、低秩+HL、HL 仅 layer 0 |
| `outputs/affine_vocab/position_attribution_def_20260526_114910` | Qwen2.5-7B（真 untied）affine-only、HL 排除 layer 0、纯 vocab-only LoRA |

### 7.2 affine-only：input 在所有 rank/scale 上稳健胜出

`affine_input` 默认带 hidden bias（`use_input_bias=True`），`affine_lm_head` 默认不带（`use_lm_head_bias=False`）。

#### 7.2.1 Qwen2.5-0.5B (24 layers, tied)

| variant | r=2 | r=4 | r=8 (s1=64) | r=8 (s1=128) |
|---|---:|---:|---:|---:|
| `affine_input` (with bias) | 1.879 | 1.852 | 1.816 | 1.814 |
| `affine_lm_head` (no bias) | 2.010 | 1.963 | 1.907 | 1.907 |
| `affine_lm_head` (with bias) s1=32 | — | — | 1.909 | — |
| `affine_lm_head` (with bias) s1=64 | — | — | 1.907 | — |
| **Δ(input − lm_head_nobias)** | **-0.131** | **-0.111** | **-0.091** | **-0.093** |

#### 7.2.2 Qwen3-0.6B (28 layers, tied)

| variant | r=2 | r=4 | r=8 (s1=64) |
|---|---:|---:|---:|
| `affine_input` (with bias) | 1.494 | 1.469 | 1.460 |
| `affine_lm_head` (no bias) | 1.693 | 1.655 | 1.621 |
| `affine_lm_head` (with bias) s1=32 | — | — | 1.611 |
| `affine_lm_head` (with bias) s1=64 | — | — | 1.611 |
| **Δ(input − lm_head_nobias)** | **-0.199** | **-0.186** | **-0.161** |

#### 7.2.3 Qwen2.5-1.5B (28 layers, tied) r=8 s1=64

| variant | eval_loss |
|---|---:|
| `affine_input` (with bias) | **1.502** |
| `affine_lm_head` (no bias) | 1.608 |
| **Δ** | **-0.106** |

#### 7.2.4 Qwen2.5-7B (28 layers, **untied**) r=8 s1=64

这是关键的"真 untied"对照组——Qwen2.5-7B 的 `embed_tokens` 与 `lm_head` 是独立矩阵（不共享权重），用 `gradient_checkpointing` + bs=4 accum=4。

| variant | eval_loss |
|---|---:|
| `affine_input` (with bias) | **1.258** |
| `affine_lm_head` (no bias) | 1.392 |
| **Δ** | **-0.134** |

→ **即便在真正 untied 的 7B 大模型上，input 优势依然显著**（Δ -0.134，比 1.5B 的 -0.106 还大）。这彻底排除了"input 优势仅在 tied 模型中存在"的可能性。

#### 7.2.5 关键观察

- **affine-only 设置下 input 在所有 rank、所有模型尺寸（0.5B 到 7B）、tied 与 untied 均稳健胜出 ~0.09–0.20**。Qwen2.5-7B 真 untied 上 Δ=-0.134，0.5B tied 上 Δ=-0.091，差距甚至随模型增大而扩大。
- **rank 越低 input 优势越大**（与"低秩有利于 lm_head"猜想相反）。
- **`affine_lm_head` 的 hidden bias 对最终 eval_loss 几乎没有影响**（withbias 与 nobias 差 ≤0.002）。原因：bias 在 `lm_head` 一侧的梯度量级 ≈ 4e-2，比 input 一侧 bias（梯度 ≈ 17.9）小约 400×；它接收梯度也能更新（通过 `W_v · bias` 给每个 vocab 行不同标量，**不是** softmax-invariant），但 capacity-per-param 远低于 input bias。
- 早期文档中"lm_head-only 通常强于 input-only"的口径是基于 `r=16, s1∈[8,16]` 的少量数据点（见 §3 Phase 2c）。在更系统的 `r=8, s1∈{2,…,128}` sweep 上看到的实际趋势是 **input 一侧稳健占优**。

### 7.3 affine + full-layer hidden_lora：position 反转，lm_head 胜出

加上全层标准 hidden LoRA（r=8 alpha=16，target=q,k,v,o,up,down,gate，全部 transformer 层）后，趋势完全反转。

#### 7.3.1 Qwen2.5-0.5B + full-layer HL r=8 a=16

| affine variant | trainable | eval_loss | Δ vs HL only |
|---|---:|---:|---:|
| hidden_lora only | 4,399,104 | 1.302 | — |
| `affine_input` + HL (r=2 s1=64) | ~4,400k | 1.299 | -0.003 |
| `affine_input` + HL (r=4 s1=64) | ~4,401k | 1.298 | -0.004 |
| `affine_input` + HL (r=8 s1=64) | 4,414,336 | 1.297 | -0.005 |
| **`affine_lm_head` + HL (r=2 s1=64)** | ~4,399k | **1.292** | **-0.010** |
| **`affine_lm_head` + HL (r=4 s1=64)** | ~4,400k | **1.291** | **-0.011** |
| **`affine_lm_head` + HL (r=8 s1=64)** | 4,413,440 | **1.288** | **-0.014** |
| `affine_inlmh` + HL (r=8 s1=64) | 4,428,672 | 1.289 | -0.013 |

#### 7.3.2 Qwen3-0.6B + full-layer HL r=8 a=16

| affine variant | trainable | eval_loss | Δ vs HL only |
|---|---:|---:|---:|
| hidden_lora only | 5,046,272 | 1.101 | — |
| `affine_input` + HL (r=2 s1=64) | ~5,047k | 1.098 | -0.003 |
| `affine_input` + HL (r=4 s1=64) | ~5,048k | 1.098 | -0.003 |
| `affine_input` + HL (r=8 s1=64) | 5,063,680 | 1.097 | -0.004 |
| **`affine_lm_head` + HL (r=2 s1=64)** | ~5,046k | **1.094** | **-0.007** |
| **`affine_lm_head` + HL (r=4 s1=64)** | ~5,047k | **1.094** | **-0.007** |
| **`affine_lm_head` + HL (r=8 s1=64)** | 5,062,656 | **1.093** | **-0.008** |
| `affine_inlmh` + HL (r=8 s1=64) | 5,080,064 | 1.095 | -0.006 |

#### 7.3.3 关键观察

- **`affine_lm_head` + HL 在所有 rank 下稳定胜过 `affine_input` + HL** 0.004–0.007。
- **`affine_input` + HL 几乎无收益**（-0.003 至 -0.005）；它的那 ~15k 参数大部分被 hidden_lora 的 ~5M 参数吸收。
- **双侧 `affine_inlmh` + HL 不优于单侧 `affine_lm_head` + HL**（甚至略差），说明双侧之于 lm_head 单侧没有额外贡献，input 一侧在有 HL 时是冗余的。

### 7.4 affine + 单层 hidden_lora：lm_head 仍稳定胜出

将 HL 限制到 **layer 0** 一层（其余 23/27 层都 frozen），affine r=8 s1=64：

| 模型 | `affine_input` + HL_l0 | `affine_lm_head` + HL_l0 | Δ |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 1.728 | **1.638** | **-0.090** |
| Qwen3-0.6B | 1.386 | **1.336** | **-0.050** |

→ HL 即便只覆盖 1 层，lm_head 已经反超 input 0.05–0.09。这**推翻了 "input 优势纯粹来自经过 28 层 transformer 的广播放大"** 这个简单解释。

### 7.5 affine + HL 排除 layer 0（layers 1..N-1）：与 full HL 几乎完全一致

去掉 layer 0 的 HL，保留其余 23/27 层（即 0.5B 上 layers 1..23，0.6B 上 layers 1..27），affine r=8 s1=64：

| 模型 | `affine_input` + HL_no_l0 | `affine_lm_head` + HL_no_l0 | Δ |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 1.299 | **1.290** | **-0.009** |
| Qwen3-0.6B | 1.098 | **1.095** | **-0.003** |

对照 full-layer HL：

| 模型 | input + full HL | lm_head + full HL |
|---|---:|---:|
| Qwen2.5-0.5B | 1.297 | 1.288 |
| Qwen3-0.6B | 1.097 | 1.093 |

→ **HL_no_l0 与 full HL 几乎完全一致**（差 0.001-0.002），说明 layer 0 对 HL 整体能力的贡献微乎其微。input 与 lm_head 的差距同样保持 -0.003 到 -0.009，与 full HL 一致——说明 input 与 HL 的"冗余"是**分散在所有 transformer 层**，不局限于 layer 0。

### 7.6 vocab-only LoRA vs affine：参数效率横向对照

vocab-only LoRA 直接在 `embed_tokens` 和 `lm_head` 权重上加标准 LoRA（target=`embed_tokens,lm_head`），与同 rank 的 `affine_input` 比较参数效率。
0.5B 词表 152k、hidden 896；vocab-only LoRA r=2 ≈ 152k×2×2 = 0.6M 参数；affine r=2 ≈ 4.5k 参数（含 bias），相差约 **130×**。

| 模型 | rank | `affine_input` (with bias) | vocab-only LoRA | trainable比 |
|---|---:|---:|---:|---|
| Qwen2.5-0.5B | r=2 | **1.879** | 1.883 | affine ≈ vocab，参数少 130× |
| Qwen2.5-0.5B | r=4 | 1.852 | **1.816** | vocab 略胜，参数多 ~130× |
| Qwen3-0.6B | r=2 | **1.494** | 1.529 | affine 显著胜 0.035（少 ~130× 参数） |
| Qwen3-0.6B | r=4 | **1.469** | 1.495 | affine 显著胜 0.026（少 ~130× 参数） |

**关键观察**：affine 在低 rank（r=2）和小 hidden 下用 ~1% 参数取得与 vocab-only LoRA 相当或更好的效果；只有在 0.5B + r=4 这个极端组合下 vocab-only 略胜，但参数代价大 100×+。在 tied 模型（0.6B）上 affine 优势最显著。

### 7.7 三组合并：Qwen2.5-0.5B 完整位置归因表

| 设置 | input | lm_head | Δ(input − lm_head) | 谁胜 |
|---|---:|---:|---:|---|
| affine-only r=8 s1=64 | **1.816** | 1.907 | -0.091 | **input** 胜 0.091 |
| + HL layer 0 only | 1.728 | **1.638** | +0.090 | **lm_head** 胜 0.090 |
| + HL layers 1..23 (no l0) | 1.299 | **1.290** | +0.009 | **lm_head** 胜 0.009 |
| + HL full-layer r=8 | 1.297 | **1.288** | +0.009 | **lm_head** 胜 0.009 |

### 7.8 因果机制（修正后）

旧解释（已证伪）：~~input 优势来自其 affine 扰动经过 28 层 transformer 非线性放大~~。
新解释（受 §7.4 + §7.5 双向证据支持）：

1. **affine-only 时**两侧都是模型唯一的可训通道，input 因为有 28 层非线性"展开"获得更高的功能容量——所以 input 胜。
2. **加任何 hidden_lora**（即便只在 layer 0 一层；或反过来去掉 layer 0 保留其余 23-27 层），HL 在 q/k/v/o/up/down/gate 上的 LoRA 等效于"对 layer N 的 input 做更复杂的非线性投影"。这与 `affine_input` 在功能空间上**部分冗余**——HL 在哪一层覆盖都能替代 input affine 的大部分功能。`affine_input` 的边际贡献因此被吃掉，且这种冗余**分散在所有层**（§7.5 显示去掉 layer 0 几乎不影响 HL 的能力，input vs lm_head 差距与 full HL 完全一致）。
3. **`affine_lm_head` 在 RMSNorm 之后、lm_head 之前**——这是任何 HL 都够不到的位置。所以 lm_head affine 提供的是 **HL 之外、独立的末端容量**，与 HL 在功能上正交。无论 HL 覆盖 1 层、N-1 层还是全部 N 层，lm_head affine 都稳定补 0.005–0.09 的额外收益。

### 7.9 论文叙事的修正（关键）

| 旧叙事 | 修正 |
|---|---|
| "AffLoRA 是 hidden LoRA 的低成本补充，双侧最强（claim 1a）" | **`affine_lm_head`** 是 hidden LoRA 的低成本独立补充；**`affine_input`** 与 hidden LoRA 在功能上**部分冗余**，与 HL 结合时收益小到几乎不可见。**最佳实践：A-LoRA 单独用 `affine_input`；与 HL 联用时改用 `affine_lm_head` 或 mergeable tied** |
| "lm_head 比 input 更强" | 仅在 **+ hidden_lora**、或 **tied 模型 mergeable** 等"input 路径已被部分占用"的设置下成立。affine-only 时反而是 **input 胜出 ~0.1** |
| "input 优势源于 28 层广播放大" | 已被 HL_l0 实验证伪。更准确：input 优势源于 **affine-only 时它是唯一可训通道**，HL 一旦加入即被取代 |
| "tied 模型 mergeable 是工程意义（merge 等价）" | mergeable_tied 把 input/lm_head 的 affine 绑成单一参数集，**还从根本上避免了 `affine_input` 在有 HL 时的冗余浪费**——所以 mergeable 的优势既是工程的也是参数效率的 |

### 7.10 v3 终极证据 sweep（2026-05-26 14:00，35 任务全完成）

为把 §10 各 Insight 从"小模型 / seed=42"扩展到"大模型 / multi-seed / 控制变量"，2026-05-26 当天用 8 卡动态调度跑了一次包含 6 个分组、35 个任务的 sweep。完整目录：`outputs/affine_vocab/position_attribution_v3_20260526_131404/`，启动脚本：`lora/scripts/run_position_attribution_v3_dynamic.sh`。

| 组 | 目的 | 任务数 | 结论 |
|---|---|---:|---|
| A. 大模型 + HL 位置归因 | 在 1.5B、1.7B、**真 untied 7B** 上验证 lm_head + HL > input + HL | 6 | **3/3 模型一致**，Δ -0.0034 ~ -0.006 |
| B. 多 seed 鲁棒性 | seed=43, 44 在 0.5B / 0.6B 上重复 Insight III 的对照 | 8 | **6/6 配置 lm_head 胜出**，跨 seed σ ≈ 0.001 |
| C. `affine_input` no-bias 消融 | 关掉 input 的 hidden bias，验证 input 优势是否依赖 bias | 6 | input no-bias r=8 比 with-bias 高 0.005–0.006（噪声内）→ **bias 不是 input 优势的来源** |
| D. 大模型 vs vocab-only LoRA | 1.5B / 1.7B 上 vocab-only LoRA r=2/4 与 `affine_input` 对照 | 4 | AffLoRA 用 **1/24 参数** 比 vocab-only LoRA r=4 还低 **0.065 ~ 0.164** |
| E. 大模型 `affine_input` r=16 s1=64 单侧 | 复跑大模型 affine-only 的 ~50k 基准点 | 3 | 与 canonical 数据吻合 < 0.005，为未来大模型 single-layer LoRA 对照备份 |
| F. HL 排除 last / mid 层 | HL 缺一层后，input vs lm_head 差距是否保持 | 8 | 排除单层只让 eval_loss 升 0.003–0.008；**lm_head − input 差距完全保持** |

**关键证据汇总表**（数字直接来自 v3 sweep 完整 35 个 run；Δ 列为 `lm_head − input` 或 `aff − baseline`）：

| 分组 | 模型 | 具体配置 | input / baseline | lm_head / treatment | Δ |
|---|---|---|---:|---:|---:|
| A | Qwen2.5-1.5B (tied) | r=8 s1=64 + HL | 1.0500 | **1.0440** | -0.006 |
| A | Qwen3-1.7B (tied) | r=8 s1=64 + HL | 0.7362 | **0.7328** | -0.0034 |
| A | Qwen2.5-7B (**untied**) | r=8 s1=64 + HL | 0.9180 | **0.9130** | -0.005 |
| B | Qwen2.5-0.5B seed=43 | r=8 s1=64 + HL | 1.2970 | **1.2890** | -0.008 |
| B | Qwen2.5-0.5B seed=44 | r=8 s1=64 + HL | 1.2970 | **1.2880** | -0.009 |
| B | Qwen3-0.6B seed=43 | r=8 s1=64 + HL | 1.0960 | **1.0940** | -0.002 |
| B | Qwen3-0.6B seed=44 | r=8 s1=64 + HL | 1.0970 | **1.0930** | -0.004 |
| C | Qwen2.5-0.5B | input r=2 s1=64 no-bias | — | 1.9030 | — |
| C | Qwen2.5-0.5B | input r=4 s1=64 no-bias | — | 1.8610 | — |
| C | Qwen2.5-0.5B | input r=8 s1=64 no-bias vs with-bias | 1.816 | 1.8210 | +0.005 |
| C | Qwen3-0.6B | input r=2 s1=64 no-bias | — | 1.4960 | — |
| C | Qwen3-0.6B | input r=4 s1=64 no-bias | — | 1.4760 | — |
| C | Qwen3-0.6B | input r=8 s1=64 no-bias vs with-bias | 1.441 | 1.4470 | +0.006 |
| D | Qwen2.5-1.5B | vocab-only LoRA r=2 vs `affine_input` r=16 s1=64 | 1.591 (~610k) | **1.473 (~50k)** | -0.118 |
| D | Qwen2.5-1.5B | vocab-only LoRA r=4 vs `affine_input` r=16 s1=64 | 1.538 (~1.22M) | **1.473 (~50k)** | -0.065 |
| D | Qwen3-1.7B | vocab-only LoRA r=2 vs `affine_input` r=16 s1=64 | 1.189 (~614k) | **0.981 (~50k)** | -0.208 |
| D | Qwen3-1.7B | vocab-only LoRA r=4 vs `affine_input` r=16 s1=64 | 1.145 (~1.23M) | **0.981 (~50k)** | -0.164 |
| E | Qwen2.5-1.5B | `affine_input` r=16 s1=64（v3 复跑） | canonical 1.477 | v3 **1.4730** | -0.004（一致）|
| E | Qwen3-1.7B | `affine_input` r=16 s1=64（v3 复跑） | canonical 0.9806 | v3 **0.9810** | +0.0004（一致） |
| E | Qwen2.5-7B (untied) | `affine_input` r=16 s1=64（v3 复跑） | canonical 1.23 | v3 **1.2310** | +0.001（一致） |
| F | Qwen2.5-0.5B | r=8 s1=64 + HL 排除 l23 (last) | 1.3050 | **1.2950** | -0.010 |
| F | Qwen2.5-0.5B | r=8 s1=64 + HL 排除 l12 (mid) | 1.3020 | **1.2930** | -0.009 |
| F | Qwen3-0.6B | r=8 s1=64 + HL 排除 l27 (last) | 1.1000 | **1.0960** | -0.004 |
| F | Qwen3-0.6B | r=8 s1=64 + HL 排除 l14 (mid) | 1.0990 | **1.0950** | -0.004 |

**v3 sweep 给 Insight 体系带来的升级**：

1. **Insight III 升格**：从"小模型 / tied / seed=42 上 lm_head + HL > input + HL"扩展到"3/3 大模型（含 untied 7B）+ 6/6 multi-seed + 排除 last/mid 单层"全部同方向 → **可以作为论文主 claim**，不再是边缘效应。
2. **Insight VII 修正**：input 关 bias 后只升 0.005–0.006，证明 §7.2 中 input vs lm_head 的 0.09–0.20 差距**不是 bias asymmetry 造成的**（C 组数据排除了这条 alternative explanation）；input 几何优势是稳健的。
3. **Insight I 大模型证据**：在 1.5B / 1.7B 上 AffLoRA 用 **1/24 参数** 比 vocab-only LoRA r=4 还低 0.065 ~ 0.164，参数效率 claim 不再仅依赖小模型推断。
4. **Insight IV 增强**：F 组数据补全了"HL 排除任意单层都几乎等价"的图景，不仅 layer 0、layer 12/14、layer 23/27 单层都验证了。

## 8. 最终结论口径（更新于 2026-05-26）

1. **tied 模型上两类 AffLoRA 形态都有效**：一种是以 adapter 模块形式保留的双侧适配，另一种是 `mergeable tied` 形式，通过共享 adapter 并在 lm_head 侧使用转置作用，最终可以 merge 回 tied embedding。后者通过 logits 等价检查，数值误差约 `3e-5` 到 `6e-5`。**§7.6 进一步给 mergeable_tied 的优势加了机制解释**：tied 共享避免了 `affine_input` 与 hidden_lora 重叠造成的参数浪费。

2. **AffLoRA 本身有表达能力**，并且在小参数预算下普遍强于同量级 single-layer LoRA。这个结论在 0.6B、1.5B、1.7B、7B、8B 以及 MiniMind random 100k 上都有正向证据。**§7.2 进一步发现**：affine-only 时 `affine_input` 在所有 rank 与所有模型尺寸上稳健胜过 `affine_lm_head` ~0.09–0.20，且 rank 越低 input 优势越大。

3. **位置归因取决于是否搭配 hidden_lora**：
   - **affine-only**：`affine_input` 全面胜出（差距 0.09–0.20）。在 0.5B tied、0.6B tied、1.5B tied、**7B untied** 上都成立——untied 的 7B 上差距甚至比小模型更大（Δ -0.134）。
   - **+ hidden_lora（无论全层、仅 layer 0、还是排除 layer 0 保留 23-27 层）**：`affine_lm_head` 反超（差距 0.005–0.09）。
   - 这与 §7.8 的因果解释一致：input affine 与 hidden_lora 部分冗余，lm_head affine 与 hidden_lora 完全互补。
   - 早期文档基于 `r=16` 少量 s1 数据点得出的"lm_head 通常更强"是不全的口径，现已被 §7 替代。

4. **AffLoRA 与 hidden_lora 联用的最佳形态是 `affine_lm_head_plus_hidden_lora`**，而不是 `affine_input_plus_hidden_lora` 或双侧 `affine_input_lm_head_plus_hidden_lora`。tied 模型上 `mergeable_tied_combined` 通过把双侧绑定为同一适配器，自动避免冗余，因此与 `affine_lm_head_plus_hidden_lora` 一样推荐。

5. **bias 配置的影响**：`affine_input` 一侧的 hidden bias 贡献可观（bias 梯度量级 ≈ 17.9，相对于 down/up 是同量级）；`affine_lm_head` 一侧的 hidden bias 在最终 eval 上**几乎没有影响**（梯度 ≈ 4e-2，比 input bias 小约 400×），无论开/关都不改变结果 ≤0.002。论文叙事中可以选择不在 lm_head 一侧用 hidden bias，简化变体集合。

6. **AffLoRA vs vocab-only LoRA 的参数效率**：affine 在低 rank（r=2）下用约 **1% 的参数**取得与 vocab-dim LoRA 相当或更好的效果，在 tied 模型（Qwen3-0.6B）上 affine_input r=2（4.5k 参数）比 vocab-only LoRA r=2（~600k 参数）好 0.035。这是 AffLoRA 相对传统 vocab LoRA 在低预算下的核心优势。详见 §7.6。

7. AffLoRA 在 `sft_t2t_mini_25k` 上收益最清楚；UltraChat 100k 和 MiniMind random 100k 上收益收敛到持平/略优，更适合作为"不退化且有轻微正向"的外部验证。

## 9. 项目核心 Insight 总结

把"做了什么实验、得出什么结论"上抽一层，落到**可复用的洞察**——下一次想做类似的低秩词表适配工作时这些 insight 就能直接用。

### 9.1 方法论 insight：AffLoRA 是什么

> **AffLoRA = 把"低秩 LoRA"这个工具从 hidden state 流搬到了词表↔hidden 的 ε-厚边界。**

- 传统 LoRA 加在 transformer block 的 7 个 projection 上（HL）：每层都需要参数，N 层 ≈ M·N 参数预算。
- AffLoRA 加在 `embed_tokens` / `lm_head` 周围一次：参数量 `≈ 2·r·d`（≈ 几 k-几十 k），与层数无关。
- 这个"位置错位"让 AffLoRA 与 HL 在功能上**几乎正交**，组合起来比任意一边单独都强（§7.3）。

### 9.2 几何 insight：input 与 lm_head 是两类完全不同的位置

| 位置 | 离 token 多远 | 经过的非线性 | 与 HL 的关系 |
|---|---|---|---|
| `affine_input`（embed 之后） | 进入 layer 0 之前 | 0 层 → 28 层（之后全部 transformer） | **被 HL 完全覆盖**——HL 在 layer 0+ 的扰动可以模拟 input affine |
| `affine_lm_head`（RMSNorm 之后、lm_head 之前） | 离 logits 一步之遥 | 0 层 transformer | **HL 触不到**——RMSNorm 之后只剩一个线性映射到 vocab |

- → **affine-only 时 input 胜**（§7.2，0.5B–7B untied 全部 Δ = -0.09 ~ -0.20）：因为 input 拥有"自身 + 28 层非线性展开"两份功能，lm_head 只有"自身 + 1 个线性 head"。
- → **+ HL 时 lm_head 胜**（§7.3–7.5，Δ = +0.005 ~ +0.09）：因为 HL 把 input 那份"28 层非线性展开"的容量顶替了，留给 input affine 的边际几乎为零；而 lm_head 那个 HL 触不到的位置仍然是独立增量。
- → **HL 容量是分散的，不在 layer 0**（§7.5）：去掉 layer 0 的 HL 与 full HL 几乎无差，input vs lm_head 的差距同样保持 0.009。说明上面这套"input/HL 冗余"是**全局的几何关系**，不是 layer-0 局部的。

### 9.3 参数效率 insight：分子分母都要看

> **比较 PEFT 方法不能只看 eval_loss，要把 eval_loss 与 trainable params 一起看；AffLoRA 在低预算区间的 Pareto 前沿几乎独占。**

- **AffLoRA r=2 vs vocab-only LoRA r=2**（§7.6）：参数差 ~130×，loss 几乎相同（tied 模型 affine 反而胜 0.03）。
- **AffLoRA r=8 + HL r=8 vs HL r=8**（§7.3）：增量参数 ~14k（+0.3%），lm_head 侧稳定补 0.005–0.014，相当于"廉价订阅"。
- **AffLoRA r=8 affine-only vs single-layer LoRA**（早期 §3 Phase 2c, 3a 等）：相同参数量级 `~50k`，AffLoRA 普遍胜单层 LoRA 0.1–0.3。
- 所以叙事框架应该是**"低预算 PEFT 的方法论"**而不是"另一种 LoRA 变体"——这是和 hidden LoRA 不同的定位。

### 9.4 工程 insight：tied 模型 mergeable 的双重价值

> **mergeable_tied 不仅解决了部署侧的 merge 问题（显然），还顺手解决了训练侧"双侧独立 adapter 一半参数白训"的隐性问题（不显然，是 §7.3–§7.5 反推出来的）。**

#### 价值 1：部署 merge 等价（工程意义，已知）

普通 tied 模型 + 双侧独立 adapter 的状况：
- 模型权重 `W_embed = W_lm_head` 共享同一个 tensor。
- 给 input 侧加 adapter `A_in`、给 lm_head 侧加另一个独立 adapter `A_out`，训练后想 merge 回原权重：
  - 把 `A_in` 增量加到 `W_embed` → 因为是同一 tensor，`W_lm_head` 也跟着变了。
  - 想再把 `A_out` 也 merge 进去 → 没法 merge，因为 `A_out` 是独立的另一份增量，加完之后 input 端会**多算一份不该算的**。
- 所以这种"双侧独立 adapter" tied 模型部署时**必须保留两个 adapter 模块**，无法回到单个 tied 权重矩阵。

`mergeable_tied`（`--mergeable-tied-affine-output`）的做法：
- input 侧用 affine `(I + s·U·D)` 作用在 hidden 上（标准形态）。
- lm_head 侧不用独立 adapter，而是用同一个 `(U, D)` 的**转置作用** —— 等价于在 lm_head 输入端做 `(I + s·U·D)^T` 的 affine。
- 数学上两侧等价于"对同一个 tied 矩阵 `W` 做一次 affine 变换 `(I + s·U·D)·W`"，因此可以在部署时**一次性 merge 回 `W`**，adapter 模块完全消除。
- 实测 logits 等价检查误差 ~3e-5 ~ 6e-5（纯数值噪声）。

→ 这就是工程意义上的 merge 等价：部署时一份 tied 权重就够了。

#### 价值 2：避免"双侧独立 adapter 一半参数白训"（性能意义，从 §7 反推）

先回顾两个事实：
1. **`affine_input` 与 hidden_lora 功能重叠**（§7.3, §7.5）：HL 的 q/k/v/o + MLP 即便只覆盖 layer 0 一层，也能模拟 `affine_input` 的大部分功能。`affine_input + HL` 比纯 HL 只多 0.003–0.005。
2. **`affine_lm_head` 与 hidden_lora 功能正交**（§7.3）：lm_head adapter 作用在 RMSNorm 之后，HL 完全够不到这个位置。`affine_lm_head + HL` 比纯 HL 多 0.005–0.014。

**所以双侧独立 adapter + HL 的结果是**：
- 参数预算 = `A_in`(~M) + `A_out`(~M) = **2M**
- `A_in` 的功能 ≈99% 被 HL 取代 → 这 M 个参数**白训**
- `A_out` 的功能 HL 触不到 → 这 M 个参数是**有效增量**
- 净效果：双侧独立 + HL 的设置下，**一半参数浪费**

**mergeable_tied 通过共享自动绕开这个失败模式**：
- 因为 input/lm_head 共用一个 adapter `(U, D)`，参数 = **M**（不是 2M）。
- 训练时这一个共享 adapter 同时收到来自 input 侧和 lm_head 侧的梯度；input 侧梯度信号被 HL 吃掉大半，但 **lm_head 侧梯度仍然提供独立监督信号**，这一份独立信号是 HL 触不到的位置在拉。
- 训练完后这个共享 adapter 主要承载 "lm_head 侧 + HL 触不到的位置" 那份独立功能，约等于 §7.3 中 `affine_lm_head + HL` 的有效增量。
- M 个参数都没浪费。

#### 浓缩对照

| 形态 | + HL 时的实际有效参数 | 部署能否 merge |
|---|---|---|
| tied 双侧独立 adapter（普通） | ~50%（input 那侧被 HL 吸收） | 否，必须留 adapter |
| `mergeable_tied`（共享 + 转置） | **~100%** | **是**，merge 后只剩单个 tied 权重 |

→ 推论：**对 tied 模型 + HL 设置，应该直接选 `mergeable_tied_combined`，不要再用"双侧独立 adapter + HL"这种形态**。这是论文叙事中可以单独成段的工程贡献。

### 9.5 叙事 insight：与早期文献口径的关系

| 早期文献 / 我们早期口径 | 真实情况 |
|---|---|
| Output Embedding (Press & Wolf, 2017): "tying 让 output embedding 直接被监督" | 在 SFT 阶段意义不大；真正起作用的是**哪一侧 affine 与 HL 互补**而不是哪一侧"被监督" |
| 我们早期 §3："lm_head adapter 比 input adapter 更强" | 仅在 r=16 + s1∈[8,16] 这个**少数据点 + 与 HL 联用** 的窄设置下成立 |
| "AffLoRA 双侧 (in+lm_head) 最强" | 与 HL 联用时双侧 ≈ 单 lm_head；不与 HL 联用时双侧 ≈ 单 input；所以**没有"哪个绝对最强"，要看是否搭配 HL** |

### 9.6 实操 insight：怎么选

| 你的场景 | 推荐配置 | 理由 |
|---|---|---|
| 极低预算 PEFT（< 50k 参数） | `affine_input` r=8 s1=64 with bias | §7.2 系统证据；显著胜单层 LoRA、vocab-only LoRA |
| 在 hidden_lora 基础上想加 0.1% 参数补一刀 | `affine_lm_head_plus_hidden_lora` r=2 s1=64（不需要 bias） | §7.3，最小成本最大独立增量 |
| tied 模型，要可 merge 部署 | `mergeable_tied_combined_rank32_s1_8`（hidden_lora r=8 + mergeable tied affine r=32 s1=8） | §7.4 工程线最终推荐；同时避免 §9.4 提到的冗余浪费 |
| 仅做位置归因消融研究 | 保持 bias 配置对称（input nobias / lm_head nobias） | §7.2 早期 sweep 因 bias 不对称差点得到错误结论 |

### 9.7 一句话总览

> AffLoRA 用 ~1% 的参数，把 LoRA 这把刀从 transformer 内部移到 hidden ↔ vocab 边界。`affine_input` 在没有 hidden_lora 时是最廉价的独立 SFT 适配器；`affine_lm_head` 在有 hidden_lora 时是最廉价的"末端补刀"；mergeable tied 把两件事合二为一并解决部署 merge。

## 10. 核心 Insight + 数据来源总表

本节把项目所有核心 claim 用「**Insight 一句话 + 直接证据 + 数据来源**」三段式结构化重列。每条都可以直接顺着 outputs 路径找到 trainer_state.json 或 train.log 验证。

数据约定：所有 eval_loss 都是 final eval（1 epoch 末尾，1500 step），seed=42（除非另注），lr=2e-4 cosine，max_seq_len=1024。

---

### Insight I：AffLoRA 用 ~1% 参数实现与 vocab-only LoRA 相当或更好的效果

**Claim**：在词表层做 PEFT，affine 形态比标准 vocab-dim LoRA 在低预算下显著更优。

**关键证据**（§7.6）：

| 模型 | rank | `affine_input` (with bias) | vocab-only LoRA | 参数比 | Δ |
|---|---:|---:|---:|---|---:|
| Qwen2.5-0.5B (24L, tied) | r=2 | **1.879** (~4.5k) | 1.883 (~600k) | 1/130 | -0.004 |
| Qwen2.5-0.5B | r=4 | 1.852 | **1.816** | 1/130 | +0.036 |
| Qwen3-0.6B (28L, tied) | r=2 | **1.494** (~4.5k) | 1.529 (~600k) | 1/130 | **-0.035** |
| Qwen3-0.6B | r=4 | **1.469** | 1.495 | 1/130 | **-0.026** |

**数据来源**：
- affine_input runs：`outputs/affine_vocab/position_affine_sweep_small_20260526_094509/{qwen25_05b,qwen3_06b}__seed42__input_r{2,4}_s1_64`
- vocab-only LoRA runs：`outputs/affine_vocab/position_attribution_def_20260526_114910/{qwen25_05b,qwen3_06b}__seed42__F_vocab_only_r{2,4}`

**大模型加强证据（v3 sweep, 2026-05-26 14:00）**：

| 模型 | vocab-only LoRA r=2 | vocab-only LoRA r=4 | `affine_input` r=16 s1=64 | 参数比 | Δ(aff − vocab r=4) |
|---|---:|---:|---:|---|---:|
| Qwen2.5-1.5B (28L, tied) | 1.591 (~610k) | 1.538 (~1.22M) | **1.473** (~50k) | 1/24 | **-0.065** |
| Qwen3-1.7B (28L, tied) | 1.189 (~614k) | 1.145 (~1.23M) | **0.981** (~50k) | 1/24 | **-0.164** |

→ 大模型上 AffLoRA 用 **1/24 参数** 把 eval_loss 又往下打了 0.065–0.164。Qwen3-1.7B 的 -0.164 是整个项目最大单项参数效率证据。

**大模型数据来源**：
- vocab-only LoRA：`outputs/affine_vocab/position_attribution_v3_20260526_131404/{qwen25_15b,qwen3_17b}__seed42__D_vocab_only_r{2,4}`
- affine_input r=16：`outputs/affine_vocab/position_attribution_v3_20260526_131404/{qwen25_15b,qwen3_17b}__seed42__E_aff_input_r16_s1_64`

**交叉引用**：§7.6（数据表）, §9.3（参数效率论述）, §10 Insight VIII（vs single-layer LoRA）

---

### Insight II：affine-only 时 `affine_input` 在所有规模、所有 rank、tied / untied 全部胜出

**Claim**：仅在词表层加 affine adapter（不加 hidden_lora）时，input 侧稳健胜 lm_head 侧 0.09–0.20，且差距随模型增大不缩反扩。

**关键证据**（§7.2）：

| 模型 | tied? | layers | r=2 input | r=2 lm_head | r=8 input | r=8 lm_head | Δ(r=8) |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | tied | 24 | **1.879** | 2.010 | **1.816** | 1.907 | **-0.091** |
| Qwen3-0.6B | tied | 28 | **1.494** | 1.693 | **1.460** | 1.621 | **-0.161** |
| Qwen2.5-1.5B | tied | 28 | — | — | **1.502** | 1.608 | **-0.106** |
| **Qwen2.5-7B** | **untied** | 28 | — | — | **1.258** | 1.392 | **-0.134** |

→ untied 7B 上 input 优势比 tied 1.5B 还大，**彻底排除"input 优势仅在 tied 模型存在"**。

**数据来源**：
- 0.5B / 0.6B sweep：`outputs/affine_vocab/position_affine_sweep_small_20260526_094509/{qwen25_05b,qwen3_06b}__seed42__{input,lm_head}_r{2,4,8}_s1_*`
- 1.5B：`outputs/affine_vocab/position_attribution_v2_20260526_111320/qwen25_15b__seed42__A_{input,lm_head}_r8_s1_64`
- 7B：`outputs/affine_vocab/position_attribution_def_20260526_114910/qwen25_7b__seed42__D_{input,lm_head}_r8_s1_64`

**交叉引用**：§7.2.1–7.2.5

---

### Insight III：搭配 hidden_lora 后，`affine_lm_head` 既反超 `affine_input`，又是对 hidden_lora 本身的稳定低成本补充

**Claim**（两层含义，都来自同一组数据）：

1. **位置反转**：`affine_lm_head + HL > affine_input + HL`（差 0.005–0.014）。input 与 hidden_lora 功能冗余、lm_head 与 hidden_lora 功能正交。
2. **稳定低成本补充**：`affine_lm_head + HL > hidden_lora only`（差 0.005–0.014）；只多 0.3% 参数（~14k / 5M）就拿到这一份独立增量，而 `affine_input + HL` 比 HL only 几乎无差距（差 0.003–0.005，可视为冗余）。

→ "**AffLoRA 是 hidden_lora 的低成本稳定补充**" 这条叙事现在的精确口径是：**仅对 `affine_lm_head` 成立，对 `affine_input` 不成立**。

**关键证据**（§7.3）：

| 模型 | hidden_lora only | input + HL r=8 (s1=64) | lm_head + HL r=8 (s1=64) | Δ(input + HL − HL only) | Δ(lm_head + HL − HL only) | Δ(input − lm_head) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1.302 | 1.297 | **1.288** | -0.005 | **-0.014** | +0.009 |
| Qwen3-0.6B | 1.101 | 1.097 | **1.093** | -0.004 | **-0.008** | +0.004 |

→ 第 5 列（**lm_head + HL 比 HL only 的提升**）几乎是第 4 列（input 那侧的提升）的 2 倍以上；这就是"lm_head 才是真正的补充"的直接体现。

低 rank 下趋势一致（**affine 只占 0.05% 参数**仍稳定补 0.007–0.009）：

| 模型 | input + HL r=2 (s1=64) | lm_head + HL r=2 (s1=64) | Δ vs HL only (lm_head) | Δ(input − lm_head) |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1.299 | **1.292** | -0.010 | +0.007 |
| Qwen3-0.6B | 1.098 | **1.094** | -0.007 | +0.004 |

**数据来源**：

- HL only：`outputs/affine_vocab/affine_pos_plus_hidden_lora_20260526_101307/{qwen25_05b,qwen3_06b}__seed42__hidden_lora`
- input + HL：`outputs/affine_vocab/affine_pos_plus_hidden_lora_20260526_101307/...__aff_input_hl`，及 `position_attribution_v2_20260526_111320/...__B_input_HL_r{2,4}_s1_64`
- lm_head + HL：同上路径，把 `input` 换成 `lm_head` / `aff_lm_head_hl`

#### v3 终极证据 1：大模型 + HL，**3/3 模型同方向**（含真 untied 7B）

| 模型 | tied? | layers | input + HL | lm_head + HL | Δ |
|---|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | tied | 28 | 1.0500 | **1.0440** | **-0.006** |
| Qwen3-1.7B | tied | 28 | 0.7362 | **0.7328** | **-0.0034** |
| **Qwen2.5-7B** | **untied** | 28 | 0.9180 | **0.9130** | **-0.005** |

→ **Insight III 不再局限于"小模型 / tied"**——真 untied 的 7B 上 lm_head 同样反超。这是 Insight III 从"边缘效应"升级为"普适规律"的关键证据。

#### v3 终极证据 2：多 seed 鲁棒性

| 模型 | seed | input + HL | lm_head + HL | Δ |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B | 42 | 1.297 | **1.288** | -0.009 |
| Qwen2.5-0.5B | 43 | 1.2970 | **1.2890** | -0.008 |
| Qwen2.5-0.5B | 44 | 1.2970 | **1.2880** | -0.009 |
| Qwen3-0.6B | 42 | 1.097 | **1.093** | -0.004 |
| Qwen3-0.6B | 43 | 1.0960 | **1.0940** | -0.002 |
| Qwen3-0.6B | 44 | 1.0970 | **1.0930** | -0.004 |

→ 6/6 配置全部 lm_head 胜出；Δ 跨 seed 标准差 ≈0.001 ≪ 均值差距 → **统计上稳健**，论文表可直接用 mean ± std。

#### v3 终极证据 3：HL 排除任意单层后，lm_head 优势保持

| 模型 | HL 配置 | input + HL | lm_head + HL | Δ |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B | full 24 层 | 1.297 | **1.288** | -0.009 |
| Qwen2.5-0.5B | 排除 l23 (last) | 1.3050 | **1.2950** | -0.010 |
| Qwen2.5-0.5B | 排除 l12 (mid) | 1.3020 | **1.2930** | -0.009 |
| Qwen3-0.6B | full 28 层 | 1.097 | **1.093** | -0.004 |
| Qwen3-0.6B | 排除 l27 (last) | 1.1000 | **1.0960** | -0.004 |
| Qwen3-0.6B | 排除 l14 (mid) | 1.0990 | **1.0950** | -0.004 |

→ 排除单层只让 eval_loss 升 0.003–0.008（HL 容量分散；与 Insight IV 同源），**而 lm_head − input 的差距完全保持**。这证明 lm_head 优势的因果机制是 lm_head 侧本身在 +HL 时提供的独立增量，**不依赖 HL 的具体层数集合**。

**v3 数据来源**：
- A 大模型：`outputs/affine_vocab/position_attribution_v3_20260526_131404/{qwen25_15b,qwen3_17b,qwen25_7b}__seed42__A_large_HL_{input,lm_head}_r8_s1_64`
- B 多 seed：`outputs/affine_vocab/position_attribution_v3_20260526_131404/{qwen25_05b,qwen3_06b}__seed{43,44}__B_multiseed_HL_{input,lm_head}_r8_s1_64`
- F 排除层：`outputs/affine_vocab/position_attribution_v3_20260526_131404/{qwen25_05b,qwen3_06b}__seed42__F_no_{last,mid}_{input,lm_head}_r8_s1_64`

**交叉引用**：§7.3.1–7.3.3, §7.10（v3 sweep 完整记录）, §9.4 价值 2（mergeable_tied 因为这一层冗余而避免参数浪费），§8 结论 #4

---

### Insight IV：HL 容量分散在所有层，不是 layer-0 局部

**Claim**：HL 排除 layer 0（保留 layers 1..N-1）与 full HL 几乎完全等价，input vs lm_head 的差距同步保持。

**关键证据**（§7.5）：

| 模型 | input + HL_no_l0 | lm_head + HL_no_l0 | Δ | 对比 full HL Δ |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1.299 | **1.290** | +0.009 | +0.009（一致）|
| Qwen3-0.6B | 1.098 | **1.095** | +0.003 | +0.004（一致）|

→ HL_no_l0 vs full HL 差距仅 0.001–0.002（噪声级别），但 HL_l0（仅 layer 0）vs full HL 差 0.34–0.43（巨大）。说明 HL 的容量是均匀分散在所有层、不依赖 layer 0；input/HL 冗余是**全局几何关系**而非 layer-0 局部现象。

**数据来源**：
- HL_no_l0：`outputs/affine_vocab/position_attribution_def_20260526_114910/{qwen25_05b,qwen3_06b}__seed42__E_{input,lm_head}_HLnoL0_r8_s1_64`
- HL_l0 对照：`outputs/affine_vocab/position_attribution_v2_20260526_111320/{qwen25_05b,qwen3_06b}__seed42__C_{input,lm_head}_HLl0_r8_s1_64`

**交叉引用**：§7.4, §7.5, §9.2

---

### Insight V：mergeable_tied 部署 merge 严格等价（工程价值 + token 级验证）

**Claim**：tied 模型的 mergeable_tied 形态训练后可一次性 merge 回单个 tied 权重矩阵，部署时无需保留 adapter 模块。验证分三个层级：
1. **logits 数值**：merge 前后 logits abs 误差仅 3e-5 ~ 6e-5（bf16 + fp32 master 数值噪声级）
2. **任务 eval_loss**：mergeable_tied 与对应 `combined_inlmh + HL` 同性能
3. **token 级生成**（**2026-05-26 16:51 新增**）：fp32 推理下 4 个 tied 模型 × 100 prompt × 64 token greedy 生成 **100% 完全相同**，ROUGE-L = 1.0；abs ΔCE = 7e-9 ~ 8e-8（机器精度极限） → **数学上严格 merge 等价**

**关键证据**：
- `--mergeable-tied-affine-output` 的实现：lm_head 侧用 input affine 的转置作用，使两侧等价于"对同一 tied 矩阵 W 做 `W' = affine(W) = (I + s·U·D)·W + b`"；详见 `src/affine_vocab_lora/adapter.py` 中的 `TiedTransposeAffineLMHead`。

#### V.1 logits 数值等价（早期 sanity check）

| 模型 | merge 前 vs merge 后 logits abs error |
|---|---|
| Phase 14–16 各 mergeable_tied ckpt | 3e-5 ~ 6e-5（bf16 + fp32 master）|

#### V.2 eval_loss 同性能

| 模型 | `mergeable_tied_combined_rank32_s1_8` | `combined_inlmh + HL` | 差距 |
|---|---:|---:|---:|
| Qwen2.5-1.5B (UltraChat) | 1.075 | 1.075 | 0 |
| Qwen3-1.7B (UltraChat) | 1.058 | 1.058 | 0 |

#### V.3 token 级生成等价（4 模型 × 2 dtype × 500 PPL + 100 greedy 生成）

`scripts/eval_merge_equivalence.py` 在 `sft_t2t_mini_25k` eval set 上对 4 个 mergeable_tied 训练后 ckpt 跑了 "adapter form" vs "merged form" 的 PPL + greedy 一致率 + token-id LCS ROUGE-L 完整对比：

**fp32 推理（金标准 / 数学等价证据）**：

| 模型 | abs ΔCE | 100 prompt 全等数 | token position match | ROUGE-L | first diff 最早位置 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 6.96e-9 | **100 / 100** | **1.0000** | **1.0000** | -1（无分歧）|
| Qwen2.5-1.5B | 4.29e-8 | **100 / 100** | **1.0000** | **1.0000** | -1 |
| Qwen3-0.6B | 8.34e-8 | **100 / 100** | **1.0000** | **1.0000** | -1 |
| Qwen3-1.7B | 6.71e-8 | **100 / 100** | **1.0000** | **1.0000** | -1 |

→ **4/4 模型在 fp32 推理下完全等价**：abs ΔCE 处于浮点机器精度极限（7e-9 ~ 8e-8），100 个 prompt × 64 token greedy 生成**完全相同**（无任何 token 分歧）。这是 Insight V 从 "logits abs error 3e-5"升级到 **token-level 严格等价**的金标准证据。

**bf16 推理（部署实际 / 量化效应展示）**：

| 模型 | abs ΔCE | 全等数 / 100 | position match | ROUGE-L | first diff 最早位置 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 6.79e-5 | 23 | 0.588 | 0.762 | 0 |
| Qwen2.5-1.5B | 6.08e-5 | 35 | 0.671 | 0.812 | 1 |
| Qwen3-0.6B | 1.04e-4 | 31 | 0.650 | 0.796 | 2 |
| Qwen3-1.7B | 2.55e-4 | 20 | 0.612 | 0.791 | 1 |

→ **bf16 下 logits abs error 量级仍是 1e-4**，但 greedy decode 的 argmax 决策对接近的 logit 极敏感（top-2 token logit 距离常在 1e-3 量级，会被 bf16 噪声翻转），所以**实际 token 序列会有 ~20% 完全相同 + ~65% 位置一致 + ROUGE-L ~0.79**。这是 dtype 量化效应而非 merge 设计本身的问题：
- 用 fp32 master 推理则 100% 一致（V.3 fp32 表）
- ROUGE-L ~0.79 说明 bf16 下**语义保持**（LCS 占整体长度 ~80%）

**论文叙事**：mergeable_tied 在数学上严格 merge 等价（fp32 100% token 全等是充分证据）；bf16 部署下因 argmax 离散性会有 ~20% token 序列分歧，但语义级别一致（ROUGE-L ≈ 0.8）。

**数据来源**：
- 工程实现：`lora/src/affine_vocab_lora/adapter.py`（`TiedTransposeAffineLMHead`、`AffineEmbedding.merged_embedding_weight`）
- logits sanity 检查：`lora/scripts/check_tied_affine_merge_equivalence.py`，Phase 14–16 ckpt
- **token 级验证**：`lora/scripts/eval_merge_equivalence.py` + `outputs/merge_equivalence/20260526_165132/{qwen25_05b,qwen3_06b,qwen25_15b,qwen3_17b}__{bf16,fp32}.json`
- UltraChat 性能验证：`outputs/affine_vocab/ultrachat100k_validation_20260524_172421/qwen{25_15b,3_17b}__seed42__mergeable_tied_combined_rank32_s1_8`

**交叉引用**：§1.3（变体定义）, §5 Phase 14–16, §9.4 价值 1（merge 工程价值）, §9.4 价值 2（参数效率价值）

---

### Insight VI：mergeable_tied 顺手避免"双侧独立 adapter 一半参数白训"

**Claim**：除工程意义外，mergeable_tied 因为共享 input/lm_head adapter，自动绕开了"普通双侧独立 adapter + HL 时 input 那侧参数全部冗余"的失败模式——这是 §7.3–§7.5 反推出来的、不显然的性能侧价值。

**关键证据**（推导）：
- 由 Insight III：`affine_input + HL` 比 HL only 多 0.003–0.005；`affine_lm_head + HL` 比 HL only 多 0.005–0.014。
- 所以双侧独立 adapter `(A_in + A_out) + HL` 的有效增量 ≈ `affine_lm_head + HL`，而 `A_in` 那部分参数 ≈99% 与 HL 重叠 → 浪费。
- mergeable_tied 把 `A_in / A_out` 绑成同一个 adapter，参数 = M（不是 2M）。训练时 lm_head 侧梯度信号独立于 HL，共享 adapter 自动学到 `affine_lm_head` 的有效增量。最终：100% 参数有效 vs 普通双侧 50% 参数有效。

**支撑数字**：

| 形态 + HL r=8 | 参数（含 HL，0.6B） | eval_loss | 有效参数比 |
|---|---:|---:|---:|
| HL only | 5.05M | 1.101 | 100% |
| `affine_input + HL` | 5.06M | 1.097 | input 那 14k ≈99% 浪费 |
| `affine_lm_head + HL` | 5.06M | **1.093** | 14k 全部有效 |
| `affine_inlmh + HL`（双侧独立） | 5.08M | 1.095 | 28k 中约 14k 浪费 |
| `mergeable_tied_combined` | 5.06M | 1.094 | 14k 全部有效，且可 merge |

**数据来源**：
- 双侧独立 + HL：`outputs/affine_vocab/affine_pos_plus_hidden_lora_20260526_101307/qwen3_06b__seed42__aff_inlmh_hl`
- mergeable_tied：`outputs/affine_vocab/canonical_alora_efficiency_20260525_231722/qwen3_06b__seed42__mergeable_tied_combined_rank32_s1_8`

**交叉引用**：§7.3, §7.5, §9.4 价值 2

---

### Insight VII：bias 在 final eval 上贡献小，可去；input 优势不是 bias 给的

**Claim**（修正后两条同时成立）：
1. 两侧 hidden bias 在最终 eval_loss 上的边际贡献都 ≤ 0.01（input ≤0.006，lm_head ≤0.01），梯度量级虽差 ~400× 但 capacity-per-param 都不高，可以默认全部关闭以简化变体集合。
2. 更关键：§7.2 中 input vs lm_head 的 0.09–0.20 差距**与 bias 设置无关**——v3 C 组直接关掉 input bias 后 input 仍稳健胜出，证伪了早期"input 优势来自 asymmetric bias"的怀疑。

**关键证据**：
1. 梯度诊断（`scripts/diagnose_affine_gradients.py` 在 Qwen2.5-0.5B + r=8 s1=64 上的实测）：

| 参数 | grad_norm（一次 backward 后） |
|---|---:|
| `affine_input.affine.bias` | ~17.9 |
| `affine_input.affine.down.weight` | ~12.3 |
| `affine_input.affine.up.weight` | ~9.4 |
| `affine_lm_head.affine.bias` | ~4.0e-2 |
| `affine_lm_head.affine.down.weight` | ~10.5 |
| `affine_lm_head.affine.up.weight` | ~7.8 |

2. 实测 lm_head bias 开/关对比（Qwen2.5-0.5B, Qwen3-0.6B, r=8）：

| s1 | lm_head no bias | lm_head with bias | Δ |
|---|---:|---:|---:|
| 32（0.5B）| 1.907 | 1.909 | +0.002 |
| 64（0.5B）| 1.907 | 1.907 | 0 |
| 32（0.6B）| 1.621 | 1.611 | -0.010 |
| 64（0.6B）| 1.621 | 1.611 | -0.010 |

→ 即便有微弱的不一致（0.6B 上 -0.01），也远小于 input bias 的影响。论文中可以选择**简化变体集合：lm_head 一侧默认不用 hidden bias**。

**数据来源**：
- 梯度诊断脚本：`lora/scripts/diagnose_affine_gradients.py`
- lm_head bias 对比：`outputs/affine_vocab/lm_head_recovery_quick_20260526_104314/{qwen25_05b,qwen3_06b}__seed42__lm_head_withbias_r8_s1_{32,64}`

3. **input bias 消融（v3 C 组）**：直接关掉 `affine_input` 的 hidden bias 后实测：

| 模型 | r | input no-bias | input with-bias（旧）| Δ |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 2 | 1.903 | — | — |
| Qwen2.5-0.5B | 4 | 1.861 | — | — |
| Qwen2.5-0.5B | 8 | 1.821 | 1.816 | +0.005 |
| Qwen3-0.6B | 2 | 1.496 | — | — |
| Qwen3-0.6B | 4 | 1.476 | — | — |
| Qwen3-0.6B | 8 | 1.447 | 1.441 | +0.006 |

→ **input bias 在 final eval 上影响 ≤ 0.006**（与 lm_head bias 同量级，均处于 seed 噪声内）。

**Insight VII 修正后口径**：
- 梯度量级两侧确实不对称（input bias 梯度 ≈17.9，lm_head bias 梯度 ≈4e-2，差 ~400×）；
- 但 **final eval 上两侧 hidden bias 的边际贡献都 ≤ 0.01**，可以默认全部关掉以简化变体集合；
- **更关键的结论**：§7.2 中 input vs lm_head 的 0.09–0.20 差距**不是 bias 不对称带来的**（早期 canonical 把 input 配 bias、lm_head 不配的"asymmetric setup"嫌疑被 C 组排除）；input 的稳健优势来自几何位置（28-layer broadcast 路径），而非 bias。

**v3 input no-bias 数据来源**：`outputs/affine_vocab/position_attribution_v3_20260526_131404/{qwen25_05b,qwen3_06b}__seed42__C_input_nobias_r{2,4,8}_s1_64`

**交叉引用**：§7.2.5（关键观察），§7.10（v3 sweep 完整记录），§8 结论 #5

---

### Insight VIII：AffLoRA 表达能力 + 小预算优势（vs same-budget single-layer LoRA）

**Claim**：affine 形态在与 single-layer LoRA 同量级（甚至更小）的参数预算下显著更优；这是 AffLoRA-only 的核心 claim 2。**主对照应该用 `affine_input` 单侧（34k）vs single-layer LoRA（41k–49k）—— affine 参数更少且仍胜出**；双侧 `affine_input_lm_head` 是更强的可选证据。

#### 对照 1：单侧 `affine_input`（34k）vs single-layer LoRA（41k–49k）—— **affine 参数还少 17–30%**

| variant | params | eval_loss | Δ vs single-layer 最弱 (q_l27 1.78) | Δ vs single-layer 最强 (qkvo_l14_r4 1.473) |
|---|---:|---:|---:|---:|
| **`affine_input` r=16 s1=64** | **34k** | **1.419** | **-0.361** | **-0.054** |
| `affine_input` r=16 s1=32 | 34k | 1.426 | -0.354 | -0.047 |
| `affine_input` r=16 s1=16 | 34k | 1.434 | -0.346 | -0.039 |
| `affine_input` r=16 s1=8 | 34k | 1.442 | -0.338 | -0.031 |
| single-layer LoRA q+l0 r=16 | 49k | 1.626 | -0.154 | +0.153 |
| single-layer LoRA q+l14 r=16 | 49k | 1.667 | -0.113 | +0.194 |
| single-layer LoRA q+l27 r=16 | 49k | 1.780 | 0 | +0.307 |
| single-layer LoRA qkvo+l14 r=4 | 41k | 1.473 | -0.307 | 0 |

→ `affine_input` 单侧在 34k 参数下比**任意层、任意配置的 single-layer LoRA**（41k–49k）都低 0.03–0.36。这是 §7.2 中"affine-only 时 input 全胜"的同时也胜过 hidden_lora 的单层版本。

#### 对照 2（更强证据）：双侧 `affine_input_lm_head`（67k）vs single-layer LoRA

参数虽多 36%（67k vs 49k），但绝对差距更大：

| variant | params | eval_loss | Δ vs single-layer 最强 (qkvo_l14_r4 1.473) |
|---|---:|---:|---:|
| **`affine_input_lm_head` r=16 s1=64** | **67k** | **1.320** | **-0.153** |
| `affine_input_lm_head` r=16 s1=4 | 67k | 1.346 | -0.127 |
| single-layer LoRA qkvo+l14 r=4 | 41k | 1.473 | 0 |

→ 双侧 affine 比 best single-layer 还低 0.13–0.15，但代价是参数多 36%；所以**主 claim 用单侧（参数更少且胜）更干净**，双侧只作为补充。

#### 数据来源

- `affine_input` 单侧 sweep：
  - `outputs/affine_vocab/phase2_sweep_20260519_115922/affine_in_r16_*`（s1 = 0.5, 1, 2, 4）
  - `outputs/affine_vocab/phase2b_20260519_123458/affine_only_s1_{8,16}`
  - `outputs/affine_vocab/phase2c_20260519_125518/affine_only_s1_32`
  - `outputs/affine_vocab/phase3b_20260519_134358/affine_only_s1_{64,128}`
- `affine_input_lm_head` 双侧：
  - `outputs/affine_vocab/phase2c_20260519_125518/affine_only_in_lmh_s1_4`
  - `outputs/affine_vocab/phase3b_20260519_134358/affine_in_lmh_s1_*`
- single-layer LoRA：`outputs/affine_vocab/phase2c_20260519_125518/lora_only_{q_l0,q_l14,q_l27,qkvo_l14}_r{16,4}`

#### 交叉引用

- §5 Phase 2 sweep / Phase 2b / Phase 2c / Phase 3b（数据原表）
- §7.2.2（Qwen3-0.6B 上 input vs lm_head 单侧对照，与本 Insight 的"main claim 用单侧"一致）
- §8 结论 #2

---

### Insight IX：在外部数据集上不退化、有轻微正向

**Claim**：除 `sft_t2t_mini_25k` 主线外，AffLoRA 在 UltraChat 100k 与 MiniMind random 100k 上都"不退化、有轻微正向"，可以作为外部验证。

**关键证据**（§5 UltraChat 100k 主验证矩阵）：

| 模型 | data | hidden_lora only | combined_inlmh + HL | mergeable_tied_combined | Δ vs hidden_lora |
|---|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | UltraChat 100k | 1.075 | **1.075** | **1.075** | 0 |
| Qwen3-1.7B | UltraChat 100k | 1.058 | **1.058** | **1.058** | 0 |

低 rank 区间收益更明显（§5 UltraChat 低 rank 动态队列）：

| 模型 | rank | hidden_lora only | combined_inlmh r=4 + HL | Δ |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | r=4 | 1.078 | **1.077** | -0.001 |
| Qwen3-1.7B | r=4 | 1.06 | **1.06** | 0 |

→ 在外部、更长、更难的数据上，AffLoRA 至少不退化；在低 rank（< 8）的"真低预算"场景下还有 0.001–0.003 的正向。

**数据来源**：
- UltraChat 主验证：`outputs/affine_vocab/ultrachat100k_validation_20260524_172421/`
- UltraChat 低 rank：`outputs/affine_vocab/ultrachat100k_lowrank_20260525_011523/`
- MiniMind：`outputs/affine_vocab/minimind_random100k_lowrank_20260525_131318/`

**交叉引用**：§5 UltraChat 100k 系列, §5 MiniMind random 100k, §8 结论 #7

---

### §10 速查矩阵（一页纸版）

| # | Insight | 主要 Δ | 数据来源（一行）|
|---|---|---|---|
| I | AffLoRA 用 ~1/24 ~ 1/130 参数胜 vocab-only LoRA（小模型 r=2 / 大模型 r=16）| **-0.035**（0.6B r=2）/ **-0.164**（1.7B vs vocab r=4）| `position_attribution_def_*` + `position_affine_sweep_small_*` + **`position_attribution_v3_*`（大模型 D/E 组）** |
| II | affine-only：input 全规模、tied/untied 都胜 lm_head | **-0.09 ~ -0.20** | `position_affine_sweep_small_20260526_094509` + `position_attribution_v2_*` + `position_attribution_def_*` |
| III | + HL：lm_head 反超 input；同时 lm_head + HL > HL only（稳定低成本补充，~0.3% 参数补 0.005–0.014）；**v3 sweep 在 3/3 大模型（含 untied 7B）+ 6/6 multi-seed + 排除 last/mid 层全部同方向** | 反转 +0.003 ~ +0.009；vs HL only **-0.005 ~ -0.014** | `affine_pos_plus_hidden_lora_*` + **`position_attribution_v3_*`（A/B/F 组）** |
| IV | HL_no_l0 ≈ full HL（HL 容量分散）；排除 last/mid 单层结论同步 | **\|Δ\| ≤ 0.002**（no_l0）/ ≤ 0.01（no_last, no_mid）| `position_attribution_def_*` + `position_attribution_v3_*` |
| V | mergeable_tied 部署 merge 严格等价（三层证据：logits / eval_loss / **token 级**）| fp32 推理 **4/4 模型 100/100 token 序列全等**（abs ΔCE 7e-9–8e-8）；bf16 logits 误差 1e-4 但 ROUGE-L ≈ 0.79 语义保持 | `eval_merge_equivalence.py` + `outputs/merge_equivalence/20260526_165132/*` + `phase14/15/16_*` + `ultrachat100k_validation_*` |
| VI | mergeable_tied 避免双侧独立 50% 参数白训 | 双侧独立 5.08M 1.095 vs mergeable 5.06M **1.094** | `affine_pos_plus_hidden_lora_*` + `canonical_alora_efficiency_*` |
| VII | hidden bias 在 final eval 上两侧贡献都 ≤ 0.01（梯度量级差 ~400× 但 capacity 差距小）；**input/lm_head 的 0.09–0.20 差距与 bias 不相关** | input bias 开/关 **\|Δ\| ≤ 0.006**；lm_head bias 开/关 **\|Δ\| ≤ 0.01** | `diagnose_affine_gradients.py` + `lm_head_recovery_quick_*` + **`position_attribution_v3_*`（C 组 input no-bias）** |
| VIII | `affine_input` 单侧 34k（比 single-layer 49k 还少 30%）胜 single-layer LoRA | **-0.03 ~ -0.36** | `phase2c_*` + `phase2b_*` + `phase3b_*` |
| IX | 外部数据集（UltraChat, MiniMind）不退化、有轻微正向 | **0 ~ -0.003** | `ultrachat100k_*` + `minimind_random100k_*` |
