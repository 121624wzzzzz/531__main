# im_exp 工作区上下文

本文档用于让新的对话快速接手当前工作区。重点保留路径、环境、各子项目概况、数据/权重位置和常用命令。

## 工作区全景

四个实验项目围绕同一个核心问题展开——**embedding/lm_head 词表层的参数化与训练效率**——分别从预训练架构改造、后训练适配、经典 LSTM 复现三个角度切入，外加论文产出目录。

```
im_exp/
├── minimind/              ← 预训练端：S1-S13 embedding/head 变体（MiniMind dense 模型）
├── lora/                  ← 后训练端：AffLoRA 低秩仿射适配（Qwen base 模型）
├── better-wt-relaxation/  ← 经典复现：PTB LSTM embedding relaxation（S1-S21）
├── models/                ← Qwen2.5/Qwen3 base 模型权重（供 lora/ 使用）
├── paper/                 ← 论文对外产出（不参与实验流程）
│   ├── figures/           ← 论文用图
│   └── submission/        ← EMNLP 2026 匿名投稿代码包（AGD + A-LoRA）
├── set                    ← 环境激活脚本
└── CONTEXT.md
```

**项目间关系**：
- minimind 的 S1-S12 变体定义直接启发了 better-wt-relaxation 和 lora 的实验设计
- lora 的核心洞察（base→instruct 词表层仿射关系）来自外部分析，与 minimind 的 S12/S13 组合变体思路相通
- better-wt-relaxation 的 S7（输出侧乘性低秩）发现与 lora 的 AffLoRA 输出侧适配在数学形式上高度相似
- paper/submission 包含两篇 EMNLP 2026 投稿：AGD（预训练端 untying）和 A-LoRA（后训练端 placement），分别对应 minimind 和 lora 的实验线

---

## 环境

入口路径：`/home/wz/projects/mypro/im_exp`（实际路径：`/commondocument/wz/cross_encoder_workspace/im_exp`）

```bash
source /home/wz/projects/mypro/im_exp/set
```

`set` 激活 conda 环境 `torch24`，设置 CUDA、HF/PIP 镜像和 `TOKENIZERS_PARALLELISM=false`。

核心环境：

- OS：Ubuntu 20.04.4 LTS
- GPU：8 × NVIDIA A100-SXM4-80GB
- NVIDIA Driver：550.90.07，最高支持 CUDA 12.4
- Python：3.10.20
- PyTorch：2.4.1+cu124
- CUDA runtime / nvcc：12.4
- 常用库：transformers、accelerate、datasets、peft、trl、deepspeed、bitsandbytes

---

## MiniMind — 预训练 Embedding/Head 变体实验

项目位置：`/home/wz/projects/mypro/im_exp/minimind`

该目录是从 `minimind` 复制并改造来的，仅覆盖 dense 模型的预训练阶段（无 SFT/RL/DPO）。核心问题：**在预训练阶段，不同的 embedding/lm_head 参数化方式（tied/untied、加性/乘性低秩、输入侧/输出侧）对 loss 有何影响？**

### S1-S13 变体定义

所有低秩变体使用 `rank=32`，统一开启 `lm_head_bias=True`。

| 变体 | Tied | 公式 | 额外参数（MiniMind V=6400 H=768） |
|---|---|---|---|
| **S1** | 是 | `E = U = W`（标准 tied baseline） | 0 |
| **S2** | 否 | `E ≠ U`（标准 untied baseline） | +4,915,200 (+7.69%) |
| **S3** | 是 | `E_eff = E - β`（输入侧 hidden 维平移） | +768 (+0.0012%) |
| **S4** | 是 | `E_eff = E + A·B`（输入侧词表维加性 LoRA, A:V×r, B:r×H） | +229,376 (+0.36%) |
| **S5** | 是 | `logits = h·W^T + (B_o(h))·A_o^T`（输出侧词表维加性 LoRA） | +229,376 (+0.36%) |
| **S6** | 是 | `E_eff = E + B(A(E))`（输入侧乘性低秩, A:H×r, B:r×H） | +49,152 (+0.08%) |
| **S7** | 是 | `logits = lm_head(h + B(A(h)))`（输出侧乘性低秩） | +49,152 (+0.08%) |
| **S8** | 否 | S2 + S3（untied + 输入侧 hidden 平移） | +4,915,968 |
| **S9** | 否 | S2 + S4（untied + 输入侧加性 LoRA） | +5,144,576 |
| **S10** | 否 | S2 + S6（untied + 输入侧乘性 LoRA） | +4,964,352 |
| **S11** | 是 | `logits = lm_head(h - β)`（输出侧 hidden 维平移） | +768 |
| **S12** | 是 | S3 + S6 组合：`E_eff = E - β + B(A(E))` | +49,920 (+0.08%) |
| **S13** | 是 | S4 + S6 组合（补训） | — |

### 实验配置

- 模型：MiniMind dense, hidden=768, 8 层, 无 MoE
- 训练：8×GPU DDP, bf16, lr=5e-4, batch=1792, seq_len=340
- MiniMind：2 epochs，数据 `pretrain_t2t.jsonl`（7.8G），本地 MiniMind tokenizer
- FineEdu/GPT-2：1 epoch, ~6B tokens, batch=640，数据 `fineweb_edu_gpt2_6b_packed_340`，GPT-2 tokenizer
- 验证：尾部 1% 数据切分
- 随机种子：42, 123, 2026

### 关键结果

**MiniMind 原始数据（三 seed 平均，按 loss 排序）**：

| 排名 | 变体 | 平均 Loss | 平均 PPL | vs S1 |
|---|---|---|---|---|
| 1 | **S3** | 1.903291 | 6.708 | **-0.011671** |
| 2 | **S12** | 1.906849 | 6.732 | **-0.008113** |
| 3 | S4 | 1.912134 | 6.768 | -0.002828 |
| 4 | S6 | 1.912824 | 6.772 | -0.002138 |
| 5 | S1 (基线) | 1.914962 | 6.787 | 0 |
| ... | ... | ... | ... | ... |
| 10 | S2 (untied) | 1.977121 | 7.222 | +0.062159 |

**FineWeb-Edu/GPT-2（三 seed 平均）**：

| 排名 | 变体 | 平均 Loss | 平均 PPL | vs S1 |
|---|---|---|---|---|
| 1 | **S12** | 3.230316 | 25.288 | **-0.007581** |
| 2 | S3 | 3.233494 | 25.368 | -0.004403 |
| 3 | S4 | 3.235201 | 25.411 | -0.002696 |
| ... | ... | ... | ... | ... |
| 8 | S1 (基线) | 3.237896 | 25.480 | 0 |

### 主要结论

1. **输入侧修改最稳定**：S3 在 MiniMind 上最优，S12 在 FineEdu 上最优
2. **S12（S3+S6 组合）跨数据集泛化最强**：在更大的 tokenizer 和数据上优于纯 S3
3. **输出侧改造收益不稳定**：S7/S11 在两个数据集均未稳定超过 S1——与 better-wt-relaxation 的 S7 发现形成有趣对比（PTB LSTM 上输出侧乘性低秩效果最好）
4. **Untied baseline (S2) 始终弱于 tied**：在当前模型规模下直接分离 embedding/head 不是有效方向
5. 完整实验报告：`minimind/实验.md`

### 训练命令

```bash
cd /home/wz/projects/mypro/im_exp/minimind
source ../set

# MiniMind 原始数据
GPUS=0,1,2,3,4,5,6,7 bash train_s1_s12_pretrain.sh
GPUS=0,1,2,3,4,5,6,7 SEED=123 WEIGHT_PREFIX=pretrain_v2_seed123 LOG_DIR=logs/s1-s12-pretrain-v2-seed123 bash train_s1_s12_pretrain.sh

# FineWeb-Edu/GPT-2
GPUS=0,1,2,3,4,5,6,7 SEED=42 bash train_fineedu_gpt2_pretrain.sh
```

### 常用检查

```bash
nvidia-smi
ls -lh /home/wz/projects/mypro/im_exp/minimind/out/*.pth
ls -lh /home/wz/projects/mypro/im_exp/minimind/logs/*summary*.csv
tail -f /home/wz/projects/mypro/im_exp/minimind/logs/fineedu-gpt2-6b-seed42/s12.log
```

### 注意事项

- `set` 必须用 `source`，不能直接执行
- 早期 dense/MoE、pretrain_s1-s10、对话推理、LoRA 推理、RL rollout 等产物已清理
- `checkpoints/` 占用较大，如只保留最终模型可清理 `*_resume.pth`

---

## AffLoRA — 后训练 Embedding/LM Head 低秩仿射适配

项目位置：`/home/wz/projects/mypro/im_exp/lora`

详细文档：`lora/CLAUDE.md`、`lora/README.md`、`lora/docs/`

### 核心概念

标准 LoRA 在 SFT 时通常跳过 `embed_tokens` 和 `lm_head`（词表维层参数量大且难以高效训练）。AffLoRA 的核心洞察来自 base→instruct 全词表分析：这些层的变化近似满足 hidden 维仿射关系 `W_instruct ≈ W_base × A + b`。AffLoRA 不在冻结的权重矩阵上操作，而是在 hidden 状态上应用低秩仿射适配：

```
h' = h + s1 · U(D(h)) + s2 · b
```

其中 D: H→r, U: r→H 通过 rank bottleneck。定位是 **hidden LoRA 的补充模块而非替代**——收益在低 hidden LoRA rank 时最大，随 hidden 容量增加递减。

### 变体定义

8 个训练变体由 `--variant` 控制：

| 变体 | 训练内容 | 用于 |
|---|---|---|
| `full_finetune` | 全量微调（上限参考） | — |
| `hidden_lora` | 标准 LoRA（q/k/v/o/up/down/gate） | Claim 1a baseline |
| `affine_input_plus_hidden_lora` | hidden LoRA + embed_tokens AffLoRA | Claim 1a 输入侧 |
| `affine_input_lm_head_plus_hidden_lora` | hidden LoRA + embed_tokens + lm_head AffLoRA | Claim 1a 完整条件 |
| `affine_input` | 仅 embed_tokens AffLoRA | Claim 2/3 |
| `affine_input_lm_head` | embed_tokens + lm_head AffLoRA | Claim 2/3 |
| `hidden_lora` + `--include-emb-lmh-lora-rank N` | 传统 vocab 维 LoRA（对照） | Claim 1b |
| `hidden_lora` + `--hidden-lora-layers-to-transform N` | 减少层数的 LoRA（对照） | Claim 3 |

日志速记：`combined_in` = affine_input_plus_hidden_lora，`combined_inlmh` = affine_input_lm_head_plus_hidden_lora

### 四条主张与关键结果

**Claim 1a：AffLoRA + hidden LoRA 优于纯 hidden LoRA** ✅

| 模型 | hidden_lora_r8 | combined_inlmh | Delta |
|---|---|---|---|
| Qwen2.5-1.5B（三 seed 平均） | 1.054 | 1.048 | **-0.006** |
| Qwen2.5-0.5B | 1.302 | 1.294 | -0.008 |
| Qwen3-1.7B | 0.7383 | 0.7350 | -0.0033 |
| Qwen2.5-3B | 0.9797 | 0.9762 | -0.0035 |
| Qwen3-4B | 0.8828 | 0.8810 | -0.0018 |
| Qwen2.5-7B | 0.9180 | 0.9151 | -0.0029 |
| Qwen3-8B | 0.8717 | 0.8696 | -0.0021 |

**Claim 1b：AffLoRA 比 vocab 维 LoRA 更参数高效** ✅ —— 1.5B 上 100k AffLoRA 优于 307k vocab-dim LoRA r=1；7B/8B 上 AffLoRA 用更少参数持续优于

**Claim 2：AffLoRA-only 明显优于 frozen base** ✅ —— Qwen2.5-1.5B 从 1.926 降至 1.326（Δ -0.600, 100k 参数）；Qwen3-0.6B 从 1.848 降至 1.320（Δ -0.528, 67k 参数）

**Claim 3：AffLoRA 优于匹配预算的单层 LoRA** ✅ —— Qwen3-0.6B 上 33k AffLoRA 达到 1.426，33k-49k 单层 LoRA 为 1.473-1.780

**Phase 9d 位置归因**：低 rank 下收益主要来自输出侧（lm_head）——input-only 增益很小

### 数据与模型

- 主任务：`data/sft_t2t_mini_25k/`（24k train + 1k eval），通用中文 SFT
- 数据格式：JSON/JSONL，支持 `conversations`（role/content）或平面 `question`/`answer` 字段
- 所有 Qwen 模型均为 tied embedding（`embed_tokens` 与 `lm_head` 共享权重）

### 超参与训练

```bash
cd /home/wz/projects/mypro/im_exp/lora
export PYTHONPATH=/home/wz/projects/mypro/im_exp/lora/src:${PYTHONPATH:-}
source ../set

# 主训练（单 GPU，头条超参）
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

关键超参：max_seq_len=1024, effective batch=16, lr=2e-4, cosine warmup 3%, 1 epoch, AffLoRA rank=16/alpha=128, hidden LoRA rank=8/alpha=16

### ⚠️ fp32 精度陷阱（关键）

**永远不要在未设置 `--master-dtype fp32` 的情况下训练。** bf16 master 权重 + bf16 AdamW 会静默地将 lr=2e-4 的小更新舍入为零。正确标志：`--master-dtype fp32 --base-dtype auto --bf16`。训练脚本会在开始和结束时分别打印 trainable 参数 dtype 和 Adam state dtype——两者都应显示 fp32。

### 代码结构

- `src/affine_vocab_lora/adapter.py`：核心库（AffineVocabConfig、LowRankAffineMap、AffineEmbedding、AffineLMHead、apply/save/load）
- `scripts/train_affine_vocab_lora.py`：单 GPU HuggingFace Trainer 入口
- `scripts/eval_base_loss.py`：frozen base 评估（Claim 2 参照）
- `scripts/eval_merge_equivalence.py`：tied 适配器合并等价性验证
- `outputs/affine_vocab/`：训练产物，按阶段/运行组织

---

## PTB Embedding Relaxation 复现

项目位置：`/home/wz/projects/mypro/im_exp/better-wt-relaxation`

复现 Press & Wolf (EACL 2017) "Using the Output Embedding to Improve Language Models" 的 Penn Treebank word-level LSTM 实验，并加入 S1-S21 embedding relaxation 变体研究。

### S1-S21 变体定义

**S1-S13（主体族）**，低秩 rank=r, relaxation_scale=s, V=10000, d=1500：

| ID | 描述 | 公式 | 额外参数 (r=8) |
|---|---|---|---|
| S1 | Tied（=WT） | `E = U = W` | 0 |
| S2 | Untied（=baseline） | 独立 E 和 U | 15,000,000 |
| S3 | 输入侧加性 bias | `W - s·1·β^T` | 1,500 |
| S4 | 输入侧加性低秩 | `W + s·A·B` | 92,000 |
| S5 | 输出侧加性低秩 | `W + s·A·B`（输出侧） | 92,000 |
| S6 | 输入侧乘性低秩 | `W·(I + s·P·Q)` | 24,000 |
| **S7** | **输出侧乘性低秩** | `W·(I + s·P·Q)`（输出侧） | **24,000** |
| S8-S10 | Untied + 输入侧各变体 | — | — |
| S11 | 输出侧 hidden 平移 | `logits = (h + s·β)·W^T` | 1,500 |
| S12 | 输入 shift + mult 组合 | S3 + S6 | 25,500 |
| S13 | 输入 add + mult 组合 | S4 + S6 | 116,000 |

**S14-S21（双侧/输出组合族）**：在输入侧和输出侧同时应用不同 relaxation，如 S14=S4+S5, S15=S6+S7, S16=S4+S7, S17=S13+S7, S19=S7+S11 等。

### 实验配置

| 配置 | 架构 | Hidden | Batch | Epochs | 用途 |
|---|---|---|---|---|---|
| `small` | LSTM | 200 | 20 | 13 | 复现验证 |
| `large` | LSTM | 1500 | 20 | 55 | 论文级严格设定 |
| `bayes1500` | Variational LSTM | 1500 | 20 | 55 | Bayesian dropout |
| `large4090` | LSTM | 1500 | 128 | 35 | 快速筛选（不可与论文直接对比） |

### 复现质量

| Run | 我们的 Test PPL | 论文 Test PPL |
|---|---|---|
| large baseline | 78.90 | 78.4 |
| large wt | 74.59 | 74.3 |
| bayes1500 wt + wd 1e-7 | 73.26 | 73.2 |

### 核心发现：S7 输出侧乘性低秩

**一句话结论**：用 `W → W(I + s·P·Q)` 替换 weight tying，rank=32, scale=0.03，仅增加 96K (0.19%) 参数，将 5-seed mean test PPL 从 74.79±0.19 (WT) 降至 73.21±0.18（Δ -1.58 PPL, t=14, p<1e-6）。

**S7 rank-scale 帕累托前沿**：

| Rank | Scale | 参数 | Test PPL (mean±std) | vs WT |
|---|---|---|---|---|
| 8 | 0.03 | 24K (0.05%) | 73.76±0.19 | -1.03 |
| 16 | 0.03 | 48K (0.10%) | 73.42±0.23 | -1.36 |
| **32** | **0.03** | **96K (0.19%)** | **73.21±0.18** | **-1.58** ★ |
| 256 | 0.03 | 768K (1.51%) | 73.15±0.39 | -1.64 |

> ★ = 综合最优（mean × std × params），t=14, p<1e-6

**结构特异性是关键**：100+ 多 seed 论文级测量中，**所有输入侧变体**（S3/S4/S6/S12/S13 及 untied 族 S8-S10）均无法稳定超越 WT。最好的输入侧是 S6 r=8 s=0.01 仅低 0.16 PPL（不显著，t~1.3）。而输出侧 S7 (-1.58) 和 S5 (-0.64) 均显著。

**为什么输出侧乘性有效**：S7 的扰动直接作用于 logit 层，绕过了 LSTM 35 步门控和循环对输入侧扰动的吸收/抹平。且 S7 的扰动耦合于 W 的列空间（乘性而非独立加性），参数效率更高（2dr=24K vs Vr+rd=92K for S5）。

**Rank 与 Scale 解耦规律**：
- **输出侧 (S7)**：rank 决定效果下限（越高越好），scale 决定稳定性（rank 升高时 scale 必须降低）
- **输入侧**：scale 决定是否有效（窄稳定带，两侧悬崖陡峭），rank 越高反而越差（r=16 一致地比 r=8 差 ~0.5 PPL）

**S13 陷阱**：S13 在快速筛选（large4090）中排名第一，但在论文级 `large` 中仅与 WT 持平（scale=0.1），scale=0.3 时直接发散至 PPL 618。large4090 的排名不能直接沿用。

**总计算量**：800+ 论文级 runs，约 60 GPU-hours，在 6 卡 A100 池上约 8-9 小时 wall-clock。

---

## 模型权重

路径：`/home/wz/projects/mypro/im_exp/models/`

供 lora/ 实验使用的 Qwen base 模型：

| 模型 | task6 R² | 用途 |
|---|---|---|
| `Qwen2.5-0.5B-Base` | 0.9903 | 快速 smoke test |
| `Qwen2.5-1.5B-Base` | **0.9997** | **头条主模型** |
| `Qwen2.5-3B-Base` | 0.9997 | 扩展 scale-up |
| `Qwen2.5-7B-Base` | — | 大模型验证 |
| `Qwen3-0.6B-Base` | 0.9877 | 快速超参扫描 |
| `Qwen3-1.7B-Base` | 0.9938 | 后续主模型 |
| `Qwen3-4B-Base` | 0.9901 | 扩展 scale-up |
| `Qwen3-8B-Base` | — | 大模型验证 |
| `emb/` | — | embedding 分析产物 |

task6 R² 是 base→instruct 全词表仿射拟合质量指标（越高说明仿射关系越强），来自外部分析。

所有 Qwen 模型均为 tied embedding（`embed_tokens` 与 `lm_head` 共享权重）。

---

## 论文产出

路径：`/home/wz/projects/mypro/im_exp/paper/`

**不参与实验流程**，纯对外交付物。

```
paper/
├── figures/       ← 论文用图（梯度对比、cosine 等 PNG/PDF）
└── submission/    ← EMNLP 2026 匿名投稿代码包
    ├── SUBMISSION_PACK_README.md
    ├── AGD_code_EMNLP2026_anonymous.zip    ← 论文 1 代码 + 补充材料
    ├── ALoRA_code_EMNLP2026_anonymous.zip  ← 论文 2 代码 + 补充材料
    ├── 1/           ← AGD 源代码（预训练端 untying）
    └── 2/           ← A-LoRA 源代码（后训练端 placement）
```

### 论文映射

| 标签 | 论文 | 对应实验线 | 文件夹 |
|---|---|---|---|
| Anonymous (2026a) | Efficient Untying … **AGD** | minimind（预训练架构改造） | `1/` |
| Anonymous (2026b) | Rethinking LoRA Placement … **A-LoRA** | lora（后训练适配） | `2/` |

两篇论文需要互相引用对方的 concurrent work，补充材料中已准备好交叉引用 LaTeX/bib。

### 投稿包详情

见 `paper/submission/SUBMISSION_PACK_README.md`，包含 originality checklist、交叉引用指南、zip 重建命令等。

---

## 常用检查

```bash
nvidia-smi
ls -lh /home/wz/projects/mypro/im_exp/minimind/out/*.pth
ls -lh /home/wz/projects/mypro/im_exp/minimind/logs/*summary*.csv
ls -lh /home/wz/projects/mypro/im_exp/lora/outputs/affine_vocab/
ls -lh /home/wz/projects/mypro/im_exp/better-wt-relaxation/runs/
```
