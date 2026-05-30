# Experiment Design & Paper Reference

本文档合并了原 `paper-main-experiments.md` 和 `experiment-design.md`：
前半部分记录原论文配置和复现检查口径；后半部分定义本项目在 tied embedding
附近设计的 S1-S21 松弛变体、实验阶段和决策规则。

最终结果见 [`results.md`](results.md)。

## 1. 原论文参考

论文：Ofir Press and Lior Wolf. 2017. *Using the Output Embedding to Improve
Language Models*. EACL 2017. <https://aclanthology.org/E17-2025/>

原文核心主张：在 RNN language model 中，把输入 embedding 和输出 softmax
weight 绑定 (weight tying) 可以提升泛化并减少参数量。本项目在此基础上研究
是否可以通过轻量"松弛"进一步改进。

### 1.1 数据

Mikolov PTB word-level，`data/ptb/`：

| Split | Tokens |
|---|---|
| Train | 929,589 |
| Valid | 73,760 |
| Test | 82,430 |
| Vocabulary | 10,000 |

### 1.2 原论文模型配置

**Small Model**（Zaremba et al., 2014）：

| Item | Value |
|---|---|
| Architecture | 2-layer LSTM, hidden=200 |
| Sequence length / Batch size | 20 / 20 |
| Dropout | None |
| LR / Grad clip | 1.0 / 5.0 |
| Init scale | 0.1 |
| Schedule | 13 epochs, decay after epoch 4, decay=0.5 |

CLI: `--model small`

**Large Model**（Zaremba et al., 2014）：

| Item | Value |
|---|---|
| Architecture | 2-layer LSTM, hidden=1500 |
| Sequence length / Batch size | 35 / 20 |
| Dropout keep prob | 0.35 |
| LR / Grad clip | 1.0 / 10.0 |
| Init scale | 0.04 |
| Schedule | 55 epochs, decay after epoch 14, decay=1/1.15 |

CLI: `--model large`

**Bayesian Dropout Large**：

| Item | Value |
|---|---|
| Architecture | variational LSTM, hidden=1500 |
| Dropout | x=0.3, i=0.5, h=0.3, o=0.5 |
| Weight decay | 1e-7 |
| Schedule | 55 epochs, decay after epoch 10, decay=1/1.15 |

CLI: `--model bayes1500 --legacy_weight_decay 1e-7`

### 1.3 原论文 PTB 主结果

| Model | Size | Train | Valid | Test |
|---|---|---|---|
| Large (Zaremba) | 66M | 37.8 | 82.2 | 78.4 |
| Large + Weight Tying | 51M | 48.5 | 77.7 | 74.3 |
| Large + BD + WD | 66M | 24.3 | 78.1 | 75.2 |
| Large + BD + WT | 51M | 28.2 | 75.8 | 73.2 |

| Model | Size | Train | Valid | Test |
|---|---|---|---|
| Small | 4.65M | 38.0 | 120.7 | 114.5 |
| Small + WT | 2.65M | 36.4 | 117.5 | 112.4 |
| Small + PR | 4.69M | 50.8 | 116.0 | 111.7 |
| Small + WT + PR | 2.69M | 53.5 | 104.9 | 100.9 |

### 1.4 复现检查清单

每次修改训练代码后建议按序检查：

1. 跑 smoke test（标准 + variational 分支不报错）
2. 跑 `small baseline` + `small wt`，确认 WT 改善
3. 如改 loss/batching/state/dropout/embedding/softmax，重跑 `large baseline`
4. 如改 tied embedding/output projection/BD，重跑 `large wt` 和 `bayes1500 wt wd1e-7`
5. 不要用 `large4090` 的绝对 PPL 对照原文

---

## 2. 本项目扩展：S1-S21 Embedding Relaxation

### 2.1 研究目标

1. 复现 PTB word-level LM 中 output embedding tying 的收益
2. 在 tied embedding 附近设计 S1-S21 松弛结构，寻找更高效的参数化
3. 诊断额外模块的训练稳定性（LoRA-like scale、乘法分支、输出侧分支）
4. 对有希望的结构做 paper-scale 多 seed 复现

### 2.2 模型家族

| Model | Architecture | Hidden | Layers | Batch | Steps | 用途 |
|---|---|---|---|---|---|---|
| `small` | LSTM | 200 | 2 | 20 | 20 | 基础 sanity check |
| `medium` | LSTM | 650 | 2 | 20 | 35 | 可选桥接 |
| `large` | LSTM | 1500 | 2 | 20 | 35 | Paper-scale 对照 |
| `bayes1500` | variational LSTM | 1500 | 2 | 20 | 35 | Bayesian dropout 对照 |
| `large4090` | LSTM | 1500 | 2 | 128 | 35 | 快速筛选（不可对照原文）|
| `bayes4090` | variational LSTM | 1500 | 2 | 64 | 35 | 快速 variational 筛选 |
| `rhn830` | RHN | 830 | 1 | 20 | 35 | RHN + BD paper |
| `gpt512` | Transformer | 512 | 4 | 128 | 128 | WikiText-2 probe |

### 2.3 原论文变体

| Variant | 含义 |
|---|---|
| `baseline` | Untied input E 与 output U |
| `wt` | Weight tying (E = U = W) |
| `pr` | Projection regularization (untied) |
| `wt_pr` | Weight tying + projection regularization |

### 2.4 S1-S21 松弛变体定义

所有 S 变体支持 `--lora_rank <r>` 和 `--relaxation_scale <s>`。

**S1-S13（主族）：**

| ID | 描述 | Extra params (V=10000,d=1500,r=8) |
|---|---|---|
| S1 | Tied (≡ WT) | 0 |
| S2 | Untied (≡ baseline) | 15,000,000 |
| S3 | Input shift: W − s·1βᵀ | 1,500 |
| S4 | Input additive low-rank: W + s·AB | 92,000 |
| S5 | Output additive low-rank: W + s·AB | 92,000 |
| S6 | Input multiplicative low-rank: W(I + s·PQ) | 24,000 |
| S7 | Output multiplicative low-rank: W(I + s·PQ) | 24,000 |
| S8 | Untied + input shift | 15,001,500 |
| S9 | Untied + input additive low-rank | 15,092,000 |
| S10 | Untied + input multiplicative low-rank | 15,024,000 |
| S11 | Output hidden shift: logits = (h + s·β)Wᵀ | 1,500 |
| S12 | Input shift + multiplicative | 25,500 |
| S13 | Input additive + multiplicative | 116,000 |

**S14-S21（Tied 双侧/输出复合族）：**

| ID | Input effective | Output effective | 对照 |
|---|---|---|---|
| S14 | W + sAB | W + sCD | S4 + S5 |
| S15 | W(I+sPQ) | W(I+sRS) | S6 + S7 |
| S16 | W + sAB | W(I+sPQ) | S4 + S7 |
| S17 | W + sAB + sWPQ | W(I+sRS) | S13 + S7 |
| S18 | W(I+sPQ) − sβ | W(I+sRS) | S12 + S7 |
| S19 | W | W(I+sPQ) on h+sβ | S7 + S11 |
| S20 | W(I+sPQ) − sβ | W(I+sRS)(h+sγ) | S12 + S19 |
| S21 | W | W + sAB + sWPQ | S5 + S7 |

**变体结构对比（核心四种）：**

| 变体 | 公式 | 与 W 耦合 | 扰动路径 |
|---|---|---|---|
| S4 (input add) | embed(x) = W[x] + s·A[x]B | 独立 | 穿 LSTM × 35 步 + dropout |
| S5 (output add) | W' = W + s·AB | 独立 | 直达 logits |
| S6 (input mult) | embed(x) = W[x](I + sPQ) | 同子空间 | 穿 LSTM × 35 步 + dropout |
| **S7 (output mult)** | W' = W(I + sPQ) | **同子空间** | **直达 logits** |

### 2.5 实验阶段

**Phase 0 — Smoke test：**
```bash
CUDA_VISIBLE_DEVICES=1 $PY repro_pytorch/train_ptb.py \
  --data_path data/ptb --model test --variant baseline \
  --device cuda:0 --max_epochs 1 --max_train_batches 5 \
  --max_eval_batches 5 --metrics_only --output_dir runs/_smoke
```

**Phase 1 — Paper small 复现：**
```bash
$PY scripts/a100_jobs.py gen paper-small --out runs/jobs/paper-small.json
$PY scripts/a100_run.py runs/jobs/paper-small.json --gpus 1,2,3,4
```

**Phase 2 — S1-S13 全量 rank × scale 筛选（large4090）：**
```bash
$PY scripts/a100_jobs.py gen screen-large4090 --out runs/jobs/screen-large4090.json
$PY scripts/a100_run.py runs/jobs/screen-large4090.json --gpus 1,2,3,4,5,6
```

**Phase 3 — 低 scale 追加 + S14-S17 双侧筛选：**
```bash
$PY scripts/a100_jobs.py gen screen-large4090-lowscale --out runs/jobs/screen-lowscale.json
$PY scripts/a100_jobs.py gen screen-bilateral-tied --out runs/jobs/screen-bilateral-tied.json
$PY scripts/a100_run.py runs/jobs/screen-lowscale.json --gpus 1,2,3,4,5,6
$PY scripts/a100_run.py runs/jobs/screen-bilateral-tied.json --gpus 1,2,3,4,5,6
```

**Phase 5 — Paper-scale 确认（large, 55 epoch）：**
候选配置从 Phase 2/3 的筛选最佳中选出。

**Phase 6 — Multi-seed 稳健性（5 seed）：**
```bash
$PY scripts/a100_run.py runs/jobs/multi-seed.json --gpus 1,2,3,4,5,6
```

**Phase D-J — rank × scale 解耦 + 输入侧深扫：**
详见 `results.md` §6-7 及相关 job JSON。

### 2.6 决策规则

1. Variant 在 scale=1.0 差但在 scale=0.1 好 → 标记为 scale-sensitive
2. Variant 在 scale=0.01 仍爆炸 → 标记为 unstable under current optimizer
3. 候选必须在 `large4090` 上优于 S1 才值得 paper-scale 计算
4. 候选必须在 paper-scale `large` 上优于 S1 才算真正改善
5. 最终声明必须包含 parameter delta

### 2.7 报告模板

| Model | Variant | Rank | Scale | Seed | Extra params | Best valid | Test | Output dir |
|---|---|---|---|---|---|---|---|---|
| large4090 | s4 | 8 | 0.1 | 1 | 92K | 84.03 | 80.64 | `runs/large4090-s4-r8-scale0p1` |

Short interpretation format:
```
Question:
Setup:
Main result:
Comparison to S1/S2:
Stability notes:
Next action:
```

### 2.8 已知风险

- `large4090` 排名可能与 paper-scale `large` 不一致（如 S13：筛选第1 → paper-scale 中等）
- 输出侧和乘性分支对 relaxation_scale 敏感
- 当前脚本默认单 seed，多 seed 需手动传 `--seed`
- Checkpoint 体积大（large ~250MB），`runs/` 不入 Git

---

## 3. 复现命令速查

```bash
PY=/home/wz/anaconda3/envs/torch24/bin/python
cd /home/wz/projects/mypro/im_exp/UsingTheOutputEmbedding-repro

# 论文复现
$PY scripts/a100_jobs.py gen paper-small  --out runs/jobs/paper-small.json
$PY scripts/a100_jobs.py gen paper-large  --out runs/jobs/paper-large.json
$PY scripts/a100_jobs.py gen paper-bayes  --out runs/jobs/paper-bayes.json
$PY scripts/a100_run.py runs/jobs/paper-small.json --gpus 1,2,3,4
$PY scripts/a100_run.py runs/jobs/paper-large.json --gpus 1,2
$PY scripts/a100_run.py runs/jobs/paper-bayes.json --gpus 3,4

# S1-S13 筛选 + 低 scale 追加
$PY scripts/a100_jobs.py gen screen-large4090          --out runs/jobs/screen-large4090.json
$PY scripts/a100_jobs.py gen screen-large4090-lowscale --out runs/jobs/screen-lowscale.json
$PY scripts/a100_run.py runs/jobs/screen-large4090.json --gpus 1,2,3,4,5,6
$PY scripts/a100_run.py runs/jobs/screen-lowscale.json   --gpus 1,2,3,4,5,6

# 汇总
$PY scripts/a100_summarize.py \
  --csv runs/summaries/all_runs.csv \
  --md  runs/summaries/all_runs.md \
  --best_md runs/summaries/best_per_variant.md
```

完整调度说明见 [`scripts/README.md`](../scripts/README.md)，加速与计时见
[`performance.md`](performance.md)。
