# PyTorch PTB Experiments

本目录是项目的 PyTorch 训练代码。批量调度脚本在仓库根目录的
[`scripts/`](../scripts) 下；模型/复现说明在 [`docs/`](../docs) 下。整体
项目地图见根 [README](../README.md)。

## 目标解释器

```text
/home/wz/anaconda3/envs/torch24/bin/python   # conda env "torch24"
```

激活：`source /home/wz/projects/mypro/im_exp/set`。依赖见
[`requirements.txt`](requirements.txt)：PyTorch 2.4.1+cu124、NumPy 1.26.4。

## 模块拆分

| 文件 | 职责 |
| --- | --- |
| [`train_ptb.py`](train_ptb.py) | CLI 入口、metadata 构建、流程编排 |
| [`configs.py`](configs.py) | `PTBConfig` 与 `CONFIGS`（small / medium / large / bayes1500 / rhn830 / gpt512 / test；large4090 / bayes4090 / rhn4090 由 `replace()` 派生） |
| [`variants.py`](variants.py) | `VariantSpec` 声明式注册表 + `EmbeddingVariant` 模块（新增变体只需改一处） |
| [`model.py`](model.py) | StandardLSTM、VariationalLSTM、TinyTransformerLM + `build_model()` |
| [`rhn.py`](rhn.py) | VariationalDropoutRHNModel（Recurrent Highway Network） |
| [`train_loop.py`](train_loop.py) | `run_epoch()`、`run_training()`、optimizer/scheduler 构建、checkpoint 持久化 |
| [`ptb_data.py`](ptb_data.py) | PTB/WikiText-2 词表构建与 `PTBBatchedSplit` 预缓存迭代器 |
| [`nn_utils.py`](nn_utils.py) | `inverted_bernoulli()` / `word_input_mask()`（model 与 rhn 共享） |

入口命令：

```bash
PY=/home/wz/anaconda3/envs/torch24/bin/python
CUDA_VISIBLE_DEVICES=1 $PY repro_pytorch/train_ptb.py \
  --data_path data/ptb \
  --model small \
  --variant wt \
  --device cuda:0 \
  --output_dir runs/single/small-wt
```

## 实验家族

- `small`, `medium`, `large`：Zaremba-style PTB LSTM。
- `large4090`：与 `large` 同结构，但更大 batch（128）、更短 schedule（35 epoch），
  仅做本地筛选；不可直接对照原文表格。
- `bayes1500` / `bayes4090`：variational LSTM，用于复现 Gal / Bayesian dropout 行。
- `baseline` / `wt` / `pr` / `wt_pr`：原文核心四种 embedding 共享方式。
- `s1` … `s21`：embedding relaxation 变体，其中 `s14-s21` 是 tied 双侧/输出复合 follow-up；所有 S 变体支持
  `--lora_rank` 与 `--relaxation_scale`。

完整变体定义、表格与公式见 [`docs/experiments.md`](../docs/experiments.md)。

## 推荐运行

### 单 GPU 单实验

绑卡时统一使用 `CUDA_VISIBLE_DEVICES=N` 选物理卡，`--device cuda:0` 选逻辑卡。

```bash
PY=/home/wz/anaconda3/envs/torch24/bin/python

# 筛选用 large4090（35 epoch，约 2.6 min/A100）
CUDA_VISIBLE_DEVICES=1 $PY repro_pytorch/train_ptb.py \
  --data_path data/ptb --model large4090 --variant s4 \
  --lora_rank 8 --relaxation_scale 0.1 --tf32 \
  --device cuda:0 --metrics_only \
  --output_dir runs/single/large4090-s4-r8-s0p1

# 论文对照 large（55 epoch，约 14 min/A100）
CUDA_VISIBLE_DEVICES=1 $PY repro_pytorch/train_ptb.py \
  --data_path data/ptb --model large --variant wt --tf32 \
  --paper_test_eval --device cuda:0 \
  --output_dir runs/single/paper-large-wt

# Bayesian large（55 epoch，约 2.1 h/A100）
CUDA_VISIBLE_DEVICES=1 $PY repro_pytorch/train_ptb.py \
  --data_path data/ptb --model bayes1500 --variant wt \
  --legacy_weight_decay 1e-7 --tf32 --paper_test_eval \
  --device cuda:0 --output_dir runs/single/paper-bayes1500-wt
```

### 批量并行（8×A100）

```bash
PY=/home/wz/anaconda3/envs/torch24/bin/python
$PY scripts/a100_jobs.py gen screen-large4090 --out runs/jobs/screen.json
$PY scripts/a100_run.py runs/jobs/screen.json --gpus 1,2,3,4,5,6
```

`--metrics_only` 是 sweep 时的推荐选项（不写 250MB checkpoint），脚本预设
已经按需开启。多个 S 变体在 `1.0` 不稳定但在 `0.1` 表现良好，请在结果表
中始终记录 `rank` 与 `relaxation_scale`。

完整调度说明见 [scripts/README.md](../scripts/README.md)。
