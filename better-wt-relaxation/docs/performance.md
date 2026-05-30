# Performance and Speed Modes

本文档定义训练脚本支持的几种加速口径、对应的 CLI 用法，以及当前
**Linux + A100-SXM4-80GB** 上的基准计时。所有 paper 复现实验默认仍走严格
等价路径；现代加速选项只在文档里明确登记后才进入正式表格。

主实验定义与对照表、研究路线和阶段安排见 [experiments.md](experiments.md)。

## 1. 三个等价层级

| 层级 | 含义 | 是否可直接对照原文表格 |
| --- | --- | --- |
| strict-equivalent | 与旧代码 bit-for-bit / 数值一致：仅重排了不影响数值的步骤 | 是 |
| approximate-equivalent | 数学目标不变，但允许 TF32、cuDNN 非确定性、编译器改变归约顺序 | 通常是，需要登记开关 |
| screening-only | 改 batch size、epoch 数或 schedule，只用于快速候选筛选 | 否 |

`--tf32` 属于 approximate-equivalent；`large4090` / `bayes4090` 配置属于
screening-only，本文档不重复定义。

## 2. 已落地的 strict-equivalent 优化

这些改动已经合入 `train_ptb.py` 和 `ptb_data.py`，无需任何 CLI 开关：

- `ptb_data.PTBBatchedSplit`：每个 split 只构建一次 batch layout，并在 GPU 设备上
  常驻；epoch 之间不再重复 `np.array(list(raw_data))` 和按行复制。切片顺序与
  旧 `ptb_iterator` 完全一致。
- `run_epoch` 的损失累加改在 GPU 张量上完成，仅在日志点和 epoch 末同步一次到
  CPU。perplexity 仍按 `exp(sum_loss / total_tokens)` 计算，与旧实现一致。
- 每个 epoch 结束后追加输出 `train_sec`、`valid_sec`、`train_wps`，
  并写入 JSON 指标里（`epoch_N_train_sec`、`epoch_N_train_wps`、`timing_*`）。

## 3. CLI 速度开关

`train_ptb.py` 提供的与速度相关的参数：

| 参数 | 作用 | 数值影响 |
| --- | --- | --- |
| `--tf32` | 允许 cuBLAS / cuDNN 使用 TF32 矩阵乘 | 末位浮点差异 |
| `--cudnn_benchmark` | 设置 `torch.backends.cudnn.benchmark=True` | 算法选择非确定性 |
| `--compile` | 使用 `torch.compile` 包装模型 | 编译器可能改变归约顺序 |
| `--compile_mode` | `default` / `reduce-overhead` / `max-autotune` | 同上 |
| `--log_every` | 训练阶段每多少 batch 同步并打印一次 perplexity | 仅影响日志频率，最终 PPL 不变 |
| `--speed_mode` | 预设：`strict-fp32` 或 `modern-fast` | 见下表 |

| Preset | 等价开关组合 |
| --- | --- |
| `--speed_mode strict-fp32` | 不主动启用任何加速；显式记录"严格 FP32"意图 |
| `--speed_mode modern-fast` | `--tf32 --cudnn_benchmark` |

预设永远不会关闭用户显式开启的开关：同时传 `--speed_mode strict-fp32 --tf32`
时 TF32 仍然生效，但 JSON 中会同时记录两个字段。

## 4. A100 基准计时

以下数字在本机 A100-SXM4-80GB / torch 2.4.1+cu124 上实测，单卡，
`--device cuda:0` + `CUDA_VISIBLE_DEVICES=N` 绑定。

### 4.1 单 epoch 训练耗时与吞吐

| Model | Variant | Batch | Steps | TF32 | train_sec/epoch | wps | 全程估算 |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| `small` | baseline | 20 | 20 | off | 21 | ~22k | 13 epoch ≈ 4.7 min |
| `large` | wt | 20 | 35 | on | 15.7 | 59,134 | 55 epoch ≈ 14 min |
| `large4090` | s1 | 128 | 35 | on | 4.4 | 209,708 | 35 epoch ≈ 2.6 min |
| `bayes1500` | wt | 20 | 35 | on | 138.8 | 6,692 | 55 epoch ≈ 2.1 h |

观察：

- `large` 与 `large4090` 都跑 hidden=1500 LSTM；`large4090` 用 batch=128 后
  A100 上吞吐拉到 ~210k wps，正是其作为本地筛选配置的依据。
- `bayes1500` 仍然是绝对瓶颈：variational 分支按时间步 Python 循环，无法
  走 cuDNN 融合 LSTM 内核，所以每 epoch ~140 s，全程 2 h。
- `--tf32` 在 A100 上没有破坏 cuDNN LSTM 的稳定性；本仓库 paper-large、
  paper-bayes 均默认开启 TF32，并在 `runs/<run-dir>/ptb_*.json` 的 `args`
  字段中记录。

### 4.2 8 卡批量吞吐参考

以本仓库 `scripts/a100_run.py` 在 GPU 1-6 上跑 403 任务的实测（一次性
合并 paper-large + paper-bayes + screen-large4090 + screen-large4090-lowscale）：

```text
total wall          : ~5.3 h
sum of GPU-seconds  : ~80,000 (即 ~22 GPU-hour)
fail                : 2/403  (S7 在 scale 1.5/2.0 上数值发散，与设计文档一致)
```

并行因子约为 ~4，原因是 2 个 GPU 长期被 `bayes1500` 任务占住 2.1 h，
另外 4 个 GPU 在期间快速消耗了 ~300 个筛选任务，最后 6 卡一起把剩余 ~90 个
screening 任务跑完。

## 5. `torch.compile`

Linux + torch24 上 `--compile` 可用（已具备 Triton）。当前主线脚本未默认
开启；如需启用，参考：

```bash
CUDA_VISIBLE_DEVICES=1 $PY repro_pytorch/train_ptb.py \
  --data_path data/ptb --model large --variant wt --tf32 \
  --compile --compile_mode default \
  --device cuda:0 --output_dir runs/single/large-wt-compiled
```

注意：编译选项属于 approximate-equivalent，与 `--paper_test_eval` 合并使用前
需要单独 ablation。

## 6. 验收与回归标准

任何 strict-equivalent 优化合入后，至少需要满足：

1. 冒烟测试（`scripts/README.md` 中 smoke 命令）标准与 variational 两个分支都能跑通。
2. `small baseline` 第 1 轮 `train_ppl` 与历史一致（3 位小数内）。
3. 任何 paper row 重跑后，test PPL 漂移不超过 `0.05`，且 weight tying 的趋势保留。
4. `timing_avg_train_wps` 较优化前有可观测提升；否则需要回滚并复盘。

任何 approximate-equivalent 选项（TF32、cudnn_benchmark、compile）首次启用时，
应在本文档第 4 节追加一行实测数据，明确标注开关组合与 PPL 漂移。

## 7. 后续可以继续做的事

- `VariationalDropoutLSTMModel` 内的 mask 生成与每步 `bmm` 拼接：尝试 `torch.compile`
  作用在 cell 上，或者把 Python 时间步循环改成 fused custom cell；如果能从
  ~140 s/epoch 降到 ~30 s/epoch，paper-bayes 整组实验时间将从 2 h 降到 <30 min。
- 在 A100 上对 `--tf32 --cudnn_benchmark` 在标准 LSTM 上做完整 paper-scale
  对照，确认是否有显著速度收益（已在 `runs/paper-large/*` 中跑过 TF32 单种子
  baseline）。
- 长期可考虑 `bfloat16` autocast，与论文比对前需要单独 ablation。
