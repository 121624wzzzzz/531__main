# Performance and Speed Modes

本文档定义训练脚本支持的几种加速口径，对应的 CLI 用法，以及当前 RTX 4090 上
的基准计时。所有 paper 复现实验默认仍走严格等价路径；现代加速选项只在文档
里明确登记后才进入正式表格。

主实验定义与对照表见 [paper-main-experiments.md](paper-main-experiments.md)。
研究路线和阶段安排见 [experiment-design.md](experiment-design.md)。

## 1. 三个等价层级

| 层级 | 含义 | 是否可直接对照原文表格 |
| --- | --- | --- |
| strict-equivalent | 与旧代码 bit-for-bit / 数值一致：仅重排了不影响数值的步骤 | 是 |
| approximate-equivalent | 数学目标不变，但允许 TF32、cuDNN 非确定性、编译器改变归约顺序 | 通常是，需要登记开关 |
| screening-only | 改 batch size、epoch 数或 schedule，只用于快速候选筛选 | 否 |

`run_paper_ptb_experiments.ps1` 的 `-Tf32` 开关属于 approximate-equivalent；
`large4090` / `bayes4090` 配置属于 screening-only，本文档不重复定义。

## 2. 已落地的 strict-equivalent 优化

这些改动已经合入 `train_ptb.py` 和 `ptb_data.py`，无需任何 CLI 开关：

- `ptb_data.PTBBatchedSplit`：每个 split 只构建一次 batch layout，并在 GPU 设备上
  常驻；epoch 之间不再重复 `np.array(list(raw_data))` 和按行复制。切片顺序与
  旧 `ptb_iterator` 完全一致。
- `run_epoch` 的损失累加改在 GPU 张量上完成，仅在日志点和 epoch 末同步一次到
  CPU。perplexity 仍按 `exp(sum_loss / total_tokens)` 计算，与旧实现一致。
- 每个 epoch 结束后追加输出 `train_sec`、`valid_sec`、`train_wps`，
  并写入 JSON 指标里（`epoch_N_train_sec`、`epoch_N_train_wps`、`timing_*`）。

验证：

- `small baseline` 第 1 轮 `train_ppl=282.988`，与旧版 `runs/paper-small-baseline`
  JSON 中的 `282.987...` 在 3 位小数内一致，且与论文 small baseline test PPL `114.5`
  对得上。详细对照见 [paper-main-experiments.md](paper-main-experiments.md) 第 6 节。

## 3. CLI 速度开关

`train_ptb.py` 新增以下参数：

| 参数 | 作用 | 数值影响 |
| --- | --- | --- |
| `--tf32` | 允许 cuBLAS / cuDNN 使用 TF32 矩阵乘 | 末位浮点差异 |
| `--cudnn_benchmark` | 设置 `torch.backends.cudnn.benchmark=True` | 算法选择非确定性 |
| `--compile` | 使用 `torch.compile` 包装模型 | 编译器可能改变归约顺序 |
| `--compile_mode` | `default` / `reduce-overhead` / `max-autotune` | 同上 |
| `--log_every` | 训练阶段每多少 batch 同步并打印一次 perplexity | 仅影响日志频率，最终 PPL 不变 |
| `--speed_mode` | 预设：`strict-fp32` 或 `modern-fast` | 见下表 |

预设展开后的等价命令：

| Preset | 等价开关组合 |
| --- | --- |
| `--speed_mode strict-fp32` | 不主动启用任何加速；显式记录"严格 FP32"意图 |
| `--speed_mode modern-fast` | `--tf32 --cudnn_benchmark` |

预设永远不会关闭用户显式开启的开关：如果同时传 `--speed_mode strict-fp32 --tf32`，
TF32 仍然生效，但 JSON 中会同时记录两个字段，方便后续检查。

## 4. 推荐运行口径

### 4.1 严格 FP32（对照原文最终值）

```powershell
.\scripts\run_ptb_main.ps1 -Model large -Variant baseline
.\scripts\run_ptb_main.ps1 -Model large -Variant wt
```

或显式：

```powershell
& "D:\DevTools\anaconda3\envs\wzdt\python.exe" .\repro_pytorch\train_ptb.py `
  --data_path .\data\ptb `
  --model large `
  --variant baseline `
  --speed_mode strict-fp32 `
  --device auto `
  --output_dir .\runs\paper-large-baseline
```

适用场景：

- 与论文表 5 / 表 6 进行最终对照。
- 复核数据加载、loss、tying、PR 等核心代码路径。

### 4.2 Modern-fast（日常开发与候选筛选）

```powershell
& "D:\DevTools\anaconda3\envs\wzdt\python.exe" .\repro_pytorch\train_ptb.py `
  --data_path .\data\ptb `
  --model large `
  --variant wt `
  --speed_mode modern-fast `
  --device auto `
  --output_dir .\runs\modernfast-large-wt
```

适用场景：

- 跑 S1-S13 变体筛选。
- 修改代码后快速校验是否仍能复现 PTB 趋势。
- 不直接与原文最终数字混合报告，必须在表头注明开启了 TF32 / benchmark。

### 4.3 Bayesian 行的现有约定

`run_paper_ptb_experiments.ps1 -Tf32` 已经对 Bayesian rows 启用 TF32。该口径
与本文档 modern-fast 的精度处理一致，差别只在脚本一次性跑完整组 paper rows。
未来如需对 Bayesian 行单独跑严格 FP32，去掉 `-Tf32` 即可。

## 5. RTX 4090 当前基准

以下数据用新代码就地测量，全部 `device=auto`，本机为 RTX 4090，
`max_epochs=1`，按 `max_train_batches` 截断。单位：`train_sec` 是训练阶段墙钟，
`wps` 是 `tokens_processed / train_sec`。

### 5.1 数据加载与同步重构带来的端到端时间

| Run | Config | Batches | Train sec | wps | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| small baseline | `small` | 2326 × 2 epochs | 7.5 / epoch | 125,000 | 与旧 JSON PPL 在 3 位内一致 |
| large baseline | `large` | 200 | 4.3 | 32,400 | 全 epoch 推算约 28-30 s |
| bayes1500 baseline | `bayes1500 --tf32` | 50 | 7.4 | 4,750 | 全 epoch 推算约 200 s |

`bayes1500` 是绝对瓶颈：variational 分支按时间步 Python 循环，无法走 cuDNN
融合 LSTM 内核。这正是后续优化（编译、自定义 cell）的主要目标。

### 5.2 修改开关在 large baseline 上的对照

同一 `large baseline`，`max_train_batches=300`，`max_eval_batches=100`：

| Run | Train sec | wps | Train PPL | Notes |
| --- | ---: | ---: | ---: | --- |
| 严格 FP32 | 6.0 | 35,200 | 1599.012 | 参考 |
| `--tf32` | 9.9 | 21,200 | 1363.387 | 数值有变化，短窗 |
| `--tf32 --cudnn_benchmark` | 11.0 | 19,200 | 1363.387 | 与上一行数值一致 |

观察：

- 短窗内 TF32 在标准 LSTM 路径上没有加速；cuDNN LSTM 在 FP32 已经接近峰值。
- 短窗内 `cudnn_benchmark` 仅增加了 autotune 开销；满 55 epoch 的实际收益需要
  一次完整 paper-scale 验证才有意义。
- 两次 TF32 运行的 `train_ppl` 完全一致，说明 `cudnn_benchmark` 的非确定性
  在这段轨迹上没有引发可见漂移。
- 短窗里 FP32 与 TF32 的 `train_ppl` 不同是预期的：300 batches 还在剧烈下降段，
  末位浮点差异被放大；55 epoch 收敛后的 valid / test PPL 才有可比性。

### 5.3 `torch.compile` 在 Windows 上的现状

```text
torch._inductor.exc.TritonMissing: Cannot find a working triton installation.
```

当前 Windows 上的 `wzdt` 环境没有可用的 Triton，`--compile` 会在第一次前向时
失败。需要在 Linux/WSL 或安装 `triton-windows` 后再启用。文档保留该开关用于
将来在支持的环境上做对照。

## 6. 验收与回归标准

任何 strict-equivalent 优化合入后，至少需要满足：

1. `run_smoke.ps1` 标准与 variational 两个分支都能跑通。
2. `small baseline` 第 1 轮 `train_ppl` 与历史 `runs/paper-small-baseline/*.json`
   第一轮值在 3 位小数内一致。
3. 任何 paper row 重跑后，test PPL 漂移不超过 `0.05`，且 weight tying 的趋势保留。
4. `timing_avg_train_wps` 较优化前有可观测提升；否则需要回滚并复盘。

任何 approximate-equivalent 选项（TF32、cudnn_benchmark、compile）首次启用时，
应在 README 或本文件第 5 节追加一行实测数据，明确标注开关组合与 PPL 漂移。

## 7. 后续可以继续做的事

下面这些方向还没动，留作下一阶段任务：

- `VariationalDropoutLSTMModel` 内的 mask 生成与每步 `bmm` 拼接：尝试 `torch.compile`
  作用在 cell 上，或者把 Python 时间步循环改成 fused custom cell。
- 在 Linux/WSL 环境上重跑 5.2 表，验证 TF32 + cudnn_benchmark 在满 epoch 下
  是否能在 standard LSTM 上获得明显速度收益。
- 在 `run_paper_ptb_experiments.ps1` 中加入跳过策略以外的并行选项（多 GPU 时）。
- 长期可考虑 `bfloat16` autocast，与论文比对前需要单独 ablation。
