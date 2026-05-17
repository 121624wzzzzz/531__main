# PyTorch PTB Experiments

本目录是项目的 PyTorch 训练代码。运行入口和实验套件在仓库根目录的
[`scripts/`](../scripts) 下；模型/复现说明在 [`docs/`](../docs) 下。整体
项目地图见根 [README](../README.md)。

## 目标解释器

```powershell
D:\DevTools\anaconda3\envs\wzdt\python.exe
```

依赖见 [`requirements-wzdt.txt`](requirements-wzdt.txt)。

## 模块拆分

| 文件 | 职责 |
| --- | --- |
| [`train_ptb.py`](train_ptb.py) | CLI 参数解析、`--speed_mode` 预设、主流程编排 |
| [`configs.py`](configs.py) | `PTBConfig` 与 `CONFIGS`（small / medium / large / large4090 / bayes1500 / bayes4090 / test） |
| [`variants.py`](variants.py) | S1-S13 公式映射与 `EmbeddingVariant` 模块、`actual_extra_params()` |
| [`model.py`](model.py) | Zaremba 风格 LSTM 与 Variational LSTM，包含 dropout 与权重初始化 |
| [`train_loop.py`](train_loop.py) | epoch 循环、评估、checkpoint / JSON 持久化 |
| [`ptb_data.py`](ptb_data.py) | PTB 词表构建与 `PTBBatchedSplit` batch 迭代器 |

入口命令保持不变：

```powershell
& "D:\DevTools\anaconda3\envs\wzdt\python.exe" .\repro_pytorch\train_ptb.py `
  --data_path .\data\ptb `
  --model small `
  --variant wt `
  --device auto
```

## 实验家族

- `small`, `medium`, `large`：Zaremba-style PTB LSTM。
- `large4090`：与 `large` 同结构，但更大 batch、更短 schedule，仅做本地筛选。
- `bayes1500` / `bayes4090`：variational LSTM，用于复现 Gal / Bayesian dropout
  行，是旧 Lua/Torch7 BayesianRNN 脚本的现代替代。
- `baseline` / `wt` / `pr` / `wt_pr`：原文核心四种 embedding 共享方式。
- `s1` … `s13`：embedding relaxation 变体，所有 S 变体支持
  `--lora_rank` 与 `--relaxation_scale`。

完整变体定义、表格与公式见 [`docs/experiment-design.md`](../docs/experiment-design.md)
与 [`docs/paper-main-experiments.md`](../docs/paper-main-experiments.md)。

## 推荐运行

冒烟测试：

```powershell
.\scripts\run_smoke.ps1
```

4090 本地：

```powershell
.\scripts\run_ptb_main.ps1 -Model large4090 -Variant wt
.\scripts\run_ptb_main.ps1 -Model bayes4090 -Variant wt -LegacyWeightDecay 0
.\scripts\run_ptb_main.ps1 -Model bayes4090 -Variant wt -LegacyWeightDecay 1e-7
```

S1-S13 sweep：

```powershell
.\scripts\run_embedding_variants.ps1 `
  -Model large4090 `
  -LoraRanks 1,2,4,8,16,32 `
  -RelaxationScales 0.1,0.3,0.5,0.7,1.0,1.5,2.0
```

`-RelaxationScales` 控制非基础松弛分支（如 `AB`、`WPQ`）的乘子网格；
多个 S 变体在 `1.0` 不稳定但在 `0.1` 表现良好，请在结果表中始终记录
`rank` 与 `relaxation_scale`。

完整本地套件：

```powershell
.\scripts\run_all_experiments.ps1 `
  -LoraRanks 1,2,4,8,16,32 `
  -RelaxationScales 0.1,0.3,0.5,0.7,1.0,1.5,2.0
```

paper-scale 单独行：

```powershell
.\scripts\run_ptb_main.ps1 -Model bayes1500 -Variant wt -LegacyWeightDecay 0
.\scripts\run_ptb_main.ps1 -Model bayes1500 -Variant wt -LegacyWeightDecay 1e-7
```

paper PTB 复现套件：

```powershell
.\scripts\run_paper_ptb_experiments.ps1 -Tf32
```

`run_paper_ptb_experiments.ps1` 默认跳过已完成的 `runs/paper-*` 目录；
传 `-Force` 强制重跑。`-Tf32` 仅启用 TF32 矩阵乘，其它训练设定保持不变。
