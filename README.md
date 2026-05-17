# UsingTheOutputEmbedding PTB 复现实验

本项目复现并扩展 Press &amp; Wolf, *Using the Output Embedding to Improve
Language Models* (EACL 2017) 中 Penn Treebank word-level language model 实验。
旧的 TensorFlow r0.8 / Lua Torch7 代码路径已经迁移到一份现代 PyTorch
实现，并加入 S1-S13 embedding relaxation 变体研究。

核心研究问题：

- 在 PTB word-level LSTM language model 中，输入 embedding 与输出 softmax
  weight 的不同共享方式（baseline / wt / pr / wt_pr）对验证 / 测试 perplexity
  的影响。
- 在 tied embedding 附近做局部松弛（S1-S13）能否在小幅增参的前提下进一步
  提升 perplexity，以及不同 LoRA-like rank 与 `relaxation_scale` 下的稳定性。

## 目录结构

```text
UsingTheOutputEmbedding/
  README.md                       项目入口与快速开始
  docs/                           复现说明与实验文档
    paper-main-experiments.md     原文主实验配置、结果与本地复现核对
    experiment-design.md          复现路线、模型族、S1-S13 变体、阶段计划
    performance.md                速度模式、CLI 开关与 RTX 4090 基准计时
    environment.md                本地环境与已验证命令
  repro_pytorch/                  现代 PyTorch 训练代码
    train_ptb.py                    CLI 入口与主流程编排
    configs.py                      模型配置（small/medium/large/...）
    variants.py                     S1-S13 embedding relaxation 模块与公式
    model.py                        Zaremba LSTM 与 Variational LSTM
    train_loop.py                   训练 / 评估循环与 checkpoint 持久化
    ptb_data.py                     PTB 词表与 batch 迭代器
    requirements-wzdt.txt           已验证的 conda 环境依赖
    README.md                       模块速览
  scripts/                        PowerShell / cmd 实验入口脚本
    run_smoke.ps1                   环境与代码路径冒烟测试
    run_ptb_main.ps1                单个实验的便捷封装
    run_embedding_variants.ps1      S1-S13 sweep
    run_paper_ptb_experiments.ps1   论文主实验复现套件
    run_core_experiments.ps1        small / 4090 核心对照
    run_all_experiments.ps1         核心 + S1-S13 完整本地套件
    scheduled_all_experiments.ps1   定时任务包装（含 dry-run）
    launch_all_experiments.ps1      简化启动器
    run_all_experiments_scheduled*.cmd  Windows 计划任务入口
  data/
    ptb/                            训练用 Penn Treebank 文本（默认数据路径）
    raw/                            原始打包资产（如 simple-examples.tgz）
  legacy/
    original/                       原始 Lua/Torch7 与 TF 实现（仅参考，非运行路径）
      BayesianRNN/
      ptb_word_lm/
  runs/                           训练结果：JSON 指标 / .pt checkpoint / 日志
```

`data/`、`runs/` 和 `simple-examples/` 默认通过 `.gitignore` 排除：它们体积
较大且与本机数据相关，不入版本库。`legacy/original/` 保留原始 Lua/Torch7 与
TensorFlow 代码，作为复现依据和历史对照，不在当前训练路径内运行。

## 文档导航

- 原文主实验配置、表格结果、对应 CLI 与本地复现核对
  → [docs/paper-main-experiments.md](docs/paper-main-experiments.md)
- 复现路线、模型族说明、S1-S13 变体定义、阶段任务
  → [docs/experiment-design.md](docs/experiment-design.md)
- 速度模式、TF32 / cuDNN benchmark / `torch.compile` 行为与基准计时
  → [docs/performance.md](docs/performance.md)
- 本地 conda 环境与可复现命令
  → [docs/environment.md](docs/environment.md)
- PyTorch 训练代码模块速览
  → [repro_pytorch/README.md](repro_pytorch/README.md)

## 环境

目标解释器：

```powershell
D:\DevTools\anaconda3\envs\wzdt\python.exe
```

已验证：

- Python 3.9.23
- PyTorch 2.7.1+cu128
- NumPy 1.26.4
- CUDA 12.8 + RTX 4090

依赖清单见 [repro_pytorch/requirements-wzdt.txt](repro_pytorch/requirements-wzdt.txt)。
环境创建与可复现命令见 [docs/environment.md](docs/environment.md)。

项目根目录：

```powershell
cd D:\Projects\code\UsingTheOutputEmbedding
```

## 数据

默认 PTB word-level 数据路径：

```text
data/ptb/
  ptb.train.txt
  ptb.valid.txt
  ptb.test.txt
```

文件来自 Mikolov et al. (2011) 的 `simple-examples` 数据包。原始打包文件
`simple-examples.tgz` 归档在 `data/raw/`，可在缺失数据时重新解压再生成。
`data/ptb/ptb.char.*.txt` 是字符级 PTB 数据，仅作为原始资产保留，不参与
当前 word-level 主实验。

数据统计：

| Split | Tokens |
| --- | ---: |
| Train | 929,589 |
| Valid | 73,760 |
| Test | 82,430 |
| Vocabulary | 10,000 |

## 快速开始

冒烟测试（环境与代码路径检查）：

```powershell
.\scripts\run_smoke.ps1
```

成功仅说明环境和代码路径可用，不代表训练质量。

单次实验，paper-style small + weight tying：

```powershell
.\scripts\run_ptb_main.ps1 -Model small -Variant wt
```

本地 4090 上的 large 配置 + S4 + LoRA rank 8 + relaxation scale 0.1：

```powershell
.\scripts\run_ptb_main.ps1 -Model large4090 -Variant s4 -LoraRank 8 -RelaxationScale 0.1
```

直接调用 Python 的等价命令：

```powershell
& "D:\DevTools\anaconda3\envs\wzdt\python.exe" .\repro_pytorch\train_ptb.py `
  --data_path .\data\ptb `
  --model large4090 `
  --variant s4 `
  --lora_rank 8 `
  --relaxation_scale 0.1 `
  --device auto `
  --output_dir .\runs\large4090-s4-r8-scale0p1
```

## 实验入口一览

| 任务 | 脚本 |
| --- | --- |
| 冒烟测试 | `.\scripts\run_smoke.ps1` |
| 单次实验 | `.\scripts\run_ptb_main.ps1` |
| S1-S13 变体 sweep | `.\scripts\run_embedding_variants.ps1` |
| 原文主实验复现套件 | `.\scripts\run_paper_ptb_experiments.ps1 -Tf32` |
| 核心对照（small / 4090） | `.\scripts\run_core_experiments.ps1` |
| 核心 + S1-S13 完整本地套件 | `.\scripts\run_all_experiments.ps1 -LoraRanks 1,2,4,8,16,32 -RelaxationScales 0.1,0.3,0.5,0.7,1.0,1.5,2.0` |
| Windows 计划任务（dry-run） | `.\scripts\run_all_experiments_scheduled_dryrun.cmd` |
| Windows 计划任务（执行） | `.\scripts\run_all_experiments_scheduled.cmd` |

`run_paper_ptb_experiments.ps1` 默认跳过已完成的 `runs/paper-*` 目录；如需
强制重跑追加 `-Force`。`-Tf32` 仅对 Bayesian-dropout 行启用 TF32 矩阵乘，
其余训练设定保持不变。

## 实验产物

`runs/` 中每个实验目录包含：

- `*.json`：参数、配置和每 epoch 训练 / 验证 / 测试 perplexity，以及
  `train_sec` / `train_wps` 等计时字段。
- `*.pt`：模型 checkpoint（large 模型约 250MB）。
- 同批运行的日志统一写入 `runs/logs-*-<timestamp>/` 下，按 job 名称分文件。

## 复现要点

- 报告时同时给出 validation 与 test perplexity；模型选择以 best validation
  为准，再报告同一次完整运行的 final test。
- `large4090` / `bayes4090` 仅用于本地快速筛选，不可直接对照原文表格；与
  原文比较时请使用 `small` 与 `large` 配置。
- S1-S13 实验必须同时记录 `model`、`variant`、`rank`、`relaxation_scale`、
  `seed` 和输出目录，方便后续核对。

## Citation

```bibtex
@InProceedings{press2017using,
  author    = {Press, Ofir and Wolf, Lior},
  title     = {Using the Output Embedding to Improve Language Models},
  booktitle = {Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers},
  month     = {April},
  year      = {2017},
  address   = {Valencia, Spain},
  publisher = {Association for Computational Linguistics},
  pages     = {157--163},
}
```
