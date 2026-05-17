# Paper Main Experiments Reference

本文档专门记录 Press & Wolf (2017) 原文中与本项目最相关的主实验做法、配置、结果和结论，用作后续复现实验的固定检查清单。

论文：

- Ofir Press and Lior Wolf. 2017. *Using the Output Embedding to Improve Language Models*. EACL 2017.
- ACL Anthology: <https://aclanthology.org/E17-2025/>

## 1. 原文核心问题

原文关注神经语言模型中的两组词向量：

- 输入 embedding：把输入 token 映射到连续向量。
- 输出 embedding / softmax weight：把 hidden state 映射到词表 logits。

原文的核心主张是：在 RNN language model 中，输出 embedding 往往比输入 embedding 更接近语义质量；把输入 embedding 和输出 softmax weight 绑定为同一个矩阵，即 weight tying，可以提升泛化能力并减少参数量。

本项目重点复现的是原文在 Penn Treebank word-level language model 上的主实验。

## 2. 数据集口径

原文 PTB 实验使用 Mikolov et al. (2011) 的 Penn Treebank word-level 切分和词表。

本项目对应路径：

```text
data/ptb/
```

本项目固定使用以下文件：

```text
ptb.train.txt
ptb.valid.txt
ptb.test.txt
```

注意：`ptb.char.*.txt` 是字符级 PTB 数据，不是本文档讨论的 word-level 主实验默认输入。

本项目当前数据统计：

| Split | Tokens |
| --- | ---: |
| Train | 929,589 |
| Valid | 73,760 |
| Test | 82,430 |
| Vocabulary | 10,000 |

## 3. 原文主实验模型

### 3.1 Small Model

原文 small model 跟随 Zaremba et al. (2014) 的 small LSTM 设置：

| Item | Value |
| --- | --- |
| Architecture | 2-layer LSTM |
| Hidden size | 200 |
| Sequence length | 20 |
| Batch size | 20 |
| Dropout | No dropout |
| Initial learning rate | 1.0 |
| Gradient clipping | 5.0 |
| Init scale | 0.1 |
| Epoch schedule | 13 epochs; learning rate starts decaying after epoch 4 |
| LR decay | 0.5 |

本项目对应配置：`--model small`。

### 3.2 Large Model

原文 large model 同样跟随 Zaremba et al. (2014)，但使用更大的 LSTM，并加入 dropout：

| Item | Value |
| --- | --- |
| Architecture | 2-layer LSTM |
| Hidden size | 1500 |
| Sequence length | 35 |
| Batch size | 20 |
| Dropout placement | before first LSTM, between LSTM layers, after second LSTM |
| Dropout probability | 0.65 |
| Keep probability | 0.35 |
| Initial learning rate | 1.0 |
| Gradient clipping | 10.0 |
| Init scale | 0.04 |
| Epoch schedule | 55 epochs; learning rate starts decaying after epoch 14 |
| LR decay | 1 / 1.15 |

本项目对应配置：`--model large`。

### 3.3 Large + Bayesian Dropout

原文还对照了 Gal / Bayesian dropout 风格的大模型，并报告 weight decay 版本。

本项目中的复现配置为：

| Item | Value |
| --- | --- |
| Project model | `bayes1500` |
| Architecture | variational LSTM |
| Hidden size | 1500 |
| Sequence length | 35 |
| Batch size | 20 |
| Input dropout | 0.3 |
| Input gate dropout | 0.5 |
| Hidden gate dropout | 0.3 |
| Output dropout | 0.5 |
| Initial learning rate | 1.0 |
| Gradient clipping | 10.0 |
| Init scale | 0.04 |
| Epoch schedule | 55 epochs; learning rate starts decaying after epoch 10 |
| LR decay | 1 / 1.15 |
| Weight decay | `1e-7` for the paper-matched WD rows |

本项目对应配置：`--model bayes1500 --legacy_weight_decay 1e-7`。

当前本地 4090 上的 Bayesian-dropout 复现使用 `--tf32`，只改变矩阵乘法精度，不改变模型结构、batch size、epoch schedule、dropout rate 或 weight decay。

## 4. 原文比较的变体

### 4.1 Baseline

输入 embedding 和输出 softmax weight 是两个独立矩阵：

```text
input embedding: E
output weight: U
```

本项目对应：`--variant baseline`。

### 4.2 Weight Tying

输入 embedding 和输出 softmax weight 使用同一个矩阵：

```text
input embedding: W
output weight: W
```

本项目对应：`--variant wt`。

原文主要结论之一是：weight tying 在 PTB large model 和 Bayesian-dropout large model 上都能降低 validation/test perplexity，同时显著减少参数量。

### 4.3 Projection Regularization

原文对无 dropout 的 small model 还引入 projection regularization。它在输出侧加入一个额外 projection，并正则化 projection 后向量与真实词输出 embedding 的距离。

本项目对应：

```text
--variant pr
--variant wt_pr
```

PR 主要用于 small/no-dropout 设置，不是 large/dropout 主表的核心变体。

## 5. 原文 PTB 主结果

### 5.1 Dropout / Bayesian Dropout Large Rows

来源：原文 Table 5，word-level PTB perplexity，越低越好。

| Model | Size | Train | Valid | Test |
| --- | ---: | ---: | ---: | ---: |
| Large (Zaremba et al., 2014) | 66M | 37.8 | 82.2 | 78.4 |
| Large + Weight Tying | 51M | 48.5 | 77.7 | 74.3 |
| Large + BD + WD | 66M | 24.3 | 78.1 | 75.2 |
| Large + BD + WT | 51M | 28.2 | 75.8 | 73.2 |

直接结论：

- `Large + WT` 相比 `Large`：test PPL 从 `78.4` 降到 `74.3`，参数量从 `66M` 降到 `51M`。
- `Large + BD + WT` 相比 `Large + BD + WD`：test PPL 从 `75.2` 降到 `73.2`，参数量同样从 `66M` 降到 `51M`。
- Weight tying 降低了 test perplexity，同时减少了约 15M 参数。
- 训练 PPL 反而更高，说明 tied model 不是通过更强记忆训练集获益，而更像是减少过拟合、改善泛化。

### 5.2 No-Dropout Small Rows

来源：原文 Table 6，word-level PTB perplexity，越低越好。

| Model | Size | Train | Valid | Test |
| --- | ---: | ---: | ---: | ---: |
| Small model | 4.65M | 38.0 | 120.7 | 114.5 |
| Small + WT | 2.65M | 36.4 | 117.5 | 112.4 |
| Small + PR | 4.69M | 50.8 | 116.0 | 111.7 |
| Small + WT + PR | 2.69M | 53.5 | 104.9 | 100.9 |

直接结论：

- 在 no-dropout small model 中，`WT`、`PR` 都能改善 test PPL。
- `WT + PR` 最强：test PPL 从 `114.5` 降到 `100.9`。
- `WT + PR` 的 train PPL 明显更高，但 valid/test 更好，仍然体现出正则化效果。

## 6. 当前项目复现检查口径

### 6.1 推荐命令

加速口径、`--speed_mode` 预设、TF32 / cuDNN benchmark / torch.compile 的行为
和实测数据见 [`performance.md`](performance.md)。下面的命令默认走严格 FP32。

small/no-dropout 对照：

```powershell
.\scripts\run_ptb_main.ps1 -Model small -Variant baseline
.\scripts\run_ptb_main.ps1 -Model small -Variant wt
.\scripts\run_ptb_main.ps1 -Model small -Variant pr
.\scripts\run_ptb_main.ps1 -Model small -Variant wt_pr
```

paper-scale PTB 对照：

```powershell
.\scripts\run_paper_ptb_experiments.ps1 -Tf32
```

`run_paper_ptb_experiments.ps1` 会跳过已完成的 `runs/paper-*` 目录；如需强制重跑，使用脚本的 `-Force` 参数。

### 6.2 当前已核对的复现结果

当前本地结果来自 `runs/paper-*/*.json`。

| Run | Our valid | Our test | Paper valid | Paper test | Assessment |
| --- | ---: | ---: | ---: | ---: | --- |
| small baseline | 119.92 | 114.53 | 120.7 | 114.5 | match |
| small wt | 115.05 | 110.27 | 117.5 | 112.4 | better |
| small pr | 113.30 | 110.06 | 116.0 | 111.7 | better |
| small wt_pr | 107.12 | 103.09 | 104.9 | 100.9 | close trend |
| large baseline | 82.46 | 78.44 | 82.2 | 78.4 | match |
| large wt | 78.06 | 74.97 | 77.7 | 74.3 | close |
| bayes1500 baseline wd1e-7 | 78.89 | 75.02 | 78.1 | 75.2 | close |
| bayes1500 wt wd1e-7 | 76.70 | 73.43 | 75.8 | 73.2 | close |

判断标准：

- `large baseline` 是最重要的 sanity row，当前 test PPL `78.44` 对原文 `78.4`，说明 PTB 数据、Zaremba large 配置、loss/eval 口径基本正确。
- `large wt` 和 `bayes1500 wt` 都保留了原文趋势：weight tying 明显优于 untied baseline。
- Bayesian-dropout rows 与原文差距在约 1 PPL 内，当前可认为基本复现。
- `small wt_pr` 比原文高约 2.2 test PPL，但相对 baseline 的提升趋势正确；它更适合作为代码路径 sanity check，而不是最终主结论的唯一依据。

## 7. 后续实验检查清单

后续每次修改训练代码或新增 embedding 变体后，建议按以下顺序检查：

1. 跑 smoke test，确认标准 LSTM 和 variational LSTM 路径没有异常。
2. 跑 `small baseline` 和 `small wt`，确认 weight tying 至少改善 small baseline。
3. 如修改涉及 loss、batching、state handling、dropout、embedding 或 softmax，重跑 `large baseline`。
4. 如修改涉及 tied embedding、output projection 或 Bayesian dropout，重跑 `large wt` 和 `bayes1500 wt wd1e-7`。
5. 不要用 `large4090` 的绝对 PPL 对照原文；它只用于本地快速筛选。

## 8. 一句话结论

原文主实验的核心结论是：在 PTB word-level LSTM language model 中，weight tying 在减少参数量的同时降低 validation/test perplexity；projection regularization 在 no-dropout small setting 中进一步增强这一效果。当前项目的 `runs/paper-*` 结果已经基本复现了这个结论。
