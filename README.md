# UsingTheOutputEmbedding PTB 复现实验

本项目复现并扩展 Press & Wolf, *Using the Output Embedding to Improve
Language Models* (EACL 2017) 中 Penn Treebank word-level language model 实验。
代码统一为现代 PyTorch 实现，并加入 S1-S21 embedding relaxation 变体研究。
目标平台是 **Linux + 8×A100**，调度脚本以 Bash/Python 为准。

核心研究问题：

- 在 PTB word-level LSTM language model 中，输入 embedding 与输出 softmax
  weight 的不同共享方式（baseline / wt / pr / wt_pr）对验证 / 测试 perplexity
  的影响。
- 在 tied embedding 附近做局部松弛（S1-S21）能否在小幅增参的前提下进一步
  提升 perplexity，以及不同 LoRA-like rank 与 `relaxation_scale` 下的稳定性。

## 目录结构

```text
UsingTheOutputEmbedding-repro/
  README.md                       项目入口与快速开始
  docs/                           文档（3 份）
    experiments.md                 原论文配置、S1-S21 变体定义、实验阶段与决策规则
    results.md                     完整实验结果（论文写作主参考）
    performance.md                 速度模式、TF32 / cuDNN benchmark 与 A100 基准计时
  repro_pytorch/                  PyTorch 训练代码
    train_ptb.py                   CLI 入口与主流程编排
    configs.py                     模型配置（small / medium / large / large4090 / bayes1500 / test）
    variants.py                    S1-S21 embedding relaxation 模块与公式
    model.py                       Zaremba LSTM、Variational LSTM、TinyTransformer
    rhn.py                         Recurrent Highway Network
    train_loop.py                  训练 / 评估循环与 checkpoint 持久化
    ptb_data.py                    PTB 词表与 batch 迭代器
    requirements.txt               依赖
    README.md                      模块速览
  scripts/                        Linux 8×A100 调度脚本
    a100_jobs.py                   Job 数据类、各阶段预设、JSON 任务列表生成
    a100_run.py                    按 GPU 池并行执行任务列表
    a100_summarize.py              扫描 runs/ 生成 CSV / Markdown 汇总
    README.md                      脚本使用说明
  data/ptb/                       Penn Treebank 数据（不入版本库）
  runs/                           训练结果：JSON 指标 / checkpoint / 日志（不入版本库）
```

## 文档导航

- **最终结果（论文写作主参考）** → [**docs/results.md**](docs/results.md)
- 原论文配置、S1-S21 变体定义、实验阶段与决策规则 → [docs/experiments.md](docs/experiments.md)
- 速度模式、TF32 / cuDNN benchmark / `torch.compile` 行为与 A100 基准计时 → [docs/performance.md](docs/performance.md)
- PyTorch 训练代码模块速览 → [repro_pytorch/README.md](repro_pytorch/README.md)
- Linux 8×A100 调度脚本使用 → [scripts/README.md](scripts/README.md)

## 一句话主结论

在 Press & Wolf (2017) PTB large word-level LM 设定下，把 weight tying 替换
为「**输出端乘性低秩松弛** $W \to W(I + s \cdot PQ)$」（**最佳点 rank=32，
scale=0.03**），仅增加 **96K（0.19%）参数**，5 seed 平均 test PPL 从 weight
tying 的 **74.79 ± 0.19** 降到 **73.21 ± 0.18**，**降幅 1.58 PPL**（Welch
t≈14, p < 1e-6）。rank 推到 256 mean 可继续下降到 73.15 ± 0.39。

**结构特异性（输入侧 vs 输出侧分水岭）：** 经过 100+ 个 paper-scale 多 seed
测点的深度扫描，**所有输入侧变体（S3 加性偏置 / S4 加性低秩 / S6 乘性低秩 /
S12 shift+mult / S13 add+mult / S8-S10 untied 输入族）都无法稳定改进 WT**：
最强输入侧多 seed mean 74.63（S6 r=8 s=0.01），仅比 WT 低 0.16 PPL（t≈1.3，
不显著）；而输出侧的 S7（−1.58 PPL）和 S5（−0.64 PPL）都显著。

**rank vs scale 解耦（关键洞察）：**
- 输出端 S7：**rank 决定能不能 work**（r 越大 mean 越低），scale 决定**稳不稳**（r 上升后 s 必须减小）
- 输入端：**scale 决定能不能 work**（窄稳定带），rank 反而**有害**（r=16 比 r=8 普遍恶化 0.5 PPL）

详细 12 阶段共 800+ paper-scale runs 的完整数据、scale 悬崖测绘、rank 单调收益
曲线、统计显著性检验见 [docs/results.md](docs/results.md)。

## 环境

目标平台：Linux + 8×A100 80GB；conda 环境 `torch24`。已验证：

- Python 3.10.20
- PyTorch 2.4.1+cu124
- NumPy 1.26.4
- CUDA 12.4 (NVIDIA driver 550.x)
- 解释器：`/home/wz/anaconda3/envs/torch24/bin/python`

激活方式（项目根目录之外通用）：

```bash
source /home/wz/projects/mypro/im_exp/set
```

依赖清单见 [repro_pytorch/requirements.txt](repro_pytorch/requirements.txt)，
环境创建与可复现命令见 [docs/experiments.md](docs/experiments.md)。

## 数据

默认 PTB word-level 数据路径：

```text
data/ptb/
  ptb.train.txt
  ptb.valid.txt
  ptb.test.txt
```

文件来自 Zaremba LSTM 仓库的 Mikolov PTB word-level 切分。

数据统计：

| Split | Tokens |
| --- | ---: |
| Train | 929,589 |
| Valid | 73,760 |
| Test | 82,430 |
| Vocabulary | 10,000 |

数据下载命令（在仓库根目录执行）：

```bash
mkdir -p data/ptb && cd data/ptb
for url in \
  https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt \
  https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt \
  https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt; do
  curl -fsSLO "$url"
done
cd ../..
```

## 快速开始

冒烟测试（环境与代码路径检查）：

```bash
PY=/home/wz/anaconda3/envs/torch24/bin/python
CUDA_VISIBLE_DEVICES=1 $PY repro_pytorch/train_ptb.py \
  --data_path data/ptb --model test --variant baseline \
  --device cuda:0 --max_epochs 1 --max_train_batches 5 \
  --max_eval_batches 5 --metrics_only --output_dir runs/_smoke
```

成功仅说明环境和代码路径可用，不代表训练质量。

单次实验，paper-style small + weight tying：

```bash
CUDA_VISIBLE_DEVICES=1 $PY repro_pytorch/train_ptb.py \
  --data_path data/ptb --model small --variant wt \
  --device cuda:0 --output_dir runs/single/small-wt
```

S1-S21 变体单次实验（local screening 配置）：

```bash
CUDA_VISIBLE_DEVICES=1 $PY repro_pytorch/train_ptb.py \
  --data_path data/ptb --model large4090 --variant s4 \
  --lora_rank 8 --relaxation_scale 0.1 --tf32 \
  --device cuda:0 --metrics_only \
  --output_dir runs/single/large4090-s4-r8-s0p1
```

## 实验入口一览（8×A100 批量调度）

| 任务 | 命令 |
| --- | --- |
| 生成 paper-small 任务 | `python scripts/a100_jobs.py gen paper-small --out runs/jobs/paper-small.json` |
| 生成 paper-large 任务 | `python scripts/a100_jobs.py gen paper-large --out runs/jobs/paper-large.json` |
| 生成 paper-bayes 任务 | `python scripts/a100_jobs.py gen paper-bayes --out runs/jobs/paper-bayes.json` |
| 生成 S1-S13 全量筛选 | `python scripts/a100_jobs.py gen screen-large4090 --out runs/jobs/screen-large4090.json` |
| 生成 S14-S17 tied 双侧筛选 | `python scripts/a100_jobs.py gen screen-bilateral-tied --out runs/jobs/screen-bilateral-tied.json` |
| 生成 S14-S17 低 scale 追加 | `python scripts/a100_jobs.py gen screen-bilateral-tied-lowscale --out runs/jobs/screen-bilateral-tied-lowscale.json` |
| 生成 S14-S17 paper-scale 诊断 | `python scripts/a100_jobs.py gen paper-bilateral-tied --out runs/jobs/paper-bilateral-tied.json` |
| 生成 S18-S21 shift/mult follow-up | `python scripts/a100_jobs.py gen paper-shift-mult-followup --out runs/jobs/paper-shift-mult-followup.json` |
| 生成低 scale 追加 | `python scripts/a100_jobs.py gen screen-large4090-lowscale --out runs/jobs/screen-lowscale.json` |
| 批量执行（默认 1-6 号卡） | `python scripts/a100_run.py runs/jobs/<jobs>.json --gpus 1,2,3,4,5,6` |
| 跳过已完成、失败重试 | `python scripts/a100_run.py <jobs.json> --gpus 1,2,3,4,5,6 --retries 1` |
| 汇总全部结果 | `python scripts/a100_summarize.py --csv runs/summaries/all.csv --md runs/summaries/all.md` |
| 每变体最佳 | `python scripts/a100_summarize.py --best_md runs/summaries/best.md --best_metric best_valid_ppl` |

详细 GPU 池策略、`--include_groups`、`--force`、输出目录命名规则等说明见 [scripts/README.md](scripts/README.md)。

## 实验产物

`runs/<group>/<unique-dir>/` 中每个实验目录包含：

- `ptb_<model>_<arch>_<variant>.json`：参数、配置和每 epoch 训练 / 验证 / 测试 perplexity，
  以及 `train_sec` / `train_wps` 等计时字段。`--metrics_only` 时只写 JSON。
- `ptb_<model>_<arch>_<variant>.pt`：模型 checkpoint（large 模型约 250MB；筛选模式默认不写）。
- `run.log`：训练标准输出 + 标准错误的逐字记录。

合并运行日志统一写入 `runs/logs/`，结果汇总写入 `runs/summaries/`。

## 复现要点

- 报告时同时给出 validation 与 test perplexity；模型选择以 best validation
  为准，再报告同一次完整运行的 final test。
- `large4090` / `bayes4090` 仅用于本地快速筛选，不可直接对照原文表格；与
  原文比较时请使用 `small` 与 `large` / `bayes1500` 配置。
- S1-S21 实验必须同时记录 `model`、`variant`、`rank`、`relaxation_scale`、
  `seed` 和输出目录，方便后续核对。`scripts/a100_jobs.py` 会自动把这些
  字段编码进 `output_dir`，避免不同超参的 JSON 互相覆盖。
- 8×A100 上默认使用 GPU `1,2,3,4,5,6`，避开常被占用的 `0`、`7`；可通过
  `--gpus` 调整。

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
