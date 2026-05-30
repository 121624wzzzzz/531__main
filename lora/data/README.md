# 数据说明

本目录中的数据均为 JSONL，除主任务外，其余数据只保留用于历史审计。

## 主任务

| 数据集 | 路径 | 行数 | 说明 |
|---|---|---:|---|
| `sft_t2t_mini` train | `data/sft_t2t_mini_25k/train.jsonl` | 24000 | 来自 minimind `sft_t2t_mini` 的训练切片 |
| `sft_t2t_mini` eval | `data/sft_t2t_mini_25k/eval.jsonl` | 1000 | headline `eval_loss` holdout |

重新生成切分：

```bash
source /home/wz/projects/mypro/im_exp/set
cd /home/wz/projects/mypro/im_exp/lora
python scripts/prepare_sft_t2t_split.py
```

源文件：

```text
/home/wz/projects/mypro/im_exp/minimind/dataset/sft_t2t_mini.jsonl
```

## 保留的历史数据

这些数据不再有 active 训练或评测入口，仅用于审计旧结果和保留上下文。

### Math 轴

| 数据集 | 路径 | 行数 |
|---|---|---:|
| MetaMathQA-40K train | `data/metamathqa_40k/train.jsonl` | 40000 |
| GSM8K train | `data/gsm8k/train.jsonl` | 7473 |
| GSM8K test | `data/gsm8k/test.jsonl` | 1319 |
| MATH train | `data/math/train.jsonl` | 7500 |
| MATH test | `data/math/test.jsonl` | 5000 |

Phase 1 / 1.5 表明该路径在当前预算下不能区分 affine variant，因此已退出主流程。

### LoRMA-paper 二级数据

GLUE 路径：

```text
data/glue/<task>/<split>.jsonl
```

已准备任务包括 CoLA、SST-2、MRPC、QQP、STS-B、MNLI、QNLI、RTE。

E2E NLG：

| Split | 路径 | 行数 |
|---|---|---:|
| train | `data/e2e_nlg/train.jsonl` | 1728 |
| validation | `data/e2e_nlg/validation.jsonl` | 216 |
| test | `data/e2e_nlg/test.jsonl` | 217 |

原始归档：

```text
data/raw/glue_modelscope/
  QQP-clean.zip
  MNLI.zip
  QNLIv2.zip
```

这些历史数据的再生成脚本已移除。
