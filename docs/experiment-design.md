# Experiment Design

本文档定义当前项目的复现实验路线。目标不是只跑出一个更低 PPL，而是把论文复现、结构扩展、稳定性诊断和最终报告口径分开。

原文主实验的做法、配置、表格结果和当前复现核对清单见
[`paper-main-experiments.md`](paper-main-experiments.md)。

训练脚本支持的加速口径、CLI 开关和 RTX 4090 上的基准计时见
[`performance.md`](performance.md)。

## 1. Research Goals

本项目围绕 Press & Wolf (2017) 的核心问题继续展开：

1. 复现 PTB word-level language model 中 output embedding tying 的收益。
2. 验证现代 PyTorch 实现能否在 small 配置上接近原文结果。
3. 在 tied embedding 附近设计 S1-S13 松弛结构，寻找更高效的参数化。
4. 诊断额外模块的训练稳定性，尤其是 LoRA-like scale、乘法分支和输出侧分支。
5. 对有希望的结构做 paper-scale 和多 seed 复现。

## 2. Dataset

固定使用 Mikolov `simple-examples` 数据包的 Penn Treebank word-level 切分，
项目内默认路径为 `data/ptb/`：

| Split | Tokens |
| --- | ---: |
| Train | 929,589 |
| Valid | 73,760 |
| Test | 82,430 |
| Vocabulary | 10,000 |

所有实验都必须使用同一份 `data/ptb/`，避免因为数据预处理差异影响结论。

## 3. Model Families

### 3.1 Paper-aligned models

这些配置用于和论文表格比较：

| Model | Architecture | Hidden size | Layers | Batch | Steps | Purpose |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `small` | LSTM | 200 | 2 | 20 | 20 | Primary sanity check |
| `medium` | LSTM | 650 | 2 | 20 | 35 | Optional bridge |
| `large` | LSTM | 1500 | 2 | 20 | 35 | Paper-scale comparison |
| `bayes1500` | variational LSTM | 1500 | 2 | 20 | 35 | Bayesian dropout comparison |

### 3.2 Local screening models

These are for fast local ranking on the RTX 4090. They are not paper-equivalent
recipes.

| Model | Difference | Purpose |
| --- | --- | --- |
| `large4090` | larger batch, shorter schedule | Fast S1-S13 screening |
| `bayes4090` | larger batch, shorter schedule | Fast variational-dropout screening |

Any conclusion from `large4090` must be confirmed with `large` before being
described as a paper-scale result.

## 4. Variants

### 4.1 Original paper variants

| Variant | Meaning |
| --- | --- |
| `baseline` | Untied input embedding and output softmax matrix |
| `wt` | Weight tying between input embedding and output softmax matrix |
| `pr` | Projection regularization without weight tying |
| `wt_pr` | Weight tying plus projection regularization |

### 4.2 S1-S13 relaxation variants

The S variants are local relaxations around tied or untied embeddings.

| ID | Description | Extra params at V=10000,d=1500,r=8 |
| --- | --- | ---: |
| S1 | Tied | 0 |
| S2 | Untied | 15,000,000 |
| S3 | Input shift | 1,500 |
| S4 | Input additive low-rank | 92,000 |
| S5 | Output additive low-rank | 92,000 |
| S6 | Input multiplicative low-rank | 24,000 |
| S7 | Output multiplicative low-rank | 24,000 |
| S8 | Untied + input shift | 15,001,500 |
| S9 | Untied + input additive low-rank | 15,092,000 |
| S10 | Untied + input multiplicative low-rank | 15,024,000 |
| S11 | Output hidden shift | 1,500 |
| S12 | Input shift + multiplicative | 25,500 |
| S13 | Input additive + multiplicative | 116,000 |

All S variants support:

```text
--lora_rank <r>
--relaxation_scale <scale>
```

`relaxation_scale` multiplies every non-base relaxation term. For example S4 is
implemented as:

```text
input = W + scale * A B
output = W
```

This scale is part of the experiment definition and must be reported.

## 5. Metrics

Every completed run should record:

- Train perplexity per epoch.
- Validation perplexity per epoch.
- Final test perplexity.
- Best validation perplexity and its epoch.
- Parameter count.
- Variant metadata: model, variant, rank, relaxation scale, seed.
- Checkpoint path and JSON path.

For diagnostic runs, also record:

- `||W||`
- `||delta||` for low-rank or multiplicative branches.
- `||delta|| / ||W||`
- total gradient norm before clipping.
- per-branch gradient norm when practical.

## 6. Current Findings To Preserve

### 6.1 Small sanity check

The small model reproduces the paper trend and approximate values:

| Run | Best valid PPL | Test PPL |
| --- | ---: | ---: |
| small baseline | 119.92 | 114.53 |
| small wt | 115.05 | 110.27 |
| small pr | 113.30 | 110.06 |
| small wt_pr | 107.12 | 103.09 |

This suggests that PTB loading, loss computation, SGD training, and basic tying
logic are sound.

### 6.2 Paper PTB reproduction

The current PyTorch reproduction matches the main PTB paper rows closely:

| Run | Our valid | Our test | Paper valid | Paper test |
| --- | ---: | ---: | ---: | ---: |
| large baseline | 82.46 | 78.44 | 82.2 | 78.4 |
| large wt | 78.06 | 74.97 | 77.7 | 74.3 |
| bayes1500 baseline wd1e-7 | 78.89 | 75.02 | 78.1 | 75.2 |
| bayes1500 wt wd1e-7 | 76.70 | 73.43 | 75.8 | 73.2 |

Standard LSTM rows were run in strict FP32. Bayesian-dropout rows were run with
`--tf32`, which changes only matrix-multiply precision on the local RTX 4090
and keeps the architecture, batch size, epoch schedule, dropout rates, and
weight decay settings unchanged.

### 6.3 Large4090 screening is not paper-scale

`large4090` uses full PTB data but a larger batch and shorter schedule than the
paper. It is useful for ranking variants, but its absolute PPL should not be
compared directly against the paper's `large` table.

### 6.4 Scale matters

At `relaxation_scale=1.0`, many extra branches became unstable. At
`relaxation_scale=0.1`, several variants recovered:

| Variant | scale 1.0 test | scale 0.1 test |
| --- | ---: | ---: |
| S4 | 83.67 | 80.64 |
| S5 | 213.98 | 81.18 |
| S6 | 128.57 | 87.05 |
| S7 | 644.20 | 109.25 |
| S13 | 126.61 | 84.62 |

This means poor scale-1.0 results should be treated as optimization instability,
not as final evidence against the structures.

## 7. Experimental Phases

### Phase 0: Environment and code-path check

Run after any nontrivial edit:

```powershell
.\scripts\run_smoke.ps1
```

Success criterion:

- No exception.
- CUDA detected when `-Device auto` is used on the local machine.
- Both standard and variational paths complete.

### Phase 1: Paper small reproduction

Purpose: verify the implementation against the easiest paper-aligned setting.

Commands:

```powershell
.\scripts\run_ptb_main.ps1 -Model small -Variant baseline
.\scripts\run_ptb_main.ps1 -Model small -Variant wt
.\scripts\run_ptb_main.ps1 -Model small -Variant pr
.\scripts\run_ptb_main.ps1 -Model small -Variant wt_pr
```

Success criterion:

- Baseline test PPL close to 114.5.
- WT improves over baseline.
- WT+PR is the best or near-best small variant.

### Phase 2: Fast S1-S13 rank/scale screening

Purpose: rank S variants under a fixed fast local budget while checking
sensitivity to both low-rank capacity and relaxation scale.

Primary command:

```powershell
.\scripts\run_embedding_variants.ps1 `
  -Model large4090 `
  -LoraRanks 1,2,4,8,16,32 `
  -RelaxationScales 0.1,0.3,0.5,0.7,1.0,1.5,2.0
```

The sweep avoids redundant runs: S1/S2 run once, shift-only variants sweep
scale only, and low-rank variants sweep the full `rank x scale` grid.

Report:

- Best validation PPL.
- Final test PPL.
- Extra parameter count.
- Rank and relaxation scale.

Interpretation:

- Use this phase to choose candidates.
- Do not compare absolute PPL to the paper large model.

### Phase 3: Focused relaxation scale sweep

Purpose: identify stable scale for each branch.

Candidate variants:

```text
S4, S5, S6, S7, S10, S13
```

Candidate scales:

```text
0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0
```

Suggested run matrix:

```powershell
foreach ($Scale in 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0) {
  foreach ($Variant in "s4", "s5", "s6", "s7", "s10", "s13") {
    .\scripts\run_ptb_main.ps1 `
      -Model large4090 `
      -Variant $Variant `
      -LoraRank 8 `
      -RelaxationScale $Scale
  }
}
```

Success criterion:

- Identify variants that stay near or better than S1 without early-epoch PPL
  explosion.
- Reject only after checking at least one smaller scale.

### Phase 4: Focused rank sweep

Purpose: test whether low-rank capacity is the bottleneck.

Candidate variants:

```text
S4, S5, S13
```

Candidate ranks:

```text
1, 2, 4, 8, 16, 32
```

Use the best scale band found in Phase 2/3.

Report:

- Test PPL versus extra parameters.
- Pareto frontier: lowest PPL for each parameter budget.

### Phase 5: Paper-scale confirmation

Purpose: confirm that fast-screening winners remain good under paper-aligned
training.

Candidates:

```text
S1, S2, S4, S5, S13
```

Primary commands:

```powershell
.\scripts\run_ptb_main.ps1 -Model large -Variant s1 -LoraRank 8 -RelaxationScale 0.1
.\scripts\run_ptb_main.ps1 -Model large -Variant s2 -LoraRank 8 -RelaxationScale 0.1
.\scripts\run_ptb_main.ps1 -Model large -Variant s4 -LoraRank 8 -RelaxationScale 0.1
.\scripts\run_ptb_main.ps1 -Model large -Variant s5 -LoraRank 8 -RelaxationScale 0.1
.\scripts\run_ptb_main.ps1 -Model large -Variant s13 -LoraRank 8 -RelaxationScale 0.1
```

Optional paper-style final test:

```powershell
& "D:\DevTools\anaconda3\envs\wzdt\python.exe" .\repro_pytorch\train_ptb.py `
  --data_path .\data\ptb `
  --model large `
  --variant s4 `
  --lora_rank 8 `
  --relaxation_scale 0.1 `
  --paper_test_eval `
  --device auto `
  --output_dir .\runs\large-s4-r8-scale0p1-paper-test
```

### Phase 6: Multi-seed robustness

Purpose: make sure the best variants are not single-seed accidents.

Seeds:

```text
1, 2, 3, 4, 5
```

Candidates:

```text
S1, S2, best S4/S5/S13 configuration
```

Report:

- Mean test PPL.
- Standard deviation.
- Best validation PPL distribution.
- Any failed or unstable runs.

## 8. Decision Rules

Use these rules when deciding what to run next:

1. If a variant is bad at scale 1.0 but good at scale 0.1, classify it as
   scale-sensitive, not failed.
2. If a variant explodes even at 0.01, mark it unstable under current optimizer.
3. A candidate must beat S1 on `large4090` before it deserves paper-scale
   compute, unless it answers a diagnostic question.
4. A candidate must beat or match S1 on paper-scale `large` before it is claimed
   as a real improvement.
5. Any final claim should include parameter delta, not just PPL.

## 9. Reporting Template

Use this table for every experimental batch:

| Model | Variant | Rank | Scale | Seed | Extra params | Best valid | Test | Output dir |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| large4090 | s4 | 8 | 0.1 | 1 | 92K | 84.03 | 80.64 | `runs/large4090-s4-r8-scale0p1` |

Short interpretation:

```text
Question:
Setup:
Main result:
Comparison to S1/S2:
Stability notes:
Next action:
```

## 10. Known Risks

- `large4090` can change rankings relative to paper-scale `large`.
- Output-side and multiplicative branches are sensitive to relaxation scale.
- S8 at scale 0.1 behaved abnormally and needs separate debugging.
- Current scripts use single-seed runs unless `--seed` is passed directly to
  `train_ptb.py`.
- Checkpoints are large; keep `runs/` out of Git.
