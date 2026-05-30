# Linux 8×A100 调度脚本

本目录是项目唯一的实验调度入口，专门面向 Linux + 多卡 A100。所有调度都以
**每个实验单进程单 GPU** 为粒度，通过 `CUDA_VISIBLE_DEVICES=<gpu> --device cuda:0`
绑卡；每个实验有唯一 `output_dir`，避免 `ptb_<model>_<arch>_<variant>.json`
互相覆盖。

## 文件

- [`a100_jobs.py`](a100_jobs.py) — `Job` 数据类、各阶段预设（paper-small / paper-large /
  paper-bayes / screen-large4090 / screen-large4090-lowscale / screen-bilateral-tied 等），以及对外的
  `gen` / `count` 两个子命令把预设转成 JSON 任务列表；同时提供编程接口
  `preset_paper_confirm()` 和 `preset_multi_seed()`，可用 Python 一行生成
  Phase 5/6 的 JSON。
- [`a100_run.py`](a100_run.py) — 并行执行器：读 JSON 任务列表，按 GPU 池调度，
  跳过已完成的实验（依据 `output_dir/ptb_*.json` 是否含 `test_ppl`），
  失败可重试，每个实验日志落在 `output_dir/run.log`。
- [`a100_summarize.py`](a100_summarize.py) — 解析 `runs/**/ptb_*.json`，输出
  CSV / Markdown 汇总表，并支持「每 (model, variant) 最佳」筛选。

## 默认 Python 与数据路径

- Python：`/home/wz/anaconda3/envs/torch24/bin/python`（可通过环境变量
  `REPRO_PYTHON` 覆盖）
- 数据：`<repo>/data/ptb`
- GPU 池：默认 `1,2,3,4,5,6`（优先级最高的 6 张卡），需要时可加 `0`，避免 `7`。

## 典型用法

```bash
cd /home/wz/projects/mypro/im_exp/UsingTheOutputEmbedding-repro
PY=/home/wz/anaconda3/envs/torch24/bin/python

# 1) 论文复现实验
$PY scripts/a100_jobs.py gen paper-small --out runs/_meta/jobs/paper-small.json
$PY scripts/a100_jobs.py gen paper-large --out runs/_meta/jobs/paper-large.json
$PY scripts/a100_jobs.py gen paper-bayes --out runs/_meta/jobs/paper-bayes.json
$PY scripts/a100_run.py runs/_meta/jobs/paper-small.json --gpus 1,2,3,4
$PY scripts/a100_run.py runs/_meta/jobs/paper-large.json --gpus 1,2
$PY scripts/a100_run.py runs/_meta/jobs/paper-bayes.json --gpus 3,4

# 2) S1-S13 全量筛选（large4090 fast screening）
$PY scripts/a100_jobs.py gen screen-large4090 --out runs/_meta/jobs/screen-large4090.json
$PY scripts/a100_run.py runs/_meta/jobs/screen-large4090.json --gpus 1,2,3,4,5,6

# 3) 敏感 scale 追加
$PY scripts/a100_jobs.py gen screen-large4090-lowscale --out runs/_meta/jobs/screen-lowscale.json
$PY scripts/a100_run.py runs/_meta/jobs/screen-lowscale.json --gpus 1,2,3,4,5,6

# 4) tied 双侧 S14-S17 追加筛选
$PY scripts/a100_jobs.py gen screen-bilateral-tied --out runs/_meta/jobs/screen-bilateral-tied.json
$PY scripts/a100_jobs.py gen screen-bilateral-tied-lowscale --out runs/_meta/jobs/screen-bilateral-tied-lowscale.json
$PY scripts/a100_run.py runs/_meta/jobs/screen-bilateral-tied.json --gpus 1,2,3,4,5,6
$PY scripts/a100_run.py runs/_meta/jobs/screen-bilateral-tied-lowscale.json --gpus 1,2,3,4,5,6

# 5) tied 双侧 paper-scale 5 seed 诊断
$PY scripts/a100_jobs.py gen paper-bilateral-tied --out runs/_meta/jobs/paper-bilateral-tied.json
$PY scripts/a100_run.py runs/_meta/jobs/paper-bilateral-tied.json --gpus 1,2,3,4,5,6

# 6) shift/mult follow-up（S18-S21）
$PY scripts/a100_jobs.py gen paper-shift-mult-followup --out runs/_meta/jobs/paper-shift-mult-followup.json
$PY scripts/a100_run.py runs/_meta/jobs/paper-shift-mult-followup.json --gpus 1,2,3,4,5,6

# 7) 汇总
$PY scripts/a100_summarize.py \
  --csv runs/_meta/summaries/all.csv \
  --md runs/_meta/summaries/all.md \
  --best_md runs/_meta/summaries/best.md \
  --best_metric best_valid_ppl
```

### Phase 5 / Phase 6：paper-confirm 与 multi-seed

这两个阶段的 JSON 不能凭模板一键生成，需要传入「从 screening 选出的候选
配置」。可以直接用 Python 编程式调用：

```bash
$PY - <<'PY'
import sys, json
sys.path.insert(0, 'scripts')
from a100_jobs import preset_paper_confirm, preset_multi_seed, jobs_to_json

# Phase 5: 在 paper-scale large 上重跑候选 + s1/s2 控制
confirm = (
    preset_paper_confirm(["s1", "s2"], rank=8, scale=0.1)
    + preset_paper_confirm(["s4"], rank=2, scale=0.1)
    + preset_paper_confirm(["s5"], rank=2, scale=0.3)
    + preset_paper_confirm(["s13"], rank=1, scale=0.1)
)
open("runs/_meta/jobs/paper-confirm.json", "w").write(jobs_to_json(confirm))

# Phase 6: 多 seed 稳健性（每候选 5 seed）
seeds = preset_multi_seed(["s1", "s4", "s5", "s13"], rank=8, scale=0.1)
open("runs/_meta/jobs/multi-seed.json", "w").write(jobs_to_json(seeds))
PY

$PY scripts/a100_run.py runs/_meta/jobs/paper-confirm.json --gpus 1,2,3,4,5,6
$PY scripts/a100_run.py runs/_meta/jobs/multi-seed.json    --gpus 1,2,3,4,5,6
```

### Tied 双侧 S14-S17

`S14-S17` 是共享 `W` 基座上的双侧松弛，不再是 `E/U` 完全 untied 基座：

| Variant | Input | Output | 对照 |
| --- | --- | --- | --- |
| `s14` | `W + sAB` | `W + sCD` | `S4 + S5` |
| `s15` | `W(I+sPQ)` | `W(I+sRS)` | `S6 + S7` |
| `s16` | `W + sAB` | `W(I+sPQ)` | `S4 + S7` |
| `s17` | `W+sAB+sWPQ` | `W(I+sRS)` | `S13 + S7` |

相关 preset：

- `screen-bilateral-tied`：`large4090` 上对 `s14-s17` 做 rank × scale 全网格。
- `screen-bilateral-tied-lowscale`：rank=8 的低 scale 追加。
- `paper-bilateral-tied`：`large` paper-scale 5 seed 固定候选诊断，含 `s1`/`s2` 控制。
- `paper-shift-mult-followup`：`large` paper-scale 5 seed，覆盖 `s18-s21` 的 shift/mult 和输出 add+mult 缺口。

## 输出目录约定

```text
runs/<group>/<model>__<variant>__r<rank>__s<scaleTag>
              [__<arch>][__wd<wdTag>][__papertest][__tf32]__seed<seed>/
```

- `scaleTag` 中 `.`→`p`、`-`→`m`，例如 `0.1`→`s0p1`、`-0.05`→`sm0p05`。
- 真正的指标文件名仍然由训练脚本生成：`ptb_<model>_<arch>_<variant>.json`，
  位于上面这个独立目录内，不会出现覆盖。
- `run.log` 是该实验的标准输出 + 错误的完整流。

## 已完成判定

调度器读取 `output_dir/ptb_<model>_<arch>_<variant>.json` 是否存在并含
`metrics.test_ppl`。已完成的任务会被自动跳过；如需强制重跑，加 `--force`。
失败任务可通过 `--retries N` 在同一张 GPU 上重试。

## 其它有用参数

- `--gpus 1,2,3,4,5,6,0` —— 加入低优先级 GPU 0（避开 7）
- `--max_parallel N` —— 并发上限（默认等于 GPU 数）
- `--include_groups paper-large,paper-bayes` —— 只跑指定 group
- `--dry_run` —— 只在日志里写 "DRY_RUN" 行，不实际启动训练
- `--summary runs/_meta/logs/<name>-summary.json` —— 全量任务的最终汇总（含 rc、elapsed、output_dir）
