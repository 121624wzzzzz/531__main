param(
  [string]$Python = "D:\DevTools\anaconda3\envs\wzdt\python.exe",
  [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

& $Python "$Root\repro_pytorch\train_ptb.py" `
  --data_path "$Root\data\ptb" `
  --model test `
  --variant wt_pr `
  --architecture zaremba `
  --device $Device `
  --max_epochs 1 `
  --max_train_batches 5 `
  --max_eval_batches 20 `
  --quiet

& $Python "$Root\repro_pytorch\train_ptb.py" `
  --data_path "$Root\data\ptb" `
  --model test `
  --variant wt `
  --architecture variational `
  --device $Device `
  --max_epochs 1 `
  --max_train_batches 2 `
  --max_eval_batches 5 `
  --quiet
