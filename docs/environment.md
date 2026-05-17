# Local Environment Capture

Conda environment requested by the user:

```text
D:\DevTools\anaconda3\envs\wzdt
```

Observed runtime:

```text
Python: 3.9.23
PyTorch: 2.7.1+cu128
NumPy: 1.26.4
CUDA available: true
CUDA runtime reported by PyTorch: 12.8
CUDA device count: 1
```

Fetched source and data:

```text
Official repository: https://github.com/ofirpress/UsingTheOutputEmbedding
Checked out commit: 5675180
Modernization target: PyTorch-only code in repro_pytorch/
PTB archive: simple-examples.tgz
PTB data path: data/ptb
PTB vocab size: 10000
Train tokens: 929589
Valid tokens: 73760
Test tokens: 82430
```

Validated commands:

```powershell
.\scripts\run_smoke.ps1

& "D:\DevTools\anaconda3\envs\wzdt\python.exe" .\repro_pytorch\train_ptb.py `
  --data_path .\data\ptb `
  --model test `
  --variant baseline `
  --device auto `
  --max_epochs 1 `
  --max_train_batches 2 `
  --max_eval_batches 2 `
  --quiet

& "D:\DevTools\anaconda3\envs\wzdt\python.exe" .\repro_pytorch\train_ptb.py `
  --data_path .\data\ptb `
  --model test `
  --variant wt `
  --device auto `
  --max_epochs 1 `
  --max_train_batches 2 `
  --max_eval_batches 2 `
  --quiet

& "D:\DevTools\anaconda3\envs\wzdt\python.exe" .\repro_pytorch\train_ptb.py `
  --data_path .\data\ptb `
  --model test `
  --variant wt_pr `
  --device auto `
  --max_epochs 1 `
  --max_train_batches 2 `
  --max_eval_batches 2 `
  --quiet
```
