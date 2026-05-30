# Results: phase6d_20260520_031643

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_05b__full_finetune | full_finetune | 16 | 494.03M | 1.177 | 0.000 | 1402 |
| qwen3_06b__full_finetune | full_finetune | 16 | 596.05M | 1.019 | -0.158 | 1835 |

```json
[
  {
    "run_id": "qwen25_05b__full_finetune",
    "variant": "full_finetune",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 494032768,
    "total": 494032768,
    "pct": 100.0,
    "eval_loss": 1.177,
    "train_runtime_s": 1402.0,
    "train_loss": 1.28
  },
  {
    "run_id": "qwen3_06b__full_finetune",
    "variant": "full_finetune",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 596049920,
    "total": 596049920,
    "pct": 100.0,
    "eval_loss": 1.019,
    "train_runtime_s": 1835.0,
    "train_loss": 1.1
  }
]
```
