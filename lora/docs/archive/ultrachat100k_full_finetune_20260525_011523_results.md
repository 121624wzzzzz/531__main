# Results: ultrachat100k_full_finetune_20260525_011523

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_15b__seed42__full_finetune | full_finetune | 16 | 1543.71M | 1.245 | 0.000 | - |
| qwen3_17b__seed42__full_finetune | full_finetune | 16 | 1720.57M | 1.200 | -0.045 | - |

```json
[
  {
    "run_id": "qwen25_15b__seed42__full_finetune",
    "variant": "full_finetune",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 1543714304,
    "total": 1543714304,
    "pct": 100.0,
    "eval_loss": 1.245,
    "train_runtime_s": null,
    "train_loss": 1.427
  },
  {
    "run_id": "qwen3_17b__seed42__full_finetune",
    "variant": "full_finetune",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 1720574976,
    "total": 1720574976,
    "pct": 100.0,
    "eval_loss": 1.2,
    "train_runtime_s": null,
    "train_loss": 1.351
  }
]
```
