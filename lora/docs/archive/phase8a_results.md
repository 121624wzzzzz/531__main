# Results: phase8a_20260520_152912

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_7b__hidden_r1_vocab_lora_r1 | hidden_lora | 1 | 2.83M | 0.964 | 0.000 | 3725 |
| qwen25_7b__hidden_r2_vocab_lora_r1 | hidden_lora | 2 | 5.36M | 0.950 | -0.014 | 3786 |
| qwen3_8b__hidden_r1_vocab_lora_r1 | hidden_lora | 1 | 3.04M | 0.906 | -0.058 | 4854 |
| qwen3_8b__hidden_r2_vocab_lora_r1 | hidden_lora | 2 | 5.77M | 0.897 | -0.067 | 4629 |

```json
[
  {
    "run_id": "qwen25_7b__hidden_r1_vocab_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 2834432,
    "total": 7618450944,
    "pct": 0.0372048336444601,
    "eval_loss": 0.9637,
    "train_runtime_s": 3725.0,
    "train_loss": 1.026
  },
  {
    "run_id": "qwen25_7b__hidden_r2_vocab_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 5357568,
    "total": 7620974080,
    "pct": 0.07030030470855506,
    "eval_loss": 0.95,
    "train_runtime_s": 3786.0,
    "train_loss": 1.012
  },
  {
    "run_id": "qwen3_8b__hidden_r1_vocab_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 3040000,
    "total": 8193775360,
    "pct": 0.03710133444517571,
    "eval_loss": 0.9061,
    "train_runtime_s": 4854.0,
    "train_loss": 0.9553
  },
  {
    "run_id": "qwen3_8b__hidden_r2_vocab_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 5767936,
    "total": 8196503296,
    "pct": 0.07037069091175535,
    "eval_loss": 0.8967,
    "train_runtime_s": 4629.0,
    "train_loss": 0.9453
  }
]
```
