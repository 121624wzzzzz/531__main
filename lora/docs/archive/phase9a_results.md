# Results: phase9a_20260520_184329

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_7b__seed43__hidden_r1_vocab_lora_r1 | hidden_lora | 1 | 2.83M | 0.964 | 0.000 | 3882 |
| qwen25_7b__seed43__hidden_r2_vocab_lora_r1 | hidden_lora | 2 | 5.36M | 0.950 | -0.014 | 3957 |
| qwen25_7b__seed44__hidden_r1_vocab_lora_r1 | hidden_lora | 1 | 2.83M | 0.964 | -0.000 | 3818 |
| qwen25_7b__seed44__hidden_r2_vocab_lora_r1 | hidden_lora | 2 | 5.36M | 0.951 | -0.014 | 3873 |
| qwen3_8b__seed43__hidden_r1_vocab_lora_r1 | hidden_lora | 1 | 3.04M | 0.906 | -0.058 | 4826 |
| qwen3_8b__seed43__hidden_r2_vocab_lora_r1 | hidden_lora | 2 | 5.77M | 0.896 | -0.068 | 4911 |
| qwen3_8b__seed44__hidden_r1_vocab_lora_r1 | hidden_lora | 1 | 3.04M | 0.906 | -0.058 | 4761 |
| qwen3_8b__seed44__hidden_r2_vocab_lora_r1 | hidden_lora | 2 | 5.77M | 0.896 | -0.068 | 4770 |

```json
[
  {
    "run_id": "qwen25_7b__seed43__hidden_r1_vocab_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 2834432,
    "total": 7618450944,
    "pct": 0.0372048336444601,
    "eval_loss": 0.9641,
    "train_runtime_s": 3882.0,
    "train_loss": 1.027
  },
  {
    "run_id": "qwen25_7b__seed43__hidden_r2_vocab_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 5357568,
    "total": 7620974080,
    "pct": 0.07030030470855506,
    "eval_loss": 0.9504,
    "train_runtime_s": 3957.0,
    "train_loss": 1.014
  },
  {
    "run_id": "qwen25_7b__seed44__hidden_r1_vocab_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 2834432,
    "total": 7618450944,
    "pct": 0.0372048336444601,
    "eval_loss": 0.9638,
    "train_runtime_s": 3818.0,
    "train_loss": 1.026
  },
  {
    "run_id": "qwen25_7b__seed44__hidden_r2_vocab_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 5357568,
    "total": 7620974080,
    "pct": 0.07030030470855506,
    "eval_loss": 0.9505,
    "train_runtime_s": 3873.0,
    "train_loss": 1.013
  },
  {
    "run_id": "qwen3_8b__seed43__hidden_r1_vocab_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 3040000,
    "total": 8193775360,
    "pct": 0.03710133444517571,
    "eval_loss": 0.9063,
    "train_runtime_s": 4826.0,
    "train_loss": 0.9561
  },
  {
    "run_id": "qwen3_8b__seed43__hidden_r2_vocab_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 5767936,
    "total": 8196503296,
    "pct": 0.07037069091175535,
    "eval_loss": 0.8959,
    "train_runtime_s": 4911.0,
    "train_loss": 0.9458
  },
  {
    "run_id": "qwen3_8b__seed44__hidden_r1_vocab_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 3040000,
    "total": 8193775360,
    "pct": 0.03710133444517571,
    "eval_loss": 0.9061,
    "train_runtime_s": 4761.0,
    "train_loss": 0.9556
  },
  {
    "run_id": "qwen3_8b__seed44__hidden_r2_vocab_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 5767936,
    "total": 8196503296,
    "pct": 0.07037069091175535,
    "eval_loss": 0.8964,
    "train_runtime_s": 4770.0,
    "train_loss": 0.9454
  }
]
```
