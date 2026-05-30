# Results: phase8b_20260520_170927

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_05b__combined_inlmh_r1 | affine_input_lm_head_plus_hidden_lora | 1 | 608k | 1.413 | 0.000 | 652 |
| qwen25_05b__combined_inlmh_r2 | affine_input_lm_head_plus_hidden_lora | 2 | 1.16M | 1.378 | -0.035 | 665 |
| qwen25_05b__hidden_lora_r1 | hidden_lora | 1 | 550k | 1.440 | 0.027 | 648 |
| qwen25_05b__hidden_lora_r2 | hidden_lora | 2 | 1.10M | 1.398 | -0.015 | 658 |
| qwen25_05b__vocab_lora_r1 | hidden_lora | 1 | 856k | 1.413 | 0.000 | 682 |
| qwen25_05b__vocab_lora_r2 | hidden_lora | 2 | 1.41M | 1.378 | -0.035 | 696 |
| qwen3_17b__combined_inlmh_r1 | affine_input_lm_head_plus_hidden_lora | 1 | 1.22M | 0.777 | -0.636 | 1367 |
| qwen3_17b__combined_inlmh_r2 | affine_input_lm_head_plus_hidden_lora | 2 | 2.31M | 0.764 | -0.649 | 1399 |
| qwen3_17b__hidden_lora_r1 | hidden_lora | 1 | 1.09M | 0.787 | -0.626 | 1361 |
| qwen3_17b__hidden_lora_r2 | hidden_lora | 2 | 2.18M | 0.771 | -0.642 | 1392 |
| qwen3_17b__vocab_lora_r1 | hidden_lora | 1 | 1.40M | 0.785 | -0.628 | 1393 |
| qwen3_17b__vocab_lora_r2 | hidden_lora | 2 | 2.49M | 0.769 | -0.644 | 1429 |

```json
[
  {
    "run_id": "qwen25_05b__combined_inlmh_r1",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 608128,
    "total": 494640896,
    "pct": 0.12294333220680564,
    "eval_loss": 1.413,
    "train_runtime_s": 652.5,
    "train_loss": 1.487
  },
  {
    "run_id": "qwen25_05b__combined_inlmh_r2",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 1158016,
    "total": 495190784,
    "pct": 0.23385249431459532,
    "eval_loss": 1.378,
    "train_runtime_s": 665.3,
    "train_loss": 1.457
  },
  {
    "run_id": "qwen25_05b__hidden_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 549888,
    "total": 494582656,
    "pct": 0.11118222471594313,
    "eval_loss": 1.44,
    "train_runtime_s": 647.7,
    "train_loss": 1.513
  },
  {
    "run_id": "qwen25_05b__hidden_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 1099776,
    "total": 495132544,
    "pct": 0.22211749426028438,
    "eval_loss": 1.398,
    "train_runtime_s": 658.4,
    "train_loss": 1.476
  },
  {
    "run_id": "qwen25_05b__vocab_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 855552,
    "total": 494888320,
    "pct": 0.17287779190262562,
    "eval_loss": 1.413,
    "train_runtime_s": 681.8,
    "train_loss": 1.492
  },
  {
    "run_id": "qwen25_05b__vocab_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 1405440,
    "total": 495438208,
    "pct": 0.2836761431205564,
    "eval_loss": 1.378,
    "train_runtime_s": 695.8,
    "train_loss": 1.458
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r1",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 1222656,
    "total": 1721797632,
    "pct": 0.07101043567935399,
    "eval_loss": 0.7774,
    "train_runtime_s": 1367.0,
    "train_loss": 0.8311
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r2",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 2312192,
    "total": 1722887168,
    "pct": 0.134204493651438,
    "eval_loss": 0.7636,
    "train_runtime_s": 1399.0,
    "train_loss": 0.8182
  },
  {
    "run_id": "qwen3_17b__hidden_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 1089536,
    "total": 1721664512,
    "pct": 0.06328387397230617,
    "eval_loss": 0.7874,
    "train_runtime_s": 1361.0,
    "train_loss": 0.843
  },
  {
    "run_id": "qwen3_17b__hidden_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 2179072,
    "total": 1722754048,
    "pct": 0.1264877016269243,
    "eval_loss": 0.7715,
    "train_runtime_s": 1392.0,
    "train_loss": 0.8273
  },
  {
    "run_id": "qwen3_17b__vocab_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 1397504,
    "total": 1721972480,
    "pct": 0.08115716227938788,
    "eval_loss": 0.7849,
    "train_runtime_s": 1393.0,
    "train_loss": 0.8407
  },
  {
    "run_id": "qwen3_17b__vocab_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 2487040,
    "total": 1723062016,
    "pct": 0.1443383915904278,
    "eval_loss": 0.7688,
    "train_runtime_s": 1429.0,
    "train_loss": 0.8247
  }
]
```
