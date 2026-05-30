# Results: phase6a_20260519_204314

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_05b__affine_in_lmh_s1_16 | affine_input_lm_head | 16 | 58k | 1.637 | 0.000 | 387 |
| qwen25_05b__combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 4.46M | 1.294 | -0.343 | 629 |
| qwen25_05b__hidden_lora_r8 | hidden_lora | 8 | 4.40M | 1.302 | -0.335 | 626 |
| qwen3_17b__affine_in_lmh_s1_16 | affine_input_lm_head | 16 | 133k | 0.907 | -0.730 | 902 |
| qwen3_17b__combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 8.85M | 0.735 | -0.902 | 1350 |
| qwen3_17b__hidden_lora_r8 | hidden_lora | 8 | 8.72M | 0.738 | -0.899 | 1351 |
| qwen3_8b__affine_in_lmh_s1_16 | affine_input_lm_head | 16 | 266k | 1.010 | -0.627 | 3133 |
| qwen3_8b__combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 22.09M | 0.870 | -0.767 | 4830 |
| qwen3_8b__hidden_lora_r8 | hidden_lora | 8 | 21.82M | 0.872 | -0.765 | 4807 |

```json
[
  {
    "run_id": "qwen25_05b__affine_in_lmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 58240,
    "total": 494091008,
    "pct": 0.011787302148190482,
    "eval_loss": 1.637,
    "train_runtime_s": 387.2,
    "train_loss": 1.693
  },
  {
    "run_id": "qwen25_05b__combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 4457344,
    "total": 498490112,
    "pct": 0.8941689900561158,
    "eval_loss": 1.294,
    "train_runtime_s": 629.2,
    "train_loss": 1.384
  },
  {
    "run_id": "qwen25_05b__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 4399104,
    "total": 498431872,
    "pct": 0.8825888244963597,
    "eval_loss": 1.302,
    "train_runtime_s": 625.7,
    "train_loss": 1.391
  },
  {
    "run_id": "qwen3_17b__affine_in_lmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 133120,
    "total": 1720708096,
    "pct": 0.007736349954385291,
    "eval_loss": 0.9074,
    "train_runtime_s": 902.4,
    "train_loss": 0.9576
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 8849408,
    "total": 1729424384,
    "pct": 0.5116967288001416,
    "eval_loss": 0.735,
    "train_runtime_s": 1350.0,
    "train_loss": 0.7926
  },
  {
    "run_id": "qwen3_17b__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 8716288,
    "total": 1729291264,
    "pct": 0.5040381676270355,
    "eval_loss": 0.7383,
    "train_runtime_s": 1351.0,
    "train_loss": 0.796
  },
  {
    "run_id": "qwen3_8b__affine_in_lmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 266240,
    "total": 8191001600,
    "pct": 0.00325039614202981,
    "eval_loss": 1.01,
    "train_runtime_s": 3133.0,
    "train_loss": 1.055
  },
  {
    "run_id": "qwen3_8b__combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 22089728,
    "total": 8212825088,
    "pct": 0.2689662541611406,
    "eval_loss": 0.8696,
    "train_runtime_s": 4830.0,
    "train_loss": 0.9213
  },
  {
    "run_id": "qwen3_8b__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 21823488,
    "total": 8212558848,
    "pct": 0.26573310954495827,
    "eval_loss": 0.8717,
    "train_runtime_s": 4807.0,
    "train_loss": 0.923
  }
]
```
