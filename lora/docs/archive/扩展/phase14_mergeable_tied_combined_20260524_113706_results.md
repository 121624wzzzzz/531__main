# Results: phase14_mergeable_tied_combined_20260524_113706

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_05b__seed42__mergeable_tied_combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 4.43M | 1.296 | 0.000 | 624 |
| qwen25_15b__seed42__mergeable_tied_combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 9.28M | 1.049 | -0.247 | 1137 |
| qwen3_06b__seed42__mergeable_tied_combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 5.08M | 1.098 | -0.198 | 912 |
| qwen3_17b__seed42__mergeable_tied_combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 8.78M | 0.735 | -0.561 | 1330 |

```json
[
  {
    "run_id": "qwen25_05b__seed42__mergeable_tied_combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 4428672,
    "total": 498461440,
    "pct": 0.8884683236480639,
    "eval_loss": 1.296,
    "train_runtime_s": 624.0,
    "train_loss": 1.386
  },
  {
    "run_id": "qwen25_15b__seed42__mergeable_tied_combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 9283072,
    "total": 1552997376,
    "pct": 0.5977519436581457,
    "eval_loss": 1.049,
    "train_runtime_s": 1137.0,
    "train_loss": 1.126
  },
  {
    "run_id": "qwen3_06b__seed42__mergeable_tied_combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 5080064,
    "total": 601129984,
    "pct": 0.8450857776543718,
    "eval_loss": 1.098,
    "train_runtime_s": 911.9,
    "train_loss": 1.162
  },
  {
    "run_id": "qwen3_17b__seed42__mergeable_tied_combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 8783872,
    "total": 1729358848,
    "pct": 0.5079265075700471,
    "eval_loss": 0.7349,
    "train_runtime_s": 1330.0,
    "train_loss": 0.7941
  }
]
```
