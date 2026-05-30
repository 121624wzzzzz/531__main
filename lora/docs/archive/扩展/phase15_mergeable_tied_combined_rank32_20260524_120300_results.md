# Results: phase15_mergeable_tied_combined_rank32_20260524_120300

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_05b__seed42__mergeable_tied_combined_rank32_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 4.46M | 1.293 | 0.000 | 626 |
| qwen25_15b__seed42__mergeable_tied_combined_rank32_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 9.33M | 1.047 | -0.246 | 1139 |
| qwen3_06b__seed42__mergeable_tied_combined_rank32_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 5.11M | 1.096 | -0.197 | 914 |
| qwen3_17b__seed42__mergeable_tied_combined_rank32_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 8.85M | 0.733 | -0.560 | 1327 |

```json
[
  {
    "run_id": "qwen25_05b__seed42__mergeable_tied_combined_rank32_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 4457344,
    "total": 498490112,
    "pct": 0.8941689900561158,
    "eval_loss": 1.293,
    "train_runtime_s": 625.8,
    "train_loss": 1.384
  },
  {
    "run_id": "qwen25_15b__seed42__mergeable_tied_combined_rank32_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 9332224,
    "total": 1553046528,
    "pct": 0.6008979017530117,
    "eval_loss": 1.047,
    "train_runtime_s": 1139.0,
    "train_loss": 1.125
  },
  {
    "run_id": "qwen3_06b__seed42__mergeable_tied_combined_rank32_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 5112832,
    "total": 601162752,
    "pct": 0.8504904841476273,
    "eval_loss": 1.096,
    "train_runtime_s": 914.5,
    "train_loss": 1.161
  },
  {
    "run_id": "qwen3_17b__seed42__mergeable_tied_combined_rank32_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 8849408,
    "total": 1729424384,
    "pct": 0.5116967288001416,
    "eval_loss": 0.7333,
    "train_runtime_s": 1327.0,
    "train_loss": 0.7938
  }
]
```
