# Results: ultrachat100k_validation_20260524_172421

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_15b__seed42__affine_in_lmh_s1_16 | affine_input_lm_head | 16 | 100k | 1.099 | 0.000 | 6207 |
| qwen25_15b__seed42__combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 9.33M | 1.075 | -0.024 | 9619 |
| qwen25_15b__seed42__hidden_lora_r8 | hidden_lora | 8 | 9.23M | 1.075 | -0.024 | 9571 |
| qwen25_15b__seed42__mergeable_tied_combined_rank32_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 9.33M | 1.075 | -0.024 | 9688 |
| qwen3_17b__seed42__affine_in_lmh_s1_16 | affine_input_lm_head | 16 | 133k | 1.078 | -0.021 | 8014 |
| qwen3_17b__seed42__combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 8.85M | 1.058 | -0.041 | - |
| qwen3_17b__seed42__hidden_lora_r8 | hidden_lora | 8 | 8.72M | 1.058 | -0.041 | - |
| qwen3_17b__seed42__mergeable_tied_combined_rank32_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 8.85M | 1.058 | -0.041 | - |

```json
[
  {
    "run_id": "qwen25_15b__seed42__affine_in_lmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 99840,
    "total": 1543814144,
    "pct": 0.0064670997080850685,
    "eval_loss": 1.099,
    "train_runtime_s": 6207.0,
    "train_loss": 1.093
  },
  {
    "run_id": "qwen25_15b__seed42__combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 9332224,
    "total": 1553046528,
    "pct": 0.6008979017530117,
    "eval_loss": 1.075,
    "train_runtime_s": 9619.0,
    "train_loss": 1.072
  },
  {
    "run_id": "qwen25_15b__seed42__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 9232384,
    "total": 1552946688,
    "pct": 0.5945074657965335,
    "eval_loss": 1.075,
    "train_runtime_s": 9571.0,
    "train_loss": 1.072
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
    "eval_loss": 1.075,
    "train_runtime_s": 9688.0,
    "train_loss": 1.073
  },
  {
    "run_id": "qwen3_17b__seed42__affine_in_lmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 133120,
    "total": 1720708096,
    "pct": 0.007736349954385291,
    "eval_loss": 1.078,
    "train_runtime_s": 8014.0,
    "train_loss": 1.072
  },
  {
    "run_id": "qwen3_17b__seed42__combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 8849408,
    "total": 1729424384,
    "pct": 0.5116967288001416,
    "eval_loss": 1.058,
    "train_runtime_s": null,
    "train_loss": 1.054
  },
  {
    "run_id": "qwen3_17b__seed42__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 8716288,
    "total": 1729291264,
    "pct": 0.5040381676270355,
    "eval_loss": 1.058,
    "train_runtime_s": null,
    "train_loss": 1.054
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
    "eval_loss": 1.058,
    "train_runtime_s": null,
    "train_loss": 1.055
  }
]
```
