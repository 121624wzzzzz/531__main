# Results: phase5b_20260519_172412

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| affine_in_lmh_s1_16 | affine_input_lm_head | 16 | 133k | 1.223 | 0.243 | 1178 |
| baseline_hidden_lora_r8 | hidden_lora | 8 | 14.97M | 0.980 | 0.000 | 1812 |
| combined_in_s1_8 | affine_input_plus_hidden_lora | 8 | 15.03M | 0.979 | -0.001 | 1819 |
| combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 15.10M | 0.976 | -0.004 | 1818 |

```json
[
  {
    "run_id": "affine_in_lmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 133120,
    "total": 3086071808,
    "pct": 0.004313574287380937,
    "eval_loss": 1.223,
    "train_runtime_s": 1178.0,
    "train_loss": 1.276
  },
  {
    "run_id": "baseline_hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 14966784,
    "total": 3100905472,
    "pct": 0.4826585052380468,
    "eval_loss": 0.9797,
    "train_runtime_s": 1812.0,
    "train_loss": 1.051
  },
  {
    "run_id": "combined_in_s1_8",
    "variant": "affine_input_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 15034368,
    "total": 3100973056,
    "pct": 0.4848274308901316,
    "eval_loss": 0.9789,
    "train_runtime_s": 1819.0,
    "train_loss": 1.049
  },
  {
    "run_id": "combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 15099904,
    "total": 3101038592,
    "pct": 0.48693054123719853,
    "eval_loss": 0.9762,
    "train_runtime_s": 1818.0,
    "train_loss": 1.046
  }
]
```
