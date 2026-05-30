# Results: phase5c_20260519_180407

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| affine_in_lmh_s1_16 | affine_input_lm_head | 16 | 166k | 1.048 | 0.165 | 2364 |
| baseline_hidden_lora_r8 | hidden_lora | 8 | 16.52M | 0.883 | 0.000 | 3929 |
| combined_in_s1_8 | affine_input_plus_hidden_lora | 8 | 16.60M | 0.882 | -0.001 | 3926 |
| combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 16.68M | 0.881 | -0.002 | 3955 |

```json
[
  {
    "run_id": "affine_in_lmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 166400,
    "total": 4022634496,
    "pct": 0.004136592577960133,
    "eval_loss": 1.048,
    "train_runtime_s": 2364.0,
    "train_loss": 1.092
  },
  {
    "run_id": "baseline_hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 16515072,
    "total": 4038983168,
    "pct": 0.4088918253199316,
    "eval_loss": 0.8828,
    "train_runtime_s": 3929.0,
    "train_loss": 0.937
  },
  {
    "run_id": "combined_in_s1_8",
    "variant": "affine_input_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 16599552,
    "total": 4039067648,
    "pct": 0.410974844856077,
    "eval_loss": 0.8818,
    "train_runtime_s": 3926.0,
    "train_loss": 0.9366
  },
  {
    "run_id": "combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 16681472,
    "total": 4039149568,
    "pct": 0.41299465937479246,
    "eval_loss": 0.881,
    "train_runtime_s": 3955.0,
    "train_loss": 0.9356
  }
]
```
