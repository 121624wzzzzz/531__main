# Results: phase5d_20260519_193129

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| affine_in_lmh_s1_16 | affine_input_lm_head | 16 | 233k | 1.120 | 0.202 | 2454 |
| baseline_hidden_lora_r8 | hidden_lora | 8 | 20.19M | 0.918 | 0.000 | 3805 |
| combined_in_s1_8 | affine_input_plus_hidden_lora | 8 | 20.30M | 0.918 | 0.000 | 3816 |
| combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 20.42M | 0.915 | -0.003 | 3801 |

```json
[
  {
    "run_id": "affine_in_lmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 232960,
    "total": 7615849472,
    "pct": 0.0030588839873541026,
    "eval_loss": 1.12,
    "train_runtime_s": 2454.0,
    "train_loss": 1.172
  },
  {
    "run_id": "baseline_hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 20185088,
    "total": 7635801600,
    "pct": 0.26434798934534914,
    "eval_loss": 0.918,
    "train_runtime_s": 3805.0,
    "train_loss": 0.985
  },
  {
    "run_id": "combined_in_s1_8",
    "variant": "affine_input_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 20303360,
    "total": 7635919872,
    "pct": 0.26589278489484913,
    "eval_loss": 0.9184,
    "train_runtime_s": 3816.0,
    "train_loss": 0.9852
  },
  {
    "run_id": "combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 20418048,
    "total": 7636034560,
    "pct": 0.26739072275754605,
    "eval_loss": 0.9151,
    "train_runtime_s": 3801.0,
    "train_loss": 0.9824
  }
]
```
