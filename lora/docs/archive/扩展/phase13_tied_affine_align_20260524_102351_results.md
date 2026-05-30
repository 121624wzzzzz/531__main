# Results: phase13_tied_affine_align_20260524_102351

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_05b__seed42__tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 58k | 1.633 | 0.000 | 384 |
| qwen25_05b__seed43__tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 58k | 1.631 | -0.002 | 380 |
| qwen25_05b__seed44__tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 58k | 1.629 | -0.004 | 381 |
| qwen25_15b__seed42__tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 100k | 1.307 | -0.326 | 700 |
| qwen25_15b__seed43__tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 100k | 1.310 | -0.323 | 696 |
| qwen25_15b__seed44__tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 100k | 1.310 | -0.323 | 698 |
| qwen3_06b__seed42__tied_affine_rank16_s1_64 | affine_input_lm_head | 16 | 34k | 1.350 | -0.283 | 636 |
| qwen3_06b__seed42__tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 67k | 1.318 | -0.315 | 633 |
| qwen3_06b__seed42__tied_affine_rank32_s1_64 | affine_input_lm_head | 16 | 67k | 1.317 | -0.316 | 634 |
| qwen3_06b__seed43__tied_affine_rank16_s1_64 | affine_input_lm_head | 16 | 34k | 1.351 | -0.282 | 634 |
| qwen3_06b__seed43__tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 67k | 1.319 | -0.314 | 633 |
| qwen3_06b__seed43__tied_affine_rank32_s1_64 | affine_input_lm_head | 16 | 67k | 1.318 | -0.315 | 633 |
| qwen3_06b__seed44__tied_affine_rank16_s1_64 | affine_input_lm_head | 16 | 34k | 1.351 | -0.282 | 635 |
| qwen3_06b__seed44__tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 67k | 1.321 | -0.312 | 634 |
| qwen3_06b__seed44__tied_affine_rank32_s1_64 | affine_input_lm_head | 16 | 67k | 1.320 | -0.313 | 636 |
| qwen3_17b__seed42__tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 133k | 0.911 | -0.722 | 889 |
| qwen3_17b__seed43__tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 133k | 0.914 | -0.719 | 886 |
| qwen3_17b__seed44__tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 133k | 0.911 | -0.722 | 891 |

```json
[
  {
    "run_id": "qwen25_05b__seed42__tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 42,
    "trainable": 58240,
    "total": 494091008,
    "pct": 0.011787302148190482,
    "eval_loss": 1.633,
    "train_runtime_s": 384.3,
    "train_loss": 1.689
  },
  {
    "run_id": "qwen25_05b__seed43__tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 43,
    "trainable": 58240,
    "total": 494091008,
    "pct": 0.011787302148190482,
    "eval_loss": 1.631,
    "train_runtime_s": 380.5,
    "train_loss": 1.688
  },
  {
    "run_id": "qwen25_05b__seed44__tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 44,
    "trainable": 58240,
    "total": 494091008,
    "pct": 0.011787302148190482,
    "eval_loss": 1.629,
    "train_runtime_s": 380.6,
    "train_loss": 1.687
  },
  {
    "run_id": "qwen25_15b__seed42__tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 42,
    "trainable": 99840,
    "total": 1543814144,
    "pct": 0.0064670997080850685,
    "eval_loss": 1.307,
    "train_runtime_s": 699.6,
    "train_loss": 1.366
  },
  {
    "run_id": "qwen25_15b__seed43__tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 43,
    "trainable": 99840,
    "total": 1543814144,
    "pct": 0.0064670997080850685,
    "eval_loss": 1.31,
    "train_runtime_s": 695.9,
    "train_loss": 1.368
  },
  {
    "run_id": "qwen25_15b__seed44__tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 44,
    "trainable": 99840,
    "total": 1543814144,
    "pct": 0.0064670997080850685,
    "eval_loss": 1.31,
    "train_runtime_s": 698.2,
    "train_loss": 1.368
  },
  {
    "run_id": "qwen3_06b__seed42__tied_affine_rank16_s1_64",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 33792,
    "total": 596083712,
    "pct": 0.005669002410184964,
    "eval_loss": 1.35,
    "train_runtime_s": 635.7,
    "train_loss": 1.394
  },
  {
    "run_id": "qwen3_06b__seed42__tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 42,
    "trainable": 66560,
    "total": 596116480,
    "pct": 0.011165603071399738,
    "eval_loss": 1.318,
    "train_runtime_s": 632.9,
    "train_loss": 1.36
  },
  {
    "run_id": "qwen3_06b__seed42__tied_affine_rank32_s1_64",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 2048.0,
    "seed": 42,
    "trainable": 66560,
    "total": 596116480,
    "pct": 0.011165603071399738,
    "eval_loss": 1.317,
    "train_runtime_s": 633.8,
    "train_loss": 1.369
  },
  {
    "run_id": "qwen3_06b__seed43__tied_affine_rank16_s1_64",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 43,
    "trainable": 33792,
    "total": 596083712,
    "pct": 0.005669002410184964,
    "eval_loss": 1.351,
    "train_runtime_s": 634.1,
    "train_loss": 1.395
  },
  {
    "run_id": "qwen3_06b__seed43__tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 43,
    "trainable": 66560,
    "total": 596116480,
    "pct": 0.011165603071399738,
    "eval_loss": 1.319,
    "train_runtime_s": 633.4,
    "train_loss": 1.361
  },
  {
    "run_id": "qwen3_06b__seed43__tied_affine_rank32_s1_64",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 2048.0,
    "seed": 43,
    "trainable": 66560,
    "total": 596116480,
    "pct": 0.011165603071399738,
    "eval_loss": 1.318,
    "train_runtime_s": 633.1,
    "train_loss": 1.37
  },
  {
    "run_id": "qwen3_06b__seed44__tied_affine_rank16_s1_64",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 44,
    "trainable": 33792,
    "total": 596083712,
    "pct": 0.005669002410184964,
    "eval_loss": 1.351,
    "train_runtime_s": 635.2,
    "train_loss": 1.393
  },
  {
    "run_id": "qwen3_06b__seed44__tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 44,
    "trainable": 66560,
    "total": 596116480,
    "pct": 0.011165603071399738,
    "eval_loss": 1.321,
    "train_runtime_s": 634.2,
    "train_loss": 1.361
  },
  {
    "run_id": "qwen3_06b__seed44__tied_affine_rank32_s1_64",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 2048.0,
    "seed": 44,
    "trainable": 66560,
    "total": 596116480,
    "pct": 0.011165603071399738,
    "eval_loss": 1.32,
    "train_runtime_s": 635.7,
    "train_loss": 1.369
  },
  {
    "run_id": "qwen3_17b__seed42__tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 42,
    "trainable": 133120,
    "total": 1720708096,
    "pct": 0.007736349954385291,
    "eval_loss": 0.9109,
    "train_runtime_s": 888.9,
    "train_loss": 0.9624
  },
  {
    "run_id": "qwen3_17b__seed43__tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 43,
    "trainable": 133120,
    "total": 1720708096,
    "pct": 0.007736349954385291,
    "eval_loss": 0.9138,
    "train_runtime_s": 886.0,
    "train_loss": 0.9659
  },
  {
    "run_id": "qwen3_17b__seed44__tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 44,
    "trainable": 133120,
    "total": 1720708096,
    "pct": 0.007736349954385291,
    "eval_loss": 0.9111,
    "train_runtime_s": 891.0,
    "train_loss": 0.9638
  }
]
```
