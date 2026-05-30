# Results: phase6c_20260520_004810

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_7b__combined_inlmh_affr16 | affine_input_lm_head_plus_hidden_lora | 8 | 20.42M | 0.917 | 0.000 | 3889 |
| qwen25_7b__combined_inlmh_affr32 | affine_input_lm_head_plus_hidden_lora | 8 | 20.65M | 0.914 | -0.003 | 3877 |
| qwen25_7b__combined_inlmh_affr8 | affine_input_lm_head_plus_hidden_lora | 8 | 20.30M | 0.917 | 0.001 | 3877 |
| qwen3_17b__combined_inlmh_affr16 | affine_input_lm_head_plus_hidden_lora | 8 | 8.85M | 0.735 | -0.182 | 1382 |
| qwen3_17b__combined_inlmh_affr32 | affine_input_lm_head_plus_hidden_lora | 8 | 8.98M | 0.734 | -0.183 | 1380 |
| qwen3_17b__combined_inlmh_affr8 | affine_input_lm_head_plus_hidden_lora | 8 | 8.78M | 0.736 | -0.181 | 1382 |
| qwen3_8b__combined_inlmh_affr16 | affine_input_lm_head_plus_hidden_lora | 8 | 22.09M | 0.869 | -0.047 | 4856 |
| qwen3_8b__combined_inlmh_affr32 | affine_input_lm_head_plus_hidden_lora | 8 | 22.35M | 0.868 | -0.049 | 4857 |
| qwen3_8b__combined_inlmh_affr8 | affine_input_lm_head_plus_hidden_lora | 8 | 21.96M | 0.870 | -0.046 | 4865 |

```json
[
  {
    "run_id": "qwen25_7b__combined_inlmh_affr16",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 20418048,
    "total": 7636034560,
    "pct": 0.26739072275754605,
    "eval_loss": 0.9167,
    "train_runtime_s": 3889.0,
    "train_loss": 0.9835
  },
  {
    "run_id": "qwen25_7b__combined_inlmh_affr32",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 20647424,
    "total": 7636263936,
    "pct": 0.2703864634990008,
    "eval_loss": 0.9137,
    "train_runtime_s": 3877.0,
    "train_loss": 0.982
  },
  {
    "run_id": "qwen25_7b__combined_inlmh_affr8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 64.0,
    "seed": 42,
    "trainable": 20303360,
    "total": 7635919872,
    "pct": 0.26589278489484913,
    "eval_loss": 0.9175,
    "train_runtime_s": 3877.0,
    "train_loss": 0.9842
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_affr16",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 8849408,
    "total": 1729424384,
    "pct": 0.5116967288001416,
    "eval_loss": 0.7346,
    "train_runtime_s": 1382.0,
    "train_loss": 0.7928
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_affr32",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 8980480,
    "total": 1729555456,
    "pct": 0.5192363140971179,
    "eval_loss": 0.7338,
    "train_runtime_s": 1380.0,
    "train_loss": 0.7922
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_affr8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 64.0,
    "seed": 42,
    "trainable": 8783872,
    "total": 1729358848,
    "pct": 0.5079265075700471,
    "eval_loss": 0.736,
    "train_runtime_s": 1382.0,
    "train_loss": 0.7935
  },
  {
    "run_id": "qwen3_8b__combined_inlmh_affr16",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 22089728,
    "total": 8212825088,
    "pct": 0.2689662541611406,
    "eval_loss": 0.8692,
    "train_runtime_s": 4856.0,
    "train_loss": 0.9212
  },
  {
    "run_id": "qwen3_8b__combined_inlmh_affr32",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 22351872,
    "total": 8213087232,
    "pct": 0.2721494532885536,
    "eval_loss": 0.8677,
    "train_runtime_s": 4857.0,
    "train_loss": 0.9209
  },
  {
    "run_id": "qwen3_8b__combined_inlmh_affr8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 64.0,
    "seed": 42,
    "trainable": 21958656,
    "total": 8212694016,
    "pct": 0.2673745783931566,
    "eval_loss": 0.8703,
    "train_runtime_s": 4865.0,
    "train_loss": 0.922
  }
]
```
