# Results: phase9c_20260520_214840

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_7b__combined_inlmh_r1_affine_alpha128 | affine_input_lm_head_plus_hidden_lora | 1 | 2.76M | 0.958 | 0.000 | 3700 |
| qwen25_7b__combined_inlmh_r1_affine_alpha256 | affine_input_lm_head_plus_hidden_lora | 1 | 2.76M | 0.956 | -0.002 | 3850 |
| qwen25_7b__combined_inlmh_r1_affine_alpha64 | affine_input_lm_head_plus_hidden_lora | 1 | 2.76M | 0.959 | 0.001 | 3880 |
| qwen25_7b__combined_inlmh_r2_affine_alpha128 | affine_input_lm_head_plus_hidden_lora | 2 | 5.28M | 0.946 | -0.012 | 3777 |
| qwen25_7b__combined_inlmh_r2_affine_alpha256 | affine_input_lm_head_plus_hidden_lora | 2 | 5.28M | 0.944 | -0.014 | 3923 |
| qwen25_7b__combined_inlmh_r2_affine_alpha64 | affine_input_lm_head_plus_hidden_lora | 2 | 5.28M | 0.947 | -0.011 | 3758 |
| qwen3_17b__combined_inlmh_r1_affine_alpha128 | affine_input_lm_head_plus_hidden_lora | 1 | 1.22M | 0.777 | -0.180 | 1367 |
| qwen3_17b__combined_inlmh_r1_affine_alpha256 | affine_input_lm_head_plus_hidden_lora | 1 | 1.22M | 0.776 | -0.182 | 1363 |
| qwen3_17b__combined_inlmh_r1_affine_alpha64 | affine_input_lm_head_plus_hidden_lora | 1 | 1.22M | 0.779 | -0.179 | 1366 |
| qwen3_17b__combined_inlmh_r2_affine_alpha128 | affine_input_lm_head_plus_hidden_lora | 2 | 2.31M | 0.764 | -0.194 | 1395 |
| qwen3_17b__combined_inlmh_r2_affine_alpha256 | affine_input_lm_head_plus_hidden_lora | 2 | 2.31M | 0.763 | -0.195 | 1393 |
| qwen3_17b__combined_inlmh_r2_affine_alpha64 | affine_input_lm_head_plus_hidden_lora | 2 | 2.31M | 0.766 | -0.192 | 1393 |

```json
[
  {
    "run_id": "qwen25_7b__combined_inlmh_r1_affine_alpha128",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 2756096,
    "total": 7618372608,
    "pct": 0.03617696510545891,
    "eval_loss": 0.9578,
    "train_runtime_s": 3700.0,
    "train_loss": 1.019
  },
  {
    "run_id": "qwen25_7b__combined_inlmh_r1_affine_alpha256",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 2756096,
    "total": 7618372608,
    "pct": 0.03617696510545891,
    "eval_loss": 0.9558,
    "train_runtime_s": 3850.0,
    "train_loss": 1.018
  },
  {
    "run_id": "qwen25_7b__combined_inlmh_r1_affine_alpha64",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 64.0,
    "seed": 42,
    "trainable": 2756096,
    "total": 7618372608,
    "pct": 0.03617696510545891,
    "eval_loss": 0.9591,
    "train_runtime_s": 3880.0,
    "train_loss": 1.02
  },
  {
    "run_id": "qwen25_7b__combined_inlmh_r2_affine_alpha128",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 5279232,
    "total": 7620895744,
    "pct": 0.06927311666947271,
    "eval_loss": 0.9457,
    "train_runtime_s": 3777.0,
    "train_loss": 1.008
  },
  {
    "run_id": "qwen25_7b__combined_inlmh_r2_affine_alpha256",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 5279232,
    "total": 7620895744,
    "pct": 0.06927311666947271,
    "eval_loss": 0.9437,
    "train_runtime_s": 3923.0,
    "train_loss": 1.007
  },
  {
    "run_id": "qwen25_7b__combined_inlmh_r2_affine_alpha64",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 64.0,
    "seed": 42,
    "trainable": 5279232,
    "total": 7620895744,
    "pct": 0.06927311666947271,
    "eval_loss": 0.9472,
    "train_runtime_s": 3758.0,
    "train_loss": 1.01
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r1_affine_alpha128",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 1222656,
    "total": 1721797632,
    "pct": 0.07101043567935399,
    "eval_loss": 0.7773,
    "train_runtime_s": 1367.0,
    "train_loss": 0.8317
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r1_affine_alpha256",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 1222656,
    "total": 1721797632,
    "pct": 0.07101043567935399,
    "eval_loss": 0.776,
    "train_runtime_s": 1363.0,
    "train_loss": 0.8307
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r1_affine_alpha64",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 64.0,
    "seed": 42,
    "trainable": 1222656,
    "total": 1721797632,
    "pct": 0.07101043567935399,
    "eval_loss": 0.7789,
    "train_runtime_s": 1366.0,
    "train_loss": 0.8327
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r2_affine_alpha128",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 2312192,
    "total": 1722887168,
    "pct": 0.134204493651438,
    "eval_loss": 0.7643,
    "train_runtime_s": 1395.0,
    "train_loss": 0.819
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r2_affine_alpha256",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 2312192,
    "total": 1722887168,
    "pct": 0.134204493651438,
    "eval_loss": 0.7633,
    "train_runtime_s": 1393.0,
    "train_loss": 0.8187
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r2_affine_alpha64",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 64.0,
    "seed": 42,
    "trainable": 2312192,
    "total": 1722887168,
    "pct": 0.134204493651438,
    "eval_loss": 0.7661,
    "train_runtime_s": 1393.0,
    "train_loss": 0.8203
  }
]
```
