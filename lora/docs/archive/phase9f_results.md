# Results: phase9f_20260521_105726

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen3_17b__combined_inlmh_r1_bias_scale0 | affine_input_lm_head_plus_hidden_lora | 1 | 1.22M | 0.778 | 0.000 | 1309 |
| qwen3_17b__combined_inlmh_r1_bias_scale0p5 | affine_input_lm_head_plus_hidden_lora | 1 | 1.22M | 0.777 | -0.000 | 1307 |
| qwen3_17b__combined_inlmh_r1_bias_scale1 | affine_input_lm_head_plus_hidden_lora | 1 | 1.22M | 0.777 | -0.001 | 1306 |
| qwen3_17b__combined_inlmh_r1_bias_scale2 | affine_input_lm_head_plus_hidden_lora | 1 | 1.22M | 0.777 | -0.000 | 1305 |
| qwen3_17b__combined_inlmh_r2_bias_scale0 | affine_input_lm_head_plus_hidden_lora | 2 | 2.31M | 0.764 | -0.013 | 1339 |
| qwen3_17b__combined_inlmh_r2_bias_scale0p5 | affine_input_lm_head_plus_hidden_lora | 2 | 2.31M | 0.764 | -0.013 | 1343 |
| qwen3_17b__combined_inlmh_r2_bias_scale1 | affine_input_lm_head_plus_hidden_lora | 2 | 2.31M | 0.764 | -0.013 | 1336 |
| qwen3_17b__combined_inlmh_r2_bias_scale2 | affine_input_lm_head_plus_hidden_lora | 2 | 2.31M | 0.764 | -0.013 | 1340 |

```json
[
  {
    "run_id": "qwen3_17b__combined_inlmh_r1_bias_scale0",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 1222656,
    "total": 1721797632,
    "pct": 0.07101043567935399,
    "eval_loss": 0.7776,
    "train_runtime_s": 1309.0,
    "train_loss": 0.8312
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r1_bias_scale0p5",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 1222656,
    "total": 1721797632,
    "pct": 0.07101043567935399,
    "eval_loss": 0.7773,
    "train_runtime_s": 1307.0,
    "train_loss": 0.8316
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r1_bias_scale1",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 1222656,
    "total": 1721797632,
    "pct": 0.07101043567935399,
    "eval_loss": 0.7767,
    "train_runtime_s": 1306.0,
    "train_loss": 0.8308
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r1_bias_scale2",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 1222656,
    "total": 1721797632,
    "pct": 0.07101043567935399,
    "eval_loss": 0.7774,
    "train_runtime_s": 1305.0,
    "train_loss": 0.8315
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r2_bias_scale0",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 2312192,
    "total": 1722887168,
    "pct": 0.134204493651438,
    "eval_loss": 0.7642,
    "train_runtime_s": 1339.0,
    "train_loss": 0.8191
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r2_bias_scale0p5",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 2312192,
    "total": 1722887168,
    "pct": 0.134204493651438,
    "eval_loss": 0.7641,
    "train_runtime_s": 1343.0,
    "train_loss": 0.8186
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r2_bias_scale1",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 2312192,
    "total": 1722887168,
    "pct": 0.134204493651438,
    "eval_loss": 0.7643,
    "train_runtime_s": 1336.0,
    "train_loss": 0.819
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r2_bias_scale2",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 2312192,
    "total": 1722887168,
    "pct": 0.134204493651438,
    "eval_loss": 0.7643,
    "train_runtime_s": 1340.0,
    "train_loss": 0.8195
  }
]
```
