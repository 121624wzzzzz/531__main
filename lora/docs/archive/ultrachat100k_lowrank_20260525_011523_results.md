# Results: ultrachat100k_lowrank_20260525_011523

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_15b__seed42__combined_inlmh_r1_s1_8 | affine_input_lm_head_plus_hidden_lora | 1 | 1.25M | 1.081 | 0.000 | 9541 |
| qwen25_15b__seed42__combined_inlmh_r2_s1_8 | affine_input_lm_head_plus_hidden_lora | 2 | 2.41M | 1.079 | -0.002 | 9711 |
| qwen25_15b__seed42__combined_inlmh_r4_s1_8 | affine_input_lm_head_plus_hidden_lora | 4 | 4.72M | 1.077 | -0.004 | 9745 |
| qwen25_15b__seed42__hidden_lora_r1 | hidden_lora | 1 | 1.15M | 1.081 | 0.000 | 9532 |
| qwen25_15b__seed42__hidden_lora_r2 | hidden_lora | 2 | 2.31M | 1.079 | -0.002 | 9686 |
| qwen25_15b__seed42__hidden_lora_r4 | hidden_lora | 4 | 4.62M | 1.078 | -0.003 | 9709 |
| qwen3_17b__seed42__combined_inlmh_r1_s1_8 | affine_input_lm_head_plus_hidden_lora | 1 | 1.22M | 1.064 | -0.017 | - |
| qwen3_17b__seed42__combined_inlmh_r2_s1_8 | affine_input_lm_head_plus_hidden_lora | 2 | 2.31M | 1.062 | -0.019 | - |
| qwen3_17b__seed42__combined_inlmh_r4_s1_8 | affine_input_lm_head_plus_hidden_lora | 4 | 4.49M | 1.060 | -0.021 | - |
| qwen3_17b__seed42__hidden_lora_r1 | hidden_lora | 1 | 1.09M | 1.065 | -0.016 | - |
| qwen3_17b__seed42__hidden_lora_r2 | hidden_lora | 2 | 2.18M | 1.063 | -0.018 | - |
| qwen3_17b__seed42__hidden_lora_r4 | hidden_lora | 4 | 4.36M | 1.060 | -0.021 | - |

```json
[
  {
    "run_id": "qwen25_15b__seed42__combined_inlmh_r1_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 1253888,
    "total": 1544968192,
    "pct": 0.08115947023976013,
    "eval_loss": 1.081,
    "train_runtime_s": 9541.0,
    "train_loss": 1.076
  },
  {
    "run_id": "qwen25_15b__seed42__combined_inlmh_r2_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 2407936,
    "total": 1546122240,
    "pct": 0.1557403378402991,
    "eval_loss": 1.079,
    "train_runtime_s": 9711.0,
    "train_loss": 1.075
  },
  {
    "run_id": "qwen25_15b__seed42__combined_inlmh_r4_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 4716032,
    "total": 1548430336,
    "pct": 0.30456856148806427,
    "eval_loss": 1.077,
    "train_runtime_s": 9745.0,
    "train_loss": 1.073
  },
  {
    "run_id": "qwen25_15b__seed42__hidden_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 1154048,
    "total": 1544868352,
    "pct": 0.07470202872017925,
    "eval_loss": 1.081,
    "train_runtime_s": 9532.0,
    "train_loss": 1.076
  },
  {
    "run_id": "qwen25_15b__seed42__hidden_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 2308096,
    "total": 1546022400,
    "pct": 0.14929253288956226,
    "eval_loss": 1.079,
    "train_runtime_s": 9686.0,
    "train_loss": 1.074
  },
  {
    "run_id": "qwen25_15b__seed42__hidden_lora_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 4616192,
    "total": 1548330496,
    "pct": 0.29813996507371,
    "eval_loss": 1.078,
    "train_runtime_s": 9709.0,
    "train_loss": 1.073
  },
  {
    "run_id": "qwen3_17b__seed42__combined_inlmh_r1_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 1222656,
    "total": 1721797632,
    "pct": 0.07101043567935399,
    "eval_loss": 1.064,
    "train_runtime_s": null,
    "train_loss": 1.059
  },
  {
    "run_id": "qwen3_17b__seed42__combined_inlmh_r2_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 2312192,
    "total": 1722887168,
    "pct": 0.134204493651438,
    "eval_loss": 1.062,
    "train_runtime_s": null,
    "train_loss": 1.058
  },
  {
    "run_id": "qwen3_17b__seed42__combined_inlmh_r4_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 4491264,
    "total": 1725066240,
    "pct": 0.26035313287448025,
    "eval_loss": 1.06,
    "train_runtime_s": null,
    "train_loss": 1.056
  },
  {
    "run_id": "qwen3_17b__seed42__hidden_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 1089536,
    "total": 1721664512,
    "pct": 0.06328387397230617,
    "eval_loss": 1.065,
    "train_runtime_s": null,
    "train_loss": 1.059
  },
  {
    "run_id": "qwen3_17b__seed42__hidden_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 2179072,
    "total": 1722754048,
    "pct": 0.1264877016269243,
    "eval_loss": 1.063,
    "train_runtime_s": null,
    "train_loss": 1.057
  },
  {
    "run_id": "qwen3_17b__seed42__hidden_lora_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 4358144,
    "total": 1724933120,
    "pct": 0.2526558247081487,
    "eval_loss": 1.06,
    "train_runtime_s": null,
    "train_loss": 1.055
  }
]
```
