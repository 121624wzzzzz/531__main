# Results: minimind_random100k_lowrank_20260525_131318

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_15b__seed42__affine_in_lmh_r16_s1_8 | affine_input_lm_head | 16 | 100k | 1.019 | 0.000 | 4725 |
| qwen25_15b__seed42__combined_inlmh_r1_s1_8 | affine_input_lm_head_plus_hidden_lora | 1 | 1.25M | 0.956 | -0.063 | 7310 |
| qwen25_15b__seed42__combined_inlmh_r2_s1_8 | affine_input_lm_head_plus_hidden_lora | 2 | 2.41M | 0.949 | -0.070 | 7472 |
| qwen25_15b__seed42__combined_inlmh_r4_s1_8 | affine_input_lm_head_plus_hidden_lora | 4 | 4.72M | 0.941 | -0.078 | 7502 |
| qwen25_15b__seed42__hidden_lora_r1 | hidden_lora | 1 | 1.15M | 0.957 | -0.062 | 7278 |
| qwen25_15b__seed42__hidden_lora_r2 | hidden_lora | 2 | 2.31M | 0.949 | -0.070 | 7432 |
| qwen25_15b__seed42__hidden_lora_r4 | hidden_lora | 4 | 4.62M | 0.942 | -0.077 | 7460 |
| qwen25_15b__seed42__single_qkvo_mid_r4 | hidden_lora | 4 | 39k | 1.032 | 0.013 | 3657 |
| qwen3_17b__seed42__affine_in_lmh_r16_s1_8 | affine_input_lm_head | 16 | 133k | 0.880 | -0.139 | 6080 |
| qwen3_17b__seed42__combined_inlmh_r1_s1_8 | affine_input_lm_head_plus_hidden_lora | 1 | 1.22M | 0.839 | -0.180 | 8664 |
| qwen3_17b__seed42__combined_inlmh_r2_s1_8 | affine_input_lm_head_plus_hidden_lora | 2 | 2.31M | 0.833 | -0.186 | 8817 |
| qwen3_17b__seed42__combined_inlmh_r4_s1_8 | affine_input_lm_head_plus_hidden_lora | 4 | 4.49M | 0.828 | -0.191 | 8815 |
| qwen3_17b__seed42__hidden_lora_r1 | hidden_lora | 1 | 1.09M | 0.840 | -0.179 | 8615 |
| qwen3_17b__seed42__hidden_lora_r2 | hidden_lora | 2 | 2.18M | 0.834 | -0.185 | 8782 |
| qwen3_17b__seed42__hidden_lora_r4 | hidden_lora | 4 | 4.36M | 0.828 | -0.191 | 8780 |
| qwen3_17b__seed42__single_qkvo_mid_r4 | hidden_lora | 4 | 57k | 0.901 | -0.118 | 4701 |

```json
[
  {
    "run_id": "qwen25_15b__seed42__affine_in_lmh_r16_s1_8",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 99840,
    "total": 1543814144,
    "pct": 0.0064670997080850685,
    "eval_loss": 1.019,
    "train_runtime_s": 4725.0,
    "train_loss": 1.015
  },
  {
    "run_id": "qwen25_15b__seed42__combined_inlmh_r1_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 1253888,
    "total": 1544968192,
    "pct": 0.08115947023976013,
    "eval_loss": 0.9555,
    "train_runtime_s": 7310.0,
    "train_loss": 0.9588
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
    "eval_loss": 0.9485,
    "train_runtime_s": 7472.0,
    "train_loss": 0.9529
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
    "eval_loss": 0.9413,
    "train_runtime_s": 7502.0,
    "train_loss": 0.9471
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
    "eval_loss": 0.957,
    "train_runtime_s": 7278.0,
    "train_loss": 0.9595
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
    "eval_loss": 0.949,
    "train_runtime_s": 7432.0,
    "train_loss": 0.9527
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
    "eval_loss": 0.9417,
    "train_runtime_s": 7460.0,
    "train_loss": 0.9466
  },
  {
    "run_id": "qwen25_15b__seed42__single_qkvo_mid_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 38912,
    "total": 1543753216,
    "pct": 0.0025206101335823873,
    "eval_loss": 1.032,
    "train_runtime_s": 3657.0,
    "train_loss": 1.028
  },
  {
    "run_id": "qwen3_17b__seed42__affine_in_lmh_r16_s1_8",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 133120,
    "total": 1720708096,
    "pct": 0.007736349954385291,
    "eval_loss": 0.8804,
    "train_runtime_s": 6080.0,
    "train_loss": 0.8748
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
    "eval_loss": 0.8386,
    "train_runtime_s": 8664.0,
    "train_loss": 0.8359
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
    "eval_loss": 0.8331,
    "train_runtime_s": 8817.0,
    "train_loss": 0.8316
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
    "eval_loss": 0.8281,
    "train_runtime_s": 8815.0,
    "train_loss": 0.8272
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
    "eval_loss": 0.8395,
    "train_runtime_s": 8615.0,
    "train_loss": 0.8364
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
    "eval_loss": 0.8337,
    "train_runtime_s": 8782.0,
    "train_loss": 0.8312
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
    "eval_loss": 0.828,
    "train_runtime_s": 8780.0,
    "train_loss": 0.8266
  },
  {
    "run_id": "qwen3_17b__seed42__single_qkvo_mid_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 57344,
    "total": 1720632320,
    "pct": 0.0033327282844483586,
    "eval_loss": 0.9012,
    "train_runtime_s": 4701.0,
    "train_loss": 0.8942
  }
]
```
