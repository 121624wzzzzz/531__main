# Results: phase7c_20260520_094742

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_7b__seed43__combined_inlmh_r1 | affine_input_lm_head_plus_hidden_lora | 1 | 2.76M | 0.957 | 0.000 | 3877 |
| qwen25_7b__seed43__combined_inlmh_r2 | affine_input_lm_head_plus_hidden_lora | 2 | 5.28M | 0.946 | -0.012 | 3932 |
| qwen25_7b__seed43__combined_inlmh_r4 | affine_input_lm_head_plus_hidden_lora | 4 | 10.33M | 0.932 | -0.025 | 3938 |
| qwen25_7b__seed43__hidden_lora_r1 | hidden_lora | 1 | 2.52M | 0.966 | 0.009 | 3875 |
| qwen25_7b__seed43__hidden_lora_r2 | hidden_lora | 2 | 5.05M | 0.953 | -0.004 | 3923 |
| qwen25_7b__seed43__hidden_lora_r4 | hidden_lora | 4 | 10.09M | 0.936 | -0.021 | 3937 |
| qwen25_7b__seed44__combined_inlmh_r1 | affine_input_lm_head_plus_hidden_lora | 1 | 2.76M | 0.958 | 0.001 | 3701 |
| qwen25_7b__seed44__combined_inlmh_r2 | affine_input_lm_head_plus_hidden_lora | 2 | 5.28M | 0.946 | -0.011 | 3756 |
| qwen25_7b__seed44__combined_inlmh_r4 | affine_input_lm_head_plus_hidden_lora | 4 | 10.33M | 0.931 | -0.026 | 3754 |
| qwen25_7b__seed44__hidden_lora_r1 | hidden_lora | 1 | 2.52M | 0.966 | 0.009 | 3698 |
| qwen25_7b__seed44__hidden_lora_r2 | hidden_lora | 2 | 5.05M | 0.953 | -0.004 | 3914 |
| qwen25_7b__seed44__hidden_lora_r4 | hidden_lora | 4 | 10.09M | 0.936 | -0.021 | 3945 |
| qwen3_8b__seed43__combined_inlmh_r1 | affine_input_lm_head_plus_hidden_lora | 1 | 2.99M | 0.899 | -0.058 | 4509 |
| qwen3_8b__seed43__combined_inlmh_r2 | affine_input_lm_head_plus_hidden_lora | 2 | 5.72M | 0.890 | -0.067 | 4620 |
| qwen3_8b__seed43__combined_inlmh_r4 | affine_input_lm_head_plus_hidden_lora | 4 | 11.18M | 0.880 | -0.077 | 4616 |
| qwen3_8b__seed43__hidden_lora_r1 | hidden_lora | 1 | 2.73M | 0.907 | -0.050 | 4870 |
| qwen3_8b__seed43__hidden_lora_r2 | hidden_lora | 2 | 5.46M | 0.896 | -0.061 | 4581 |
| qwen3_8b__seed43__hidden_lora_r4 | hidden_lora | 4 | 10.91M | 0.885 | -0.072 | 4607 |
| qwen3_8b__seed44__combined_inlmh_r1 | affine_input_lm_head_plus_hidden_lora | 1 | 2.99M | 0.901 | -0.056 | 4569 |
| qwen3_8b__seed44__combined_inlmh_r2 | affine_input_lm_head_plus_hidden_lora | 2 | 5.72M | 0.891 | -0.066 | 4940 |
| qwen3_8b__seed44__combined_inlmh_r4 | affine_input_lm_head_plus_hidden_lora | 4 | 11.18M | 0.881 | -0.076 | 4932 |
| qwen3_8b__seed44__hidden_lora_r1 | hidden_lora | 1 | 2.73M | 0.907 | -0.050 | 4830 |
| qwen3_8b__seed44__hidden_lora_r2 | hidden_lora | 2 | 5.46M | 0.897 | -0.060 | 4927 |
| qwen3_8b__seed44__hidden_lora_r4 | hidden_lora | 4 | 10.91M | 0.885 | -0.072 | 4924 |

```json
[
  {
    "run_id": "qwen25_7b__seed43__combined_inlmh_r1",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 43,
    "trainable": 2756096,
    "total": 7618372608,
    "pct": 0.03617696510545891,
    "eval_loss": 0.9571,
    "train_runtime_s": 3877.0,
    "train_loss": 1.02
  },
  {
    "run_id": "qwen25_7b__seed43__combined_inlmh_r2",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 43,
    "trainable": 5279232,
    "total": 7620895744,
    "pct": 0.06927311666947271,
    "eval_loss": 0.9455,
    "train_runtime_s": 3932.0,
    "train_loss": 1.009
  },
  {
    "run_id": "qwen25_7b__seed43__combined_inlmh_r4",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 128.0,
    "seed": 43,
    "trainable": 10325504,
    "total": 7625942016,
    "pct": 0.13539971820315505,
    "eval_loss": 0.9317,
    "train_runtime_s": 3938.0,
    "train_loss": 0.9976
  },
  {
    "run_id": "qwen25_7b__seed43__hidden_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 2523136,
    "total": 7618139648,
    "pct": 0.033120106962890895,
    "eval_loss": 0.9657,
    "train_runtime_s": 3875.0,
    "train_loss": 1.029
  },
  {
    "run_id": "qwen25_7b__seed43__hidden_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 5046272,
    "total": 7620662784,
    "pct": 0.06621828235983522,
    "eval_loss": 0.9528,
    "train_runtime_s": 3923.0,
    "train_loss": 1.016
  },
  {
    "run_id": "qwen25_7b__seed43__hidden_lora_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 10092544,
    "total": 7625709056,
    "pct": 0.13234892553445982,
    "eval_loss": 0.9364,
    "train_runtime_s": 3937.0,
    "train_loss": 1.002
  },
  {
    "run_id": "qwen25_7b__seed44__combined_inlmh_r1",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 44,
    "trainable": 2756096,
    "total": 7618372608,
    "pct": 0.03617696510545891,
    "eval_loss": 0.9582,
    "train_runtime_s": 3701.0,
    "train_loss": 1.019
  },
  {
    "run_id": "qwen25_7b__seed44__combined_inlmh_r2",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 44,
    "trainable": 5279232,
    "total": 7620895744,
    "pct": 0.06927311666947271,
    "eval_loss": 0.9456,
    "train_runtime_s": 3756.0,
    "train_loss": 1.008
  },
  {
    "run_id": "qwen25_7b__seed44__combined_inlmh_r4",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 128.0,
    "seed": 44,
    "trainable": 10325504,
    "total": 7625942016,
    "pct": 0.13539971820315505,
    "eval_loss": 0.9314,
    "train_runtime_s": 3754.0,
    "train_loss": 0.9963
  },
  {
    "run_id": "qwen25_7b__seed44__hidden_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 2523136,
    "total": 7618139648,
    "pct": 0.033120106962890895,
    "eval_loss": 0.9661,
    "train_runtime_s": 3698.0,
    "train_loss": 1.028
  },
  {
    "run_id": "qwen25_7b__seed44__hidden_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 5046272,
    "total": 7620662784,
    "pct": 0.06621828235983522,
    "eval_loss": 0.9534,
    "train_runtime_s": 3914.0,
    "train_loss": 1.015
  },
  {
    "run_id": "qwen25_7b__seed44__hidden_lora_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 10092544,
    "total": 7625709056,
    "pct": 0.13234892553445982,
    "eval_loss": 0.936,
    "train_runtime_s": 3945.0,
    "train_loss": 1.0
  },
  {
    "run_id": "qwen3_8b__seed43__combined_inlmh_r1",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 43,
    "trainable": 2994176,
    "total": 8193729536,
    "pct": 0.036542285010077245,
    "eval_loss": 0.8995,
    "train_runtime_s": 4509.0,
    "train_loss": 0.9491
  },
  {
    "run_id": "qwen3_8b__seed43__combined_inlmh_r2",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 43,
    "trainable": 5722112,
    "total": 8196457472,
    "pct": 0.06981201353813356,
    "eval_loss": 0.8898,
    "train_runtime_s": 4620.0,
    "train_loss": 0.9397
  },
  {
    "run_id": "qwen3_8b__seed43__combined_inlmh_r4",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 128.0,
    "seed": 43,
    "trainable": 11177984,
    "total": 8201913344,
    "pct": 0.1362850780199611,
    "eval_loss": 0.8799,
    "train_runtime_s": 4616.0,
    "train_loss": 0.9315
  },
  {
    "run_id": "qwen3_8b__seed43__hidden_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 2727936,
    "total": 8193463296,
    "pct": 0.0332940528498097,
    "eval_loss": 0.9075,
    "train_runtime_s": 4870.0,
    "train_loss": 0.9571
  },
  {
    "run_id": "qwen3_8b__seed43__hidden_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 5455872,
    "total": 8196191232,
    "pct": 0.06656594319931065,
    "eval_loss": 0.8957,
    "train_runtime_s": 4581.0,
    "train_loss": 0.9458
  },
  {
    "run_id": "qwen3_8b__seed43__hidden_lora_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 10911744,
    "total": 8201647104,
    "pct": 0.13304332485456816,
    "eval_loss": 0.8846,
    "train_runtime_s": 4607.0,
    "train_loss": 0.9353
  },
  {
    "run_id": "qwen3_8b__seed44__combined_inlmh_r1",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 44,
    "trainable": 2994176,
    "total": 8193729536,
    "pct": 0.036542285010077245,
    "eval_loss": 0.9007,
    "train_runtime_s": 4569.0,
    "train_loss": 0.9487
  },
  {
    "run_id": "qwen3_8b__seed44__combined_inlmh_r2",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 44,
    "trainable": 5722112,
    "total": 8196457472,
    "pct": 0.06981201353813356,
    "eval_loss": 0.8908,
    "train_runtime_s": 4940.0,
    "train_loss": 0.9399
  },
  {
    "run_id": "qwen3_8b__seed44__combined_inlmh_r4",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 128.0,
    "seed": 44,
    "trainable": 11177984,
    "total": 8201913344,
    "pct": 0.1362850780199611,
    "eval_loss": 0.8808,
    "train_runtime_s": 4932.0,
    "train_loss": 0.9306
  },
  {
    "run_id": "qwen3_8b__seed44__hidden_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 2727936,
    "total": 8193463296,
    "pct": 0.0332940528498097,
    "eval_loss": 0.9075,
    "train_runtime_s": 4830.0,
    "train_loss": 0.9562
  },
  {
    "run_id": "qwen3_8b__seed44__hidden_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 5455872,
    "total": 8196191232,
    "pct": 0.06656594319931065,
    "eval_loss": 0.8972,
    "train_runtime_s": 4927.0,
    "train_loss": 0.9459
  },
  {
    "run_id": "qwen3_8b__seed44__hidden_lora_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 10911744,
    "total": 8201647104,
    "pct": 0.13304332485456816,
    "eval_loss": 0.8847,
    "train_runtime_s": 4924.0,
    "train_loss": 0.9346
  }
]
```
