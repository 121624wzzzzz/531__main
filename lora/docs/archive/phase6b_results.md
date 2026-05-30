# Results: phase6b_20260519_222852

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_05b__combined_inlmh_r16 | affine_input_lm_head_plus_hidden_lora | 16 | 8.86M | 1.250 | 0.000 | 655 |
| qwen25_05b__combined_inlmh_r2 | affine_input_lm_head_plus_hidden_lora | 2 | 1.16M | 1.380 | 0.130 | 664 |
| qwen25_05b__combined_inlmh_r32 | affine_input_lm_head_plus_hidden_lora | 32 | 17.65M | 1.210 | -0.040 | 614 |
| qwen25_05b__combined_inlmh_r4 | affine_input_lm_head_plus_hidden_lora | 4 | 2.26M | 1.339 | 0.089 | 625 |
| qwen25_05b__combined_inlmh_r8 | affine_input_lm_head_plus_hidden_lora | 8 | 4.46M | 1.294 | 0.044 | 655 |
| qwen25_05b__hidden_lora_r16 | hidden_lora | 16 | 8.80M | 1.254 | 0.004 | 649 |
| qwen25_05b__hidden_lora_r2 | hidden_lora | 2 | 1.10M | 1.398 | 0.148 | 622 |
| qwen25_05b__hidden_lora_r32 | hidden_lora | 32 | 17.60M | 1.211 | -0.039 | 612 |
| qwen25_05b__hidden_lora_r4 | hidden_lora | 4 | 2.20M | 1.354 | 0.104 | 658 |
| qwen25_05b__hidden_lora_r8 | hidden_lora | 8 | 4.40M | 1.302 | 0.052 | 649 |
| qwen25_7b__combined_inlmh_r16 | affine_input_lm_head_plus_hidden_lora | 16 | 40.60M | 0.899 | -0.351 | 3891 |
| qwen25_7b__combined_inlmh_r4 | affine_input_lm_head_plus_hidden_lora | 4 | 10.33M | 0.931 | -0.319 | 3934 |
| qwen25_7b__combined_inlmh_r8 | affine_input_lm_head_plus_hidden_lora | 8 | 20.42M | 0.916 | -0.334 | 3876 |
| qwen25_7b__hidden_lora_r16 | hidden_lora | 16 | 40.37M | 0.901 | -0.349 | 3904 |
| qwen25_7b__hidden_lora_r4 | hidden_lora | 4 | 10.09M | 0.937 | -0.313 | 3928 |
| qwen25_7b__hidden_lora_r8 | hidden_lora | 8 | 20.19M | 0.919 | -0.331 | 3693 |
| qwen3_17b__combined_inlmh_r16 | affine_input_lm_head_plus_hidden_lora | 16 | 17.57M | 0.721 | -0.529 | 1394 |
| qwen3_17b__combined_inlmh_r2 | affine_input_lm_head_plus_hidden_lora | 2 | 2.31M | 0.764 | -0.486 | 1408 |
| qwen3_17b__combined_inlmh_r32 | affine_input_lm_head_plus_hidden_lora | 32 | 35.00M | 0.709 | -0.541 | 1381 |
| qwen3_17b__combined_inlmh_r4 | affine_input_lm_head_plus_hidden_lora | 4 | 4.49M | 0.750 | -0.500 | 1402 |
| qwen3_17b__combined_inlmh_r8 | affine_input_lm_head_plus_hidden_lora | 8 | 8.85M | 0.735 | -0.515 | 1386 |
| qwen3_17b__hidden_lora_r16 | hidden_lora | 16 | 17.43M | 0.723 | -0.527 | 1382 |
| qwen3_17b__hidden_lora_r2 | hidden_lora | 2 | 2.18M | 0.771 | -0.479 | 1396 |
| qwen3_17b__hidden_lora_r32 | hidden_lora | 32 | 34.87M | 0.709 | -0.541 | 1373 |
| qwen3_17b__hidden_lora_r4 | hidden_lora | 4 | 4.36M | 0.754 | -0.496 | 1394 |
| qwen3_17b__hidden_lora_r8 | hidden_lora | 8 | 8.72M | 0.738 | -0.512 | 1378 |

```json
[
  {
    "run_id": "qwen25_05b__combined_inlmh_r16",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 8856448,
    "total": 502889216,
    "pct": 1.7611131275481555,
    "eval_loss": 1.25,
    "train_runtime_s": 655.4,
    "train_loss": 1.347
  },
  {
    "run_id": "qwen25_05b__combined_inlmh_r2",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 1158016,
    "total": 495190784,
    "pct": 0.23385249431459532,
    "eval_loss": 1.38,
    "train_runtime_s": 664.2,
    "train_loss": 1.457
  },
  {
    "run_id": "qwen25_05b__combined_inlmh_r32",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 32,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 17654656,
    "total": 511687424,
    "pct": 3.4502813967927417,
    "eval_loss": 1.21,
    "train_runtime_s": 613.5,
    "train_loss": 1.315
  },
  {
    "run_id": "qwen25_05b__combined_inlmh_r4",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 2257792,
    "total": 496290560,
    "pct": 0.45493349702238944,
    "eval_loss": 1.339,
    "train_runtime_s": 624.8,
    "train_loss": 1.422
  },
  {
    "run_id": "qwen25_05b__combined_inlmh_r8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 4457344,
    "total": 498490112,
    "pct": 0.8941689900561158,
    "eval_loss": 1.294,
    "train_runtime_s": 655.0,
    "train_loss": 1.384
  },
  {
    "run_id": "qwen25_05b__hidden_lora_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 8798208,
    "total": 502830976,
    "pct": 1.7497346861940342,
    "eval_loss": 1.254,
    "train_runtime_s": 648.9,
    "train_loss": 1.349
  },
  {
    "run_id": "qwen25_05b__hidden_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 1099776,
    "total": 495132544,
    "pct": 0.22211749426028438,
    "eval_loss": 1.398,
    "train_runtime_s": 621.6,
    "train_loss": 1.475
  },
  {
    "run_id": "qwen25_05b__hidden_lora_r32",
    "variant": "hidden_lora",
    "hidden_lora_rank": 32,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 17596416,
    "total": 511629184,
    "pct": 3.439290906438988,
    "eval_loss": 1.211,
    "train_runtime_s": 611.6,
    "train_loss": 1.315
  },
  {
    "run_id": "qwen25_05b__hidden_lora_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 2199552,
    "total": 496232320,
    "pct": 0.4432504517239022,
    "eval_loss": 1.354,
    "train_runtime_s": 658.3,
    "train_loss": 1.436
  },
  {
    "run_id": "qwen25_05b__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 4399104,
    "total": 498431872,
    "pct": 0.8825888244963597,
    "eval_loss": 1.302,
    "train_runtime_s": 648.8,
    "train_loss": 1.391
  },
  {
    "run_id": "qwen25_7b__combined_inlmh_r16",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 40603136,
    "total": 7656219648,
    "pct": 0.530328776690812,
    "eval_loss": 0.8995,
    "train_runtime_s": 3891.0,
    "train_loss": 0.9706
  },
  {
    "run_id": "qwen25_7b__combined_inlmh_r4",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 10325504,
    "total": 7625942016,
    "pct": 0.13539971820315505,
    "eval_loss": 0.9314,
    "train_runtime_s": 3934.0,
    "train_loss": 0.9958
  },
  {
    "run_id": "qwen25_7b__combined_inlmh_r8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 20418048,
    "total": 7636034560,
    "pct": 0.26739072275754605,
    "eval_loss": 0.9161,
    "train_runtime_s": 3876.0,
    "train_loss": 0.9832
  },
  {
    "run_id": "qwen25_7b__hidden_lora_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 40370176,
    "total": 7655986688,
    "pct": 0.5273020662807087,
    "eval_loss": 0.9008,
    "train_runtime_s": 3904.0,
    "train_loss": 0.9715
  },
  {
    "run_id": "qwen25_7b__hidden_lora_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 10092544,
    "total": 7625709056,
    "pct": 0.13234892553445982,
    "eval_loss": 0.9366,
    "train_runtime_s": 3928.0,
    "train_loss": 1.001
  },
  {
    "run_id": "qwen25_7b__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 20185088,
    "total": 7635801600,
    "pct": 0.26434798934534914,
    "eval_loss": 0.9186,
    "train_runtime_s": 3693.0,
    "train_loss": 0.9853
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r16",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 17565696,
    "total": 1738140672,
    "pct": 1.010602667722397,
    "eval_loss": 0.721,
    "train_runtime_s": 1394.0,
    "train_loss": 0.781
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r2",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 2312192,
    "total": 1722887168,
    "pct": 0.134204493651438,
    "eval_loss": 0.7642,
    "train_runtime_s": 1408.0,
    "train_loss": 0.8186
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r32",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 32,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 34998272,
    "total": 1755573248,
    "pct": 1.9935523647259406,
    "eval_loss": 0.7087,
    "train_runtime_s": 1381.0,
    "train_loss": 0.7731
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r4",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 4491264,
    "total": 1725066240,
    "pct": 0.26035313287448025,
    "eval_loss": 0.7496,
    "train_runtime_s": 1402.0,
    "train_loss": 0.8056
  },
  {
    "run_id": "qwen3_17b__combined_inlmh_r8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 8849408,
    "total": 1729424384,
    "pct": 0.5116967288001416,
    "eval_loss": 0.7347,
    "train_runtime_s": 1386.0,
    "train_loss": 0.7925
  },
  {
    "run_id": "qwen3_17b__hidden_lora_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 17432576,
    "total": 1738007552,
    "pct": 1.0030207279559622,
    "eval_loss": 0.7227,
    "train_runtime_s": 1382.0,
    "train_loss": 0.7825
  },
  {
    "run_id": "qwen3_17b__hidden_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 2179072,
    "total": 1722754048,
    "pct": 0.1264877016269243,
    "eval_loss": 0.7713,
    "train_runtime_s": 1396.0,
    "train_loss": 0.8274
  },
  {
    "run_id": "qwen3_17b__hidden_lora_r32",
    "variant": "hidden_lora",
    "hidden_lora_rank": 32,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 34865152,
    "total": 1755440128,
    "pct": 1.9861202580416344,
    "eval_loss": 0.7093,
    "train_runtime_s": 1373.0,
    "train_loss": 0.7726
  },
  {
    "run_id": "qwen3_17b__hidden_lora_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 4358144,
    "total": 1724933120,
    "pct": 0.2526558247081487,
    "eval_loss": 0.7545,
    "train_runtime_s": 1394.0,
    "train_loss": 0.8113
  },
  {
    "run_id": "qwen3_17b__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 8716288,
    "total": 1729291264,
    "pct": 0.5040381676270355,
    "eval_loss": 0.7381,
    "train_runtime_s": 1378.0,
    "train_loss": 0.7962
  }
]
```
