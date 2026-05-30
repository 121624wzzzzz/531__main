# Results: phase11_large_single_layer_20260523_130101

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_7b__seed42__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 233k | 1.119 | 0.000 | 2453 |
| qwen25_7b__seed42__affine_input_s1_64 | affine_input | 16 | 118k | 1.227 | 0.108 | 2465 |
| qwen25_7b__seed42__single_q_l0_r16 | hidden_lora | 16 | 115k | 1.553 | 0.434 | 2462 |
| qwen25_7b__seed42__single_q_l14_r16 | hidden_lora | 16 | 115k | 1.514 | 0.395 | 2461 |
| qwen25_7b__seed42__single_q_l27_r16 | hidden_lora | 16 | 115k | 1.616 | 0.497 | 2434 |
| qwen25_7b__seed42__single_qkvo_l14_r4 | hidden_lora | 4 | 90k | 1.259 | 0.140 | 2444 |
| qwen25_7b__seed43__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 233k | 1.118 | -0.001 | 2441 |
| qwen25_7b__seed43__affine_input_s1_64 | affine_input | 16 | 118k | 1.232 | 0.113 | 2451 |
| qwen25_7b__seed43__single_q_l0_r16 | hidden_lora | 16 | 115k | 1.555 | 0.436 | 2450 |
| qwen25_7b__seed43__single_q_l14_r16 | hidden_lora | 16 | 115k | 1.514 | 0.395 | 2450 |
| qwen25_7b__seed43__single_q_l27_r16 | hidden_lora | 16 | 115k | 1.617 | 0.498 | 2423 |
| qwen25_7b__seed43__single_qkvo_l14_r4 | hidden_lora | 4 | 90k | 1.257 | 0.138 | 2431 |
| qwen25_7b__seed44__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 233k | 1.120 | 0.001 | 2446 |
| qwen25_7b__seed44__affine_input_s1_64 | affine_input | 16 | 118k | 1.233 | 0.114 | 2460 |
| qwen25_7b__seed44__single_q_l0_r16 | hidden_lora | 16 | 115k | 1.553 | 0.434 | 2456 |
| qwen25_7b__seed44__single_q_l14_r16 | hidden_lora | 16 | 115k | 1.516 | 0.397 | 2453 |
| qwen25_7b__seed44__single_q_l27_r16 | hidden_lora | 16 | 115k | 1.618 | 0.499 | 2429 |
| qwen25_7b__seed44__single_qkvo_l14_r4 | hidden_lora | 4 | 90k | 1.253 | 0.134 | 2437 |
| qwen3_8b__seed42__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 266k | 1.008 | -0.111 | 3091 |
| qwen3_8b__seed42__affine_input_s1_64 | affine_input | 16 | 135k | 1.065 | -0.054 | 3097 |
| qwen3_8b__seed42__single_q_l0_r16 | hidden_lora | 16 | 131k | 1.347 | 0.228 | 3098 |
| qwen3_8b__seed42__single_q_l18_r16 | hidden_lora | 16 | 131k | 1.308 | 0.189 | 3097 |
| qwen3_8b__seed42__single_q_l35_r16 | hidden_lora | 16 | 131k | 1.379 | 0.260 | 3069 |
| qwen3_8b__seed42__single_qkvo_l18_r4 | hidden_lora | 4 | 106k | 1.120 | 0.001 | 3079 |
| qwen3_8b__seed43__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 266k | 1.011 | -0.108 | 3073 |
| qwen3_8b__seed43__affine_input_s1_64 | affine_input | 16 | 135k | 1.068 | -0.051 | 3087 |
| qwen3_8b__seed43__single_q_l0_r16 | hidden_lora | 16 | 131k | 1.354 | 0.235 | 3086 |
| qwen3_8b__seed43__single_q_l18_r16 | hidden_lora | 16 | 131k | 1.309 | 0.190 | 3088 |
| qwen3_8b__seed43__single_q_l35_r16 | hidden_lora | 16 | 131k | 1.380 | 0.261 | 3055 |
| qwen3_8b__seed43__single_qkvo_l18_r4 | hidden_lora | 4 | 106k | 1.118 | -0.001 | 3063 |
| qwen3_8b__seed44__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 266k | 1.009 | -0.110 | 3092 |
| qwen3_8b__seed44__affine_input_s1_64 | affine_input | 16 | 135k | 1.066 | -0.053 | 3108 |
| qwen3_8b__seed44__single_q_l0_r16 | hidden_lora | 16 | 131k | 1.350 | 0.231 | 3096 |
| qwen3_8b__seed44__single_q_l18_r16 | hidden_lora | 16 | 131k | 1.307 | 0.188 | 3109 |
| qwen3_8b__seed44__single_q_l35_r16 | hidden_lora | 16 | 131k | 1.378 | 0.259 | 3070 |
| qwen3_8b__seed44__single_qkvo_l18_r4 | hidden_lora | 4 | 106k | 1.121 | 0.002 | 3090 |

```json
[
  {
    "run_id": "qwen25_7b__seed42__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 232960,
    "total": 7615849472,
    "pct": 0.0030588839873541026,
    "eval_loss": 1.119,
    "train_runtime_s": 2453.0,
    "train_loss": 1.172
  },
  {
    "run_id": "qwen25_7b__seed42__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 118272,
    "total": 7615734784,
    "pct": 0.0015529952572466053,
    "eval_loss": 1.227,
    "train_runtime_s": 2465.0,
    "train_loss": 1.276
  },
  {
    "run_id": "qwen25_7b__seed42__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.553,
    "train_runtime_s": 2462.0,
    "train_loss": 1.587
  },
  {
    "run_id": "qwen25_7b__seed42__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.514,
    "train_runtime_s": 2461.0,
    "train_loss": 1.55
  },
  {
    "run_id": "qwen25_7b__seed42__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.616,
    "train_runtime_s": 2434.0,
    "train_loss": 1.644
  },
  {
    "run_id": "qwen25_7b__seed42__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 90112,
    "total": 7615706624,
    "pct": 0.0011832388568648728,
    "eval_loss": 1.259,
    "train_runtime_s": 2444.0,
    "train_loss": 1.31
  },
  {
    "run_id": "qwen25_7b__seed43__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 43,
    "trainable": 232960,
    "total": 7615849472,
    "pct": 0.0030588839873541026,
    "eval_loss": 1.118,
    "train_runtime_s": 2441.0,
    "train_loss": 1.17
  },
  {
    "run_id": "qwen25_7b__seed43__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 43,
    "trainable": 118272,
    "total": 7615734784,
    "pct": 0.0015529952572466053,
    "eval_loss": 1.232,
    "train_runtime_s": 2451.0,
    "train_loss": 1.281
  },
  {
    "run_id": "qwen25_7b__seed43__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.555,
    "train_runtime_s": 2450.0,
    "train_loss": 1.588
  },
  {
    "run_id": "qwen25_7b__seed43__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.514,
    "train_runtime_s": 2450.0,
    "train_loss": 1.551
  },
  {
    "run_id": "qwen25_7b__seed43__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.617,
    "train_runtime_s": 2423.0,
    "train_loss": 1.646
  },
  {
    "run_id": "qwen25_7b__seed43__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 90112,
    "total": 7615706624,
    "pct": 0.0011832388568648728,
    "eval_loss": 1.257,
    "train_runtime_s": 2431.0,
    "train_loss": 1.305
  },
  {
    "run_id": "qwen25_7b__seed44__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 44,
    "trainable": 232960,
    "total": 7615849472,
    "pct": 0.0030588839873541026,
    "eval_loss": 1.12,
    "train_runtime_s": 2446.0,
    "train_loss": 1.172
  },
  {
    "run_id": "qwen25_7b__seed44__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 44,
    "trainable": 118272,
    "total": 7615734784,
    "pct": 0.0015529952572466053,
    "eval_loss": 1.233,
    "train_runtime_s": 2460.0,
    "train_loss": 1.281
  },
  {
    "run_id": "qwen25_7b__seed44__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.553,
    "train_runtime_s": 2456.0,
    "train_loss": 1.586
  },
  {
    "run_id": "qwen25_7b__seed44__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.516,
    "train_runtime_s": 2453.0,
    "train_loss": 1.551
  },
  {
    "run_id": "qwen25_7b__seed44__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.618,
    "train_runtime_s": 2429.0,
    "train_loss": 1.646
  },
  {
    "run_id": "qwen25_7b__seed44__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 90112,
    "total": 7615706624,
    "pct": 0.0011832388568648728,
    "eval_loss": 1.253,
    "train_runtime_s": 2437.0,
    "train_loss": 1.303
  },
  {
    "run_id": "qwen3_8b__seed42__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 266240,
    "total": 8191001600,
    "pct": 0.00325039614202981,
    "eval_loss": 1.008,
    "train_runtime_s": 3091.0,
    "train_loss": 1.052
  },
  {
    "run_id": "qwen3_8b__seed42__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 135168,
    "total": 8190870528,
    "pct": 0.001650227525120026,
    "eval_loss": 1.065,
    "train_runtime_s": 3097.0,
    "train_loss": 1.108
  },
  {
    "run_id": "qwen3_8b__seed42__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.347,
    "train_runtime_s": 3098.0,
    "train_loss": 1.371
  },
  {
    "run_id": "qwen3_8b__seed42__single_q_l18_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.308,
    "train_runtime_s": 3097.0,
    "train_loss": 1.337
  },
  {
    "run_id": "qwen3_8b__seed42__single_q_l35_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.379,
    "train_runtime_s": 3069.0,
    "train_loss": 1.4
  },
  {
    "run_id": "qwen3_8b__seed42__single_qkvo_l18_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 106496,
    "total": 8190841856,
    "pct": 0.0013001838134866317,
    "eval_loss": 1.12,
    "train_runtime_s": 3079.0,
    "train_loss": 1.157
  },
  {
    "run_id": "qwen3_8b__seed43__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 43,
    "trainable": 266240,
    "total": 8191001600,
    "pct": 0.00325039614202981,
    "eval_loss": 1.011,
    "train_runtime_s": 3073.0,
    "train_loss": 1.055
  },
  {
    "run_id": "qwen3_8b__seed43__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 43,
    "trainable": 135168,
    "total": 8190870528,
    "pct": 0.001650227525120026,
    "eval_loss": 1.068,
    "train_runtime_s": 3087.0,
    "train_loss": 1.112
  },
  {
    "run_id": "qwen3_8b__seed43__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.354,
    "train_runtime_s": 3086.0,
    "train_loss": 1.375
  },
  {
    "run_id": "qwen3_8b__seed43__single_q_l18_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.309,
    "train_runtime_s": 3088.0,
    "train_loss": 1.338
  },
  {
    "run_id": "qwen3_8b__seed43__single_q_l35_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.38,
    "train_runtime_s": 3055.0,
    "train_loss": 1.401
  },
  {
    "run_id": "qwen3_8b__seed43__single_qkvo_l18_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 106496,
    "total": 8190841856,
    "pct": 0.0013001838134866317,
    "eval_loss": 1.118,
    "train_runtime_s": 3063.0,
    "train_loss": 1.157
  },
  {
    "run_id": "qwen3_8b__seed44__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 44,
    "trainable": 266240,
    "total": 8191001600,
    "pct": 0.00325039614202981,
    "eval_loss": 1.009,
    "train_runtime_s": 3092.0,
    "train_loss": 1.054
  },
  {
    "run_id": "qwen3_8b__seed44__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 44,
    "trainable": 135168,
    "total": 8190870528,
    "pct": 0.001650227525120026,
    "eval_loss": 1.066,
    "train_runtime_s": 3108.0,
    "train_loss": 1.108
  },
  {
    "run_id": "qwen3_8b__seed44__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.35,
    "train_runtime_s": 3096.0,
    "train_loss": 1.372
  },
  {
    "run_id": "qwen3_8b__seed44__single_q_l18_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.307,
    "train_runtime_s": 3109.0,
    "train_loss": 1.336
  },
  {
    "run_id": "qwen3_8b__seed44__single_q_l35_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.378,
    "train_runtime_s": 3070.0,
    "train_loss": 1.399
  },
  {
    "run_id": "qwen3_8b__seed44__single_qkvo_l18_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 106496,
    "total": 8190841856,
    "pct": 0.0013001838134866317,
    "eval_loss": 1.121,
    "train_runtime_s": 3090.0,
    "train_loss": 1.159
  }
]
```
