# Results: phase12_tied_shared_20260524_003753

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_05b__seed42__hidden_lora_r8 | hidden_lora | 8 | 4.40M | 1.301 | 0.000 | 609 |
| qwen25_05b__seed42__tied_affine_inlmh_s1_16 | affine_input_lm_head | 16 | 30k | 1.687 | 0.386 | 385 |
| qwen25_05b__seed42__tied_combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 4.43M | 1.296 | -0.005 | 616 |
| qwen25_05b__seed43__hidden_lora_r8 | hidden_lora | 8 | 4.40M | 1.303 | 0.002 | 607 |
| qwen25_05b__seed43__tied_affine_inlmh_s1_16 | affine_input_lm_head | 16 | 30k | 1.689 | 0.388 | 381 |
| qwen25_05b__seed43__tied_combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 4.43M | 1.296 | -0.005 | 611 |
| qwen25_05b__seed44__hidden_lora_r8 | hidden_lora | 8 | 4.40M | 1.301 | 0.000 | 609 |
| qwen25_05b__seed44__tied_affine_inlmh_s1_16 | affine_input_lm_head | 16 | 30k | 1.686 | 0.385 | 381 |
| qwen25_05b__seed44__tied_combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 4.43M | 1.294 | -0.007 | 615 |
| qwen25_15b__seed42__hidden_lora_r8 | hidden_lora | 8 | 9.23M | 1.054 | -0.247 | 1125 |
| qwen25_15b__seed42__tied_affine_inlmh_s1_16 | affine_input_lm_head | 16 | 51k | 1.355 | 0.054 | 700 |
| qwen25_15b__seed42__tied_combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 9.28M | 1.049 | -0.252 | 1134 |
| qwen25_15b__seed43__hidden_lora_r8 | hidden_lora | 8 | 9.23M | 1.054 | -0.247 | 1120 |
| qwen25_15b__seed43__tied_affine_inlmh_s1_16 | affine_input_lm_head | 16 | 51k | 1.359 | 0.058 | 696 |
| qwen25_15b__seed43__tied_combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 9.28M | 1.048 | -0.253 | 1129 |
| qwen25_15b__seed44__hidden_lora_r8 | hidden_lora | 8 | 9.23M | 1.055 | -0.246 | 1122 |
| qwen25_15b__seed44__tied_affine_inlmh_s1_16 | affine_input_lm_head | 16 | 51k | 1.352 | 0.051 | 696 |
| qwen25_15b__seed44__tied_combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 9.28M | 1.048 | -0.253 | 1128 |
| qwen3_06b__seed42__hidden_lora_r8 | hidden_lora | 8 | 5.05M | 1.100 | -0.201 | 898 |
| qwen3_06b__seed42__tied_affine_inlmh_s1_16 | affine_input_lm_head | 16 | 34k | 1.357 | 0.056 | 635 |
| qwen3_06b__seed42__tied_combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 5.08M | 1.097 | -0.204 | 902 |
| qwen3_06b__seed43__hidden_lora_r8 | hidden_lora | 8 | 5.05M | 1.101 | -0.200 | 894 |
| qwen3_06b__seed43__tied_affine_inlmh_s1_16 | affine_input_lm_head | 16 | 34k | 1.356 | 0.055 | 632 |
| qwen3_06b__seed43__tied_combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 5.08M | 1.096 | -0.205 | 899 |
| qwen3_06b__seed44__hidden_lora_r8 | hidden_lora | 8 | 5.05M | 1.101 | -0.200 | 898 |
| qwen3_06b__seed44__tied_affine_inlmh_s1_16 | affine_input_lm_head | 16 | 34k | 1.356 | 0.055 | 634 |
| qwen3_06b__seed44__tied_combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 5.08M | 1.097 | -0.204 | 902 |
| qwen3_17b__seed42__hidden_lora_r8 | hidden_lora | 8 | 8.72M | 0.738 | -0.563 | 1308 |
| qwen3_17b__seed42__tied_affine_inlmh_s1_16 | affine_input_lm_head | 16 | 68k | 0.936 | -0.365 | 893 |
| qwen3_17b__seed42__tied_combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 8.78M | 0.735 | -0.566 | 1322 |
| qwen3_17b__seed43__hidden_lora_r8 | hidden_lora | 8 | 8.72M | 0.738 | -0.563 | 1305 |
| qwen3_17b__seed43__tied_affine_inlmh_s1_16 | affine_input_lm_head | 16 | 68k | 0.932 | -0.369 | 892 |
| qwen3_17b__seed43__tied_combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 8.78M | 0.735 | -0.566 | 1315 |
| qwen3_17b__seed44__hidden_lora_r8 | hidden_lora | 8 | 8.72M | 0.738 | -0.563 | 1308 |
| qwen3_17b__seed44__tied_affine_inlmh_s1_16 | affine_input_lm_head | 16 | 68k | 0.938 | -0.363 | 889 |
| qwen3_17b__seed44__tied_combined_inlmh_s1_8 | affine_input_lm_head_plus_hidden_lora | 8 | 8.78M | 0.736 | -0.565 | 1317 |

```json
[
  {
    "run_id": "qwen25_05b__seed42__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 4399104,
    "total": 498431872,
    "pct": 0.8825888244963597,
    "eval_loss": 1.301,
    "train_runtime_s": 609.1,
    "train_loss": 1.39
  },
  {
    "run_id": "qwen25_05b__seed42__tied_affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 29568,
    "total": 494062336,
    "pct": 0.005984669918250964,
    "eval_loss": 1.687,
    "train_runtime_s": 385.3,
    "train_loss": 1.74
  },
  {
    "run_id": "qwen25_05b__seed42__tied_combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 4428672,
    "total": 498461440,
    "pct": 0.8884683236480639,
    "eval_loss": 1.296,
    "train_runtime_s": 615.5,
    "train_loss": 1.386
  },
  {
    "run_id": "qwen25_05b__seed43__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 4399104,
    "total": 498431872,
    "pct": 0.8825888244963597,
    "eval_loss": 1.303,
    "train_runtime_s": 607.0,
    "train_loss": 1.392
  },
  {
    "run_id": "qwen25_05b__seed43__tied_affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 43,
    "trainable": 29568,
    "total": 494062336,
    "pct": 0.005984669918250964,
    "eval_loss": 1.689,
    "train_runtime_s": 381.1,
    "train_loss": 1.741
  },
  {
    "run_id": "qwen25_05b__seed43__tied_combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 43,
    "trainable": 4428672,
    "total": 498461440,
    "pct": 0.8884683236480639,
    "eval_loss": 1.296,
    "train_runtime_s": 611.2,
    "train_loss": 1.386
  },
  {
    "run_id": "qwen25_05b__seed44__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 4399104,
    "total": 498431872,
    "pct": 0.8825888244963597,
    "eval_loss": 1.301,
    "train_runtime_s": 609.3,
    "train_loss": 1.392
  },
  {
    "run_id": "qwen25_05b__seed44__tied_affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 44,
    "trainable": 29568,
    "total": 494062336,
    "pct": 0.005984669918250964,
    "eval_loss": 1.686,
    "train_runtime_s": 381.1,
    "train_loss": 1.74
  },
  {
    "run_id": "qwen25_05b__seed44__tied_combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 44,
    "trainable": 4428672,
    "total": 498461440,
    "pct": 0.8884683236480639,
    "eval_loss": 1.294,
    "train_runtime_s": 614.8,
    "train_loss": 1.385
  },
  {
    "run_id": "qwen25_15b__seed42__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 9232384,
    "total": 1552946688,
    "pct": 0.5945074657965335,
    "eval_loss": 1.054,
    "train_runtime_s": 1125.0,
    "train_loss": 1.13
  },
  {
    "run_id": "qwen25_15b__seed42__tied_affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 50688,
    "total": 1543764992,
    "pct": 0.003283401311901235,
    "eval_loss": 1.355,
    "train_runtime_s": 699.6,
    "train_loss": 1.409
  },
  {
    "run_id": "qwen25_15b__seed42__tied_combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 9283072,
    "total": 1552997376,
    "pct": 0.5977519436581457,
    "eval_loss": 1.049,
    "train_runtime_s": 1134.0,
    "train_loss": 1.124
  },
  {
    "run_id": "qwen25_15b__seed43__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 9232384,
    "total": 1552946688,
    "pct": 0.5945074657965335,
    "eval_loss": 1.054,
    "train_runtime_s": 1120.0,
    "train_loss": 1.132
  },
  {
    "run_id": "qwen25_15b__seed43__tied_affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 43,
    "trainable": 50688,
    "total": 1543764992,
    "pct": 0.003283401311901235,
    "eval_loss": 1.359,
    "train_runtime_s": 696.4,
    "train_loss": 1.41
  },
  {
    "run_id": "qwen25_15b__seed43__tied_combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 43,
    "trainable": 9283072,
    "total": 1552997376,
    "pct": 0.5977519436581457,
    "eval_loss": 1.048,
    "train_runtime_s": 1129.0,
    "train_loss": 1.125
  },
  {
    "run_id": "qwen25_15b__seed44__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 9232384,
    "total": 1552946688,
    "pct": 0.5945074657965335,
    "eval_loss": 1.055,
    "train_runtime_s": 1122.0,
    "train_loss": 1.131
  },
  {
    "run_id": "qwen25_15b__seed44__tied_affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 44,
    "trainable": 50688,
    "total": 1543764992,
    "pct": 0.003283401311901235,
    "eval_loss": 1.352,
    "train_runtime_s": 696.0,
    "train_loss": 1.405
  },
  {
    "run_id": "qwen25_15b__seed44__tied_combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 44,
    "trainable": 9283072,
    "total": 1552997376,
    "pct": 0.5977519436581457,
    "eval_loss": 1.048,
    "train_runtime_s": 1128.0,
    "train_loss": 1.124
  },
  {
    "run_id": "qwen3_06b__seed42__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 5046272,
    "total": 601096192,
    "pct": 0.8395115569123418,
    "eval_loss": 1.1,
    "train_runtime_s": 898.2,
    "train_loss": 1.165
  },
  {
    "run_id": "qwen3_06b__seed42__tied_affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 33792,
    "total": 596083712,
    "pct": 0.005669002410184964,
    "eval_loss": 1.357,
    "train_runtime_s": 634.9,
    "train_loss": 1.394
  },
  {
    "run_id": "qwen3_06b__seed42__tied_combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 5080064,
    "total": 601129984,
    "pct": 0.8450857776543718,
    "eval_loss": 1.097,
    "train_runtime_s": 901.8,
    "train_loss": 1.16
  },
  {
    "run_id": "qwen3_06b__seed43__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 5046272,
    "total": 601096192,
    "pct": 0.8395115569123418,
    "eval_loss": 1.101,
    "train_runtime_s": 894.2,
    "train_loss": 1.166
  },
  {
    "run_id": "qwen3_06b__seed43__tied_affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 43,
    "trainable": 33792,
    "total": 596083712,
    "pct": 0.005669002410184964,
    "eval_loss": 1.356,
    "train_runtime_s": 631.8,
    "train_loss": 1.395
  },
  {
    "run_id": "qwen3_06b__seed43__tied_combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 43,
    "trainable": 5080064,
    "total": 601129984,
    "pct": 0.8450857776543718,
    "eval_loss": 1.096,
    "train_runtime_s": 898.6,
    "train_loss": 1.161
  },
  {
    "run_id": "qwen3_06b__seed44__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 5046272,
    "total": 601096192,
    "pct": 0.8395115569123418,
    "eval_loss": 1.101,
    "train_runtime_s": 898.3,
    "train_loss": 1.165
  },
  {
    "run_id": "qwen3_06b__seed44__tied_affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 44,
    "trainable": 33792,
    "total": 596083712,
    "pct": 0.005669002410184964,
    "eval_loss": 1.356,
    "train_runtime_s": 633.5,
    "train_loss": 1.393
  },
  {
    "run_id": "qwen3_06b__seed44__tied_combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 44,
    "trainable": 5080064,
    "total": 601129984,
    "pct": 0.8450857776543718,
    "eval_loss": 1.097,
    "train_runtime_s": 901.5,
    "train_loss": 1.161
  },
  {
    "run_id": "qwen3_17b__seed42__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 8716288,
    "total": 1729291264,
    "pct": 0.5040381676270355,
    "eval_loss": 0.7376,
    "train_runtime_s": 1308.0,
    "train_loss": 0.796
  },
  {
    "run_id": "qwen3_17b__seed42__tied_affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 67584,
    "total": 1720642560,
    "pct": 0.003927834959516519,
    "eval_loss": 0.9363,
    "train_runtime_s": 893.2,
    "train_loss": 0.9855
  },
  {
    "run_id": "qwen3_17b__seed42__tied_combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 8783872,
    "total": 1729358848,
    "pct": 0.5079265075700471,
    "eval_loss": 0.7346,
    "train_runtime_s": 1322.0,
    "train_loss": 0.7927
  },
  {
    "run_id": "qwen3_17b__seed43__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 8716288,
    "total": 1729291264,
    "pct": 0.5040381676270355,
    "eval_loss": 0.7381,
    "train_runtime_s": 1305.0,
    "train_loss": 0.7971
  },
  {
    "run_id": "qwen3_17b__seed43__tied_affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 43,
    "trainable": 67584,
    "total": 1720642560,
    "pct": 0.003927834959516519,
    "eval_loss": 0.9316,
    "train_runtime_s": 892.2,
    "train_loss": 0.9839
  },
  {
    "run_id": "qwen3_17b__seed43__tied_combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 43,
    "trainable": 8783872,
    "total": 1729358848,
    "pct": 0.5079265075700471,
    "eval_loss": 0.735,
    "train_runtime_s": 1315.0,
    "train_loss": 0.7935
  },
  {
    "run_id": "qwen3_17b__seed44__hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 8716288,
    "total": 1729291264,
    "pct": 0.5040381676270355,
    "eval_loss": 0.7377,
    "train_runtime_s": 1308.0,
    "train_loss": 0.7963
  },
  {
    "run_id": "qwen3_17b__seed44__tied_affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 44,
    "trainable": 67584,
    "total": 1720642560,
    "pct": 0.003927834959516519,
    "eval_loss": 0.9375,
    "train_runtime_s": 888.8,
    "train_loss": 0.9874
  },
  {
    "run_id": "qwen3_17b__seed44__tied_combined_inlmh_s1_8",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 128.0,
    "seed": 44,
    "trainable": 8783872,
    "total": 1729358848,
    "pct": 0.5079265075700471,
    "eval_loss": 0.7356,
    "train_runtime_s": 1317.0,
    "train_loss": 0.7934
  }
]
```
