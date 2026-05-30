# Results: phase10_single_layer_20260523_102449

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_15b__seed42__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 100k | 1.322 | 0.000 | 698 |
| qwen25_15b__seed42__affine_input_s1_64 | affine_input | 16 | 51k | 1.471 | 0.149 | 698 |
| qwen25_15b__seed42__single_q_l0_r16 | hidden_lora | 16 | 49k | 1.771 | 0.449 | 697 |
| qwen25_15b__seed42__single_q_l14_r16 | hidden_lora | 16 | 49k | 1.720 | 0.398 | 551 |
| qwen25_15b__seed42__single_q_l27_r16 | hidden_lora | 16 | 49k | 1.850 | 0.528 | 409 |
| qwen25_15b__seed42__single_qkvo_l14_r4 | hidden_lora | 4 | 39k | 1.486 | 0.164 | 552 |
| qwen25_15b__seed43__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 100k | 1.322 | 0.000 | 696 |
| qwen25_15b__seed43__affine_input_s1_64 | affine_input | 16 | 51k | 1.471 | 0.149 | 696 |
| qwen25_15b__seed43__single_q_l0_r16 | hidden_lora | 16 | 49k | 1.777 | 0.455 | 695 |
| qwen25_15b__seed43__single_q_l14_r16 | hidden_lora | 16 | 49k | 1.720 | 0.398 | 549 |
| qwen25_15b__seed43__single_q_l27_r16 | hidden_lora | 16 | 49k | 1.850 | 0.528 | 409 |
| qwen25_15b__seed43__single_qkvo_l14_r4 | hidden_lora | 4 | 39k | 1.477 | 0.155 | 551 |
| qwen25_15b__seed44__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 100k | 1.324 | 0.002 | 698 |
| qwen25_15b__seed44__affine_input_s1_64 | affine_input | 16 | 51k | 1.474 | 0.152 | 698 |
| qwen25_15b__seed44__single_q_l0_r16 | hidden_lora | 16 | 49k | 1.776 | 0.454 | 696 |
| qwen25_15b__seed44__single_q_l14_r16 | hidden_lora | 16 | 49k | 1.718 | 0.396 | 549 |
| qwen25_15b__seed44__single_q_l27_r16 | hidden_lora | 16 | 49k | 1.850 | 0.528 | 410 |
| qwen25_15b__seed44__single_qkvo_l14_r4 | hidden_lora | 4 | 39k | 1.487 | 0.165 | 552 |
| qwen3_17b__seed42__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 133k | 0.909 | -0.413 | 890 |
| qwen3_17b__seed42__affine_input_s1_64 | affine_input | 16 | 68k | 0.980 | -0.342 | 890 |
| qwen3_17b__seed42__single_q_l0_r16 | hidden_lora | 16 | 66k | 1.270 | -0.052 | 886 |
| qwen3_17b__seed42__single_q_l14_r16 | hidden_lora | 16 | 66k | 1.269 | -0.053 | 686 |
| qwen3_17b__seed42__single_q_l27_r16 | hidden_lora | 16 | 66k | 1.395 | 0.073 | 493 |
| qwen3_17b__seed42__single_qkvo_l14_r4 | hidden_lora | 4 | 57k | 1.053 | -0.269 | 689 |
| qwen3_17b__seed43__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 133k | 0.912 | -0.410 | 888 |
| qwen3_17b__seed43__affine_input_s1_64 | affine_input | 16 | 68k | 0.979 | -0.343 | 888 |
| qwen3_17b__seed43__single_q_l0_r16 | hidden_lora | 16 | 66k | 1.268 | -0.054 | 884 |
| qwen3_17b__seed43__single_q_l14_r16 | hidden_lora | 16 | 66k | 1.270 | -0.052 | 682 |
| qwen3_17b__seed43__single_q_l27_r16 | hidden_lora | 16 | 66k | 1.395 | 0.073 | 491 |
| qwen3_17b__seed43__single_qkvo_l14_r4 | hidden_lora | 4 | 57k | 1.060 | -0.262 | 688 |
| qwen3_17b__seed44__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 133k | 0.910 | -0.412 | 890 |
| qwen3_17b__seed44__affine_input_s1_64 | affine_input | 16 | 68k | 0.977 | -0.345 | 888 |
| qwen3_17b__seed44__single_q_l0_r16 | hidden_lora | 16 | 66k | 1.277 | -0.045 | 884 |
| qwen3_17b__seed44__single_q_l14_r16 | hidden_lora | 16 | 66k | 1.271 | -0.051 | 684 |
| qwen3_17b__seed44__single_q_l27_r16 | hidden_lora | 16 | 66k | 1.396 | 0.074 | 492 |
| qwen3_17b__seed44__single_qkvo_l14_r4 | hidden_lora | 4 | 57k | 1.055 | -0.267 | 689 |

```json
[
  {
    "run_id": "qwen25_15b__seed42__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 99840,
    "total": 1543814144,
    "pct": 0.0064670997080850685,
    "eval_loss": 1.322,
    "train_runtime_s": 698.5,
    "train_loss": 1.377
  },
  {
    "run_id": "qwen25_15b__seed42__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 50688,
    "total": 1543764992,
    "pct": 0.003283401311901235,
    "eval_loss": 1.471,
    "train_runtime_s": 697.9,
    "train_loss": 1.524
  },
  {
    "run_id": "qwen25_15b__seed42__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.771,
    "train_runtime_s": 696.9,
    "train_loss": 1.803
  },
  {
    "run_id": "qwen25_15b__seed42__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.72,
    "train_runtime_s": 550.7,
    "train_loss": 1.764
  },
  {
    "run_id": "qwen25_15b__seed42__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.85,
    "train_runtime_s": 409.2,
    "train_loss": 1.88
  },
  {
    "run_id": "qwen25_15b__seed42__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 38912,
    "total": 1543753216,
    "pct": 0.0025206101335823873,
    "eval_loss": 1.486,
    "train_runtime_s": 552.2,
    "train_loss": 1.548
  },
  {
    "run_id": "qwen25_15b__seed43__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 43,
    "trainable": 99840,
    "total": 1543814144,
    "pct": 0.0064670997080850685,
    "eval_loss": 1.322,
    "train_runtime_s": 696.1,
    "train_loss": 1.379
  },
  {
    "run_id": "qwen25_15b__seed43__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 43,
    "trainable": 50688,
    "total": 1543764992,
    "pct": 0.003283401311901235,
    "eval_loss": 1.471,
    "train_runtime_s": 695.9,
    "train_loss": 1.523
  },
  {
    "run_id": "qwen25_15b__seed43__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.777,
    "train_runtime_s": 695.1,
    "train_loss": 1.811
  },
  {
    "run_id": "qwen25_15b__seed43__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.72,
    "train_runtime_s": 548.7,
    "train_loss": 1.764
  },
  {
    "run_id": "qwen25_15b__seed43__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.85,
    "train_runtime_s": 408.6,
    "train_loss": 1.88
  },
  {
    "run_id": "qwen25_15b__seed43__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 38912,
    "total": 1543753216,
    "pct": 0.0025206101335823873,
    "eval_loss": 1.477,
    "train_runtime_s": 551.2,
    "train_loss": 1.541
  },
  {
    "run_id": "qwen25_15b__seed44__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 44,
    "trainable": 99840,
    "total": 1543814144,
    "pct": 0.0064670997080850685,
    "eval_loss": 1.324,
    "train_runtime_s": 697.6,
    "train_loss": 1.379
  },
  {
    "run_id": "qwen25_15b__seed44__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 44,
    "trainable": 50688,
    "total": 1543764992,
    "pct": 0.003283401311901235,
    "eval_loss": 1.474,
    "train_runtime_s": 698.2,
    "train_loss": 1.526
  },
  {
    "run_id": "qwen25_15b__seed44__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.776,
    "train_runtime_s": 696.2,
    "train_loss": 1.812
  },
  {
    "run_id": "qwen25_15b__seed44__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.718,
    "train_runtime_s": 549.3,
    "train_loss": 1.761
  },
  {
    "run_id": "qwen25_15b__seed44__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.85,
    "train_runtime_s": 410.1,
    "train_loss": 1.879
  },
  {
    "run_id": "qwen25_15b__seed44__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 38912,
    "total": 1543753216,
    "pct": 0.0025206101335823873,
    "eval_loss": 1.487,
    "train_runtime_s": 551.8,
    "train_loss": 1.545
  },
  {
    "run_id": "qwen3_17b__seed42__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 133120,
    "total": 1720708096,
    "pct": 0.007736349954385291,
    "eval_loss": 0.9091,
    "train_runtime_s": 890.3,
    "train_loss": 0.9602
  },
  {
    "run_id": "qwen3_17b__seed42__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 67584,
    "total": 1720642560,
    "pct": 0.003927834959516519,
    "eval_loss": 0.9804,
    "train_runtime_s": 889.7,
    "train_loss": 1.03
  },
  {
    "run_id": "qwen3_17b__seed42__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.27,
    "train_runtime_s": 885.6,
    "train_loss": 1.309
  },
  {
    "run_id": "qwen3_17b__seed42__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.269,
    "train_runtime_s": 685.7,
    "train_loss": 1.308
  },
  {
    "run_id": "qwen3_17b__seed42__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.395,
    "train_runtime_s": 493.4,
    "train_loss": 1.419
  },
  {
    "run_id": "qwen3_17b__seed42__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 57344,
    "total": 1720632320,
    "pct": 0.0033327282844483586,
    "eval_loss": 1.053,
    "train_runtime_s": 689.2,
    "train_loss": 1.104
  },
  {
    "run_id": "qwen3_17b__seed43__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 43,
    "trainable": 133120,
    "total": 1720708096,
    "pct": 0.007736349954385291,
    "eval_loss": 0.9123,
    "train_runtime_s": 888.1,
    "train_loss": 0.9613
  },
  {
    "run_id": "qwen3_17b__seed43__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 43,
    "trainable": 67584,
    "total": 1720642560,
    "pct": 0.003927834959516519,
    "eval_loss": 0.9787,
    "train_runtime_s": 887.6,
    "train_loss": 1.029
  },
  {
    "run_id": "qwen3_17b__seed43__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.268,
    "train_runtime_s": 883.6,
    "train_loss": 1.309
  },
  {
    "run_id": "qwen3_17b__seed43__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.27,
    "train_runtime_s": 682.5,
    "train_loss": 1.31
  },
  {
    "run_id": "qwen3_17b__seed43__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.395,
    "train_runtime_s": 491.1,
    "train_loss": 1.42
  },
  {
    "run_id": "qwen3_17b__seed43__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 57344,
    "total": 1720632320,
    "pct": 0.0033327282844483586,
    "eval_loss": 1.06,
    "train_runtime_s": 687.9,
    "train_loss": 1.111
  },
  {
    "run_id": "qwen3_17b__seed44__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 44,
    "trainable": 133120,
    "total": 1720708096,
    "pct": 0.007736349954385291,
    "eval_loss": 0.9105,
    "train_runtime_s": 889.8,
    "train_loss": 0.961
  },
  {
    "run_id": "qwen3_17b__seed44__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 44,
    "trainable": 67584,
    "total": 1720642560,
    "pct": 0.003927834959516519,
    "eval_loss": 0.9771,
    "train_runtime_s": 888.1,
    "train_loss": 1.028
  },
  {
    "run_id": "qwen3_17b__seed44__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.277,
    "train_runtime_s": 884.4,
    "train_loss": 1.313
  },
  {
    "run_id": "qwen3_17b__seed44__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.271,
    "train_runtime_s": 684.2,
    "train_loss": 1.311
  },
  {
    "run_id": "qwen3_17b__seed44__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.396,
    "train_runtime_s": 491.7,
    "train_loss": 1.419
  },
  {
    "run_id": "qwen3_17b__seed44__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 57344,
    "total": 1720632320,
    "pct": 0.0033327282844483586,
    "eval_loss": 1.055,
    "train_runtime_s": 688.7,
    "train_loss": 1.107
  }
]
```
