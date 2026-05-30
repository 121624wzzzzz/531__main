# Results: position_affine_sweep_small_20260526_094509

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_05b__seed42__input_r16_s1_16 | - | - | - | 1.879 | 0.000 | - |
| qwen25_05b__seed42__input_r16_s1_32 | - | - | - | 1.871 | -0.008 | - |
| qwen25_05b__seed42__input_r8_s1_128 | affine_input | 16 | 15k | 1.814 | -0.065 | 384 |
| qwen25_05b__seed42__input_r8_s1_16 | affine_input | 16 | 15k | 1.837 | -0.042 | 384 |
| qwen25_05b__seed42__input_r8_s1_2 | affine_input | 16 | 15k | 1.880 | 0.001 | 383 |
| qwen25_05b__seed42__input_r8_s1_32 | affine_input | 16 | 15k | 1.828 | -0.051 | 381 |
| qwen25_05b__seed42__input_r8_s1_4 | affine_input | 16 | 15k | 1.866 | -0.013 | 384 |
| qwen25_05b__seed42__input_r8_s1_64 | affine_input | 16 | 15k | 1.816 | -0.063 | 384 |
| qwen25_05b__seed42__input_r8_s1_8 | affine_input | 16 | 15k | 1.845 | -0.034 | 382 |
| qwen25_05b__seed42__lm_head_r16_s1_16 | - | - | - | 1.888 | 0.009 | - |
| qwen25_05b__seed42__lm_head_r16_s1_32 | - | - | - | 1.916 | 0.037 | - |
| qwen25_05b__seed42__lm_head_r8_s1_128 | affine_lm_head | 16 | 14k | 1.907 | 0.028 | 242 |
| qwen25_05b__seed42__lm_head_r8_s1_16 | affine_lm_head | 16 | 14k | 1.915 | 0.036 | 242 |
| qwen25_05b__seed42__lm_head_r8_s1_2 | affine_lm_head | 16 | 14k | 1.945 | 0.066 | 242 |
| qwen25_05b__seed42__lm_head_r8_s1_32 | affine_lm_head | 16 | 14k | 1.909 | 0.030 | 241 |
| qwen25_05b__seed42__lm_head_r8_s1_4 | affine_lm_head | 16 | 14k | 1.930 | 0.051 | 241 |
| qwen25_05b__seed42__lm_head_r8_s1_64 | affine_lm_head | 16 | 14k | 1.907 | 0.028 | 241 |
| qwen25_05b__seed42__lm_head_r8_s1_8 | affine_lm_head | 16 | 14k | 1.922 | 0.043 | 241 |
| qwen3_06b__seed42__input_r8_s1_128 | affine_input | 16 | 17k | 1.443 | -0.436 | 630 |
| qwen3_06b__seed42__input_r8_s1_16 | affine_input | 16 | 17k | 1.458 | -0.421 | 632 |
| qwen3_06b__seed42__input_r8_s1_2 | affine_input | 16 | 17k | 1.485 | -0.394 | 633 |
| qwen3_06b__seed42__input_r8_s1_32 | affine_input | 16 | 17k | 1.448 | -0.431 | 631 |
| qwen3_06b__seed42__input_r8_s1_4 | affine_input | 16 | 17k | 1.480 | -0.399 | 633 |
| qwen3_06b__seed42__input_r8_s1_64 | affine_input | 16 | 17k | 1.441 | -0.438 | 630 |
| qwen3_06b__seed42__input_r8_s1_8 | affine_input | 16 | 17k | 1.465 | -0.414 | 632 |
| qwen3_06b__seed42__lm_head_r8_s1_128 | affine_lm_head | 16 | 16k | 1.613 | -0.266 | 337 |
| qwen3_06b__seed42__lm_head_r8_s1_16 | affine_lm_head | 16 | 16k | 1.618 | -0.261 | 339 |
| qwen3_06b__seed42__lm_head_r8_s1_2 | affine_lm_head | 16 | 16k | 1.640 | -0.239 | 339 |
| qwen3_06b__seed42__lm_head_r8_s1_32 | affine_lm_head | 16 | 16k | 1.612 | -0.267 | 338 |
| qwen3_06b__seed42__lm_head_r8_s1_4 | affine_lm_head | 16 | 16k | 1.630 | -0.249 | 337 |
| qwen3_06b__seed42__lm_head_r8_s1_64 | affine_lm_head | 16 | 16k | 1.610 | -0.269 | 337 |
| qwen3_06b__seed42__lm_head_r8_s1_8 | affine_lm_head | 16 | 16k | 1.621 | -0.258 | 338 |

```json
[
  {
    "run_id": "qwen25_05b__seed42__input_r16_s1_16",
    "variant": null,
    "hidden_lora_rank": null,
    "affine_alpha": null,
    "seed": null,
    "trainable": null,
    "total": null,
    "pct": null,
    "eval_loss": 1.879,
    "train_runtime_s": null,
    "train_loss": null
  },
  {
    "run_id": "qwen25_05b__seed42__input_r16_s1_32",
    "variant": null,
    "hidden_lora_rank": null,
    "affine_alpha": null,
    "seed": null,
    "trainable": null,
    "total": null,
    "pct": null,
    "eval_loss": 1.871,
    "train_runtime_s": null,
    "train_loss": null
  },
  {
    "run_id": "qwen25_05b__seed42__input_r8_s1_128",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 15232,
    "total": 494048000,
    "pct": 0.0030831012371267566,
    "eval_loss": 1.814,
    "train_runtime_s": 384.5,
    "train_loss": 1.864
  },
  {
    "run_id": "qwen25_05b__seed42__input_r8_s1_16",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 15232,
    "total": 494048000,
    "pct": 0.0030831012371267566,
    "eval_loss": 1.837,
    "train_runtime_s": 383.5,
    "train_loss": 1.887
  },
  {
    "run_id": "qwen25_05b__seed42__input_r8_s1_2",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 16.0,
    "seed": 42,
    "trainable": 15232,
    "total": 494048000,
    "pct": 0.0030831012371267566,
    "eval_loss": 1.88,
    "train_runtime_s": 382.8,
    "train_loss": 1.929
  },
  {
    "run_id": "qwen25_05b__seed42__input_r8_s1_32",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 15232,
    "total": 494048000,
    "pct": 0.0030831012371267566,
    "eval_loss": 1.828,
    "train_runtime_s": 381.3,
    "train_loss": 1.877
  },
  {
    "run_id": "qwen25_05b__seed42__input_r8_s1_4",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 15232,
    "total": 494048000,
    "pct": 0.0030831012371267566,
    "eval_loss": 1.866,
    "train_runtime_s": 383.9,
    "train_loss": 1.916
  },
  {
    "run_id": "qwen25_05b__seed42__input_r8_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 42,
    "trainable": 15232,
    "total": 494048000,
    "pct": 0.0030831012371267566,
    "eval_loss": 1.816,
    "train_runtime_s": 383.6,
    "train_loss": 1.865
  },
  {
    "run_id": "qwen25_05b__seed42__input_r8_s1_8",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 64.0,
    "seed": 42,
    "trainable": 15232,
    "total": 494048000,
    "pct": 0.0030831012371267566,
    "eval_loss": 1.845,
    "train_runtime_s": 381.9,
    "train_loss": 1.898
  },
  {
    "run_id": "qwen25_05b__seed42__lm_head_r16_s1_16",
    "variant": null,
    "hidden_lora_rank": null,
    "affine_alpha": null,
    "seed": null,
    "trainable": null,
    "total": null,
    "pct": null,
    "eval_loss": 1.888,
    "train_runtime_s": null,
    "train_loss": null
  },
  {
    "run_id": "qwen25_05b__seed42__lm_head_r16_s1_32",
    "variant": null,
    "hidden_lora_rank": null,
    "affine_alpha": null,
    "seed": null,
    "trainable": null,
    "total": null,
    "pct": null,
    "eval_loss": 1.916,
    "train_runtime_s": null,
    "train_loss": null
  },
  {
    "run_id": "qwen25_05b__seed42__lm_head_r8_s1_128",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 14336,
    "total": 494047104,
    "pct": 0.0029017476034026097,
    "eval_loss": 1.907,
    "train_runtime_s": 242.4,
    "train_loss": 1.948
  },
  {
    "run_id": "qwen25_05b__seed42__lm_head_r8_s1_16",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 14336,
    "total": 494047104,
    "pct": 0.0029017476034026097,
    "eval_loss": 1.915,
    "train_runtime_s": 241.9,
    "train_loss": 1.953
  },
  {
    "run_id": "qwen25_05b__seed42__lm_head_r8_s1_2",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 16.0,
    "seed": 42,
    "trainable": 14336,
    "total": 494047104,
    "pct": 0.0029017476034026097,
    "eval_loss": 1.945,
    "train_runtime_s": 242.0,
    "train_loss": 1.989
  },
  {
    "run_id": "qwen25_05b__seed42__lm_head_r8_s1_32",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 14336,
    "total": 494047104,
    "pct": 0.0029017476034026097,
    "eval_loss": 1.909,
    "train_runtime_s": 241.2,
    "train_loss": 1.945
  },
  {
    "run_id": "qwen25_05b__seed42__lm_head_r8_s1_4",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 14336,
    "total": 494047104,
    "pct": 0.0029017476034026097,
    "eval_loss": 1.93,
    "train_runtime_s": 241.1,
    "train_loss": 1.972
  },
  {
    "run_id": "qwen25_05b__seed42__lm_head_r8_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 42,
    "trainable": 14336,
    "total": 494047104,
    "pct": 0.0029017476034026097,
    "eval_loss": 1.907,
    "train_runtime_s": 240.9,
    "train_loss": 1.944
  },
  {
    "run_id": "qwen25_05b__seed42__lm_head_r8_s1_8",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 64.0,
    "seed": 42,
    "trainable": 14336,
    "total": 494047104,
    "pct": 0.0029017476034026097,
    "eval_loss": 1.922,
    "train_runtime_s": 240.7,
    "train_loss": 1.961
  },
  {
    "run_id": "qwen3_06b__seed42__input_r8_s1_128",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 17408,
    "total": 596067328,
    "pct": 0.002920475453403814,
    "eval_loss": 1.443,
    "train_runtime_s": 630.3,
    "train_loss": 1.479
  },
  {
    "run_id": "qwen3_06b__seed42__input_r8_s1_16",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 17408,
    "total": 596067328,
    "pct": 0.002920475453403814,
    "eval_loss": 1.458,
    "train_runtime_s": 632.4,
    "train_loss": 1.493
  },
  {
    "run_id": "qwen3_06b__seed42__input_r8_s1_2",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 16.0,
    "seed": 42,
    "trainable": 17408,
    "total": 596067328,
    "pct": 0.002920475453403814,
    "eval_loss": 1.485,
    "train_runtime_s": 632.9,
    "train_loss": 1.521
  },
  {
    "run_id": "qwen3_06b__seed42__input_r8_s1_32",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 17408,
    "total": 596067328,
    "pct": 0.002920475453403814,
    "eval_loss": 1.448,
    "train_runtime_s": 630.8,
    "train_loss": 1.482
  },
  {
    "run_id": "qwen3_06b__seed42__input_r8_s1_4",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 17408,
    "total": 596067328,
    "pct": 0.002920475453403814,
    "eval_loss": 1.48,
    "train_runtime_s": 632.7,
    "train_loss": 1.515
  },
  {
    "run_id": "qwen3_06b__seed42__input_r8_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 42,
    "trainable": 17408,
    "total": 596067328,
    "pct": 0.002920475453403814,
    "eval_loss": 1.441,
    "train_runtime_s": 630.1,
    "train_loss": 1.477
  },
  {
    "run_id": "qwen3_06b__seed42__input_r8_s1_8",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 64.0,
    "seed": 42,
    "trainable": 17408,
    "total": 596067328,
    "pct": 0.002920475453403814,
    "eval_loss": 1.465,
    "train_runtime_s": 631.8,
    "train_loss": 1.5
  },
  {
    "run_id": "qwen3_06b__seed42__lm_head_r8_s1_128",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 16384,
    "total": 596066304,
    "pct": 0.0027486875017179294,
    "eval_loss": 1.613,
    "train_runtime_s": 337.1,
    "train_loss": 1.643
  },
  {
    "run_id": "qwen3_06b__seed42__lm_head_r8_s1_16",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 16384,
    "total": 596066304,
    "pct": 0.0027486875017179294,
    "eval_loss": 1.618,
    "train_runtime_s": 338.8,
    "train_loss": 1.644
  },
  {
    "run_id": "qwen3_06b__seed42__lm_head_r8_s1_2",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 16.0,
    "seed": 42,
    "trainable": 16384,
    "total": 596066304,
    "pct": 0.0027486875017179294,
    "eval_loss": 1.64,
    "train_runtime_s": 339.3,
    "train_loss": 1.671
  },
  {
    "run_id": "qwen3_06b__seed42__lm_head_r8_s1_32",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 16384,
    "total": 596066304,
    "pct": 0.0027486875017179294,
    "eval_loss": 1.612,
    "train_runtime_s": 337.7,
    "train_loss": 1.638
  },
  {
    "run_id": "qwen3_06b__seed42__lm_head_r8_s1_4",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 16384,
    "total": 596066304,
    "pct": 0.0027486875017179294,
    "eval_loss": 1.63,
    "train_runtime_s": 337.2,
    "train_loss": 1.66
  },
  {
    "run_id": "qwen3_06b__seed42__lm_head_r8_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 42,
    "trainable": 16384,
    "total": 596066304,
    "pct": 0.0027486875017179294,
    "eval_loss": 1.61,
    "train_runtime_s": 337.1,
    "train_loss": 1.636
  },
  {
    "run_id": "qwen3_06b__seed42__lm_head_r8_s1_8",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 64.0,
    "seed": 42,
    "trainable": 16384,
    "total": 596066304,
    "pct": 0.0027486875017179294,
    "eval_loss": 1.621,
    "train_runtime_s": 338.0,
    "train_loss": 1.649
  }
]
```
