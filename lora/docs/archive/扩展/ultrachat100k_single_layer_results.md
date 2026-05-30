# Results: ultrachat100k_single_layer_20260524_205104

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_15b__seed42__single_q_l0_r16 | hidden_lora | 16 | 49k | 1.135 | 0.000 | 6176 |
| qwen25_15b__seed42__single_q_l14_r16 | hidden_lora | 16 | 49k | 1.121 | -0.014 | 4752 |
| qwen25_15b__seed42__single_q_l27_r16 | hidden_lora | 16 | 49k | 1.168 | 0.033 | 3471 |
| qwen25_15b__seed42__single_qkvo_l14_r4 | hidden_lora | 4 | 39k | 1.102 | -0.033 | 4799 |
| qwen3_17b__seed42__single_q_l0_r16 | hidden_lora | 16 | 66k | 1.108 | -0.027 | 7928 |
| qwen3_17b__seed42__single_q_l14_r16 | hidden_lora | 16 | 66k | 1.105 | -0.030 | 5984 |
| qwen3_17b__seed42__single_q_l27_r16 | hidden_lora | 16 | 66k | 1.128 | -0.007 | 4196 |
| qwen3_17b__seed42__single_qkvo_l14_r4 | hidden_lora | 4 | 57k | 1.088 | -0.047 | 6026 |

```json
[
  {
    "run_id": "qwen25_15b__seed42__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.135,
    "train_runtime_s": 6176.0,
    "train_loss": 1.128
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
    "eval_loss": 1.121,
    "train_runtime_s": 4752.0,
    "train_loss": 1.115
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
    "eval_loss": 1.168,
    "train_runtime_s": 3471.0,
    "train_loss": 1.161
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
    "eval_loss": 1.102,
    "train_runtime_s": 4799.0,
    "train_loss": 1.096
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
    "eval_loss": 1.108,
    "train_runtime_s": 7928.0,
    "train_loss": 1.101
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
    "eval_loss": 1.105,
    "train_runtime_s": 5984.0,
    "train_loss": 1.098
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
    "eval_loss": 1.128,
    "train_runtime_s": 4196.0,
    "train_loss": 1.119
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
    "eval_loss": 1.088,
    "train_runtime_s": 6026.0,
    "train_loss": 1.081
  }
]
```
