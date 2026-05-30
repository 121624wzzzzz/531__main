# Results: phase7b_20260520_045817

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| combined_inlmh_r1 | affine_input_lm_head_plus_hidden_lora | 1 | 2.99M | 0.900 | 0.000 | 4856 |
| combined_inlmh_r2 | affine_input_lm_head_plus_hidden_lora | 2 | 5.72M | 0.891 | -0.009 | 4968 |
| combined_inlmh_r4 | affine_input_lm_head_plus_hidden_lora | 4 | 11.18M | 0.881 | -0.019 | 4929 |
| hidden_lora_r1 | hidden_lora | 1 | 2.73M | 0.907 | 0.007 | 4850 |
| hidden_lora_r2 | hidden_lora | 2 | 5.46M | 0.897 | -0.003 | 4910 |
| hidden_lora_r4 | hidden_lora | 4 | 10.91M | 0.885 | -0.015 | 4930 |

```json
[
  {
    "run_id": "combined_inlmh_r1",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 2994176,
    "total": 8193729536,
    "pct": 0.036542285010077245,
    "eval_loss": 0.9,
    "train_runtime_s": 4856.0,
    "train_loss": 0.9488
  },
  {
    "run_id": "combined_inlmh_r2",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 5722112,
    "total": 8196457472,
    "pct": 0.06981201353813356,
    "eval_loss": 0.8915,
    "train_runtime_s": 4968.0,
    "train_loss": 0.9402
  },
  {
    "run_id": "combined_inlmh_r4",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 11177984,
    "total": 8201913344,
    "pct": 0.1362850780199611,
    "eval_loss": 0.8808,
    "train_runtime_s": 4929.0,
    "train_loss": 0.9308
  },
  {
    "run_id": "hidden_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 2727936,
    "total": 8193463296,
    "pct": 0.0332940528498097,
    "eval_loss": 0.9071,
    "train_runtime_s": 4850.0,
    "train_loss": 0.9559
  },
  {
    "run_id": "hidden_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 5455872,
    "total": 8196191232,
    "pct": 0.06656594319931065,
    "eval_loss": 0.8971,
    "train_runtime_s": 4910.0,
    "train_loss": 0.9455
  },
  {
    "run_id": "hidden_lora_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 10911744,
    "total": 8201647104,
    "pct": 0.13304332485456816,
    "eval_loss": 0.8848,
    "train_runtime_s": 4930.0,
    "train_loss": 0.9346
  }
]
```
