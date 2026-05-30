# Results: phase9b_20260520_212708

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_15b__combined_inlmh_r1 | affine_input_lm_head_plus_hidden_lora | 1 | 1.25M | 1.113 | 0.000 | 1157 |
| qwen25_15b__combined_inlmh_r2 | affine_input_lm_head_plus_hidden_lora | 2 | 2.41M | 1.095 | -0.018 | 1194 |
| qwen25_15b__hidden_lora_r1 | hidden_lora | 1 | 1.15M | 1.127 | 0.014 | 1155 |
| qwen25_15b__hidden_lora_r2 | hidden_lora | 2 | 2.31M | 1.105 | -0.008 | 1186 |
| qwen25_15b__vocab_lora_r1 | hidden_lora | 1 | 1.46M | 1.119 | 0.006 | 1191 |
| qwen25_15b__vocab_lora_r2 | hidden_lora | 2 | 2.62M | 1.099 | -0.014 | 1224 |

```json
[
  {
    "run_id": "qwen25_15b__combined_inlmh_r1",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 1253888,
    "total": 1544968192,
    "pct": 0.08115947023976013,
    "eval_loss": 1.113,
    "train_runtime_s": 1157.0,
    "train_loss": 1.181
  },
  {
    "run_id": "qwen25_15b__combined_inlmh_r2",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 2407936,
    "total": 1546122240,
    "pct": 0.1557403378402991,
    "eval_loss": 1.095,
    "train_runtime_s": 1194.0,
    "train_loss": 1.164
  },
  {
    "run_id": "qwen25_15b__hidden_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 1154048,
    "total": 1544868352,
    "pct": 0.07470202872017925,
    "eval_loss": 1.127,
    "train_runtime_s": 1155.0,
    "train_loss": 1.197
  },
  {
    "run_id": "qwen25_15b__hidden_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 2308096,
    "total": 1546022400,
    "pct": 0.14929253288956226,
    "eval_loss": 1.105,
    "train_runtime_s": 1186.0,
    "train_loss": 1.176
  },
  {
    "run_id": "qwen25_15b__vocab_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 1460992,
    "total": 1545175296,
    "pct": 0.09455186112424101,
    "eval_loss": 1.119,
    "train_runtime_s": 1191.0,
    "train_loss": 1.19
  },
  {
    "run_id": "qwen25_15b__vocab_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 2615040,
    "total": 1546329344,
    "pct": 0.16911274497549728,
    "eval_loss": 1.099,
    "train_runtime_s": 1224.0,
    "train_loss": 1.171
  }
]
```
