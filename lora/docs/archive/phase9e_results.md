# Results: phase9e_20260521_015553

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_15b__seed43__combined_inlmh_r1 | affine_input_lm_head_plus_hidden_lora | 1 | 1.25M | 1.113 | 0.000 | 1181 |
| qwen25_15b__seed43__combined_inlmh_r2 | affine_input_lm_head_plus_hidden_lora | 2 | 2.41M | 1.095 | -0.018 | 1217 |
| qwen25_15b__seed43__hidden_lora_r1 | hidden_lora | 1 | 1.15M | 1.126 | 0.013 | 1177 |
| qwen25_15b__seed43__hidden_lora_r2 | hidden_lora | 2 | 2.31M | 1.103 | -0.010 | 1208 |
| qwen25_15b__seed43__vocab_lora_r1 | hidden_lora | 1 | 1.46M | 1.121 | 0.008 | 1212 |
| qwen25_15b__seed43__vocab_lora_r2 | hidden_lora | 2 | 2.62M | 1.099 | -0.014 | 1172 |
| qwen25_15b__seed44__combined_inlmh_r1 | affine_input_lm_head_plus_hidden_lora | 1 | 1.25M | 1.114 | 0.001 | 1188 |
| qwen25_15b__seed44__combined_inlmh_r2 | affine_input_lm_head_plus_hidden_lora | 2 | 2.41M | 1.096 | -0.017 | 1218 |
| qwen25_15b__seed44__hidden_lora_r1 | hidden_lora | 1 | 1.15M | 1.127 | 0.014 | 1183 |
| qwen25_15b__seed44__hidden_lora_r2 | hidden_lora | 2 | 2.31M | 1.105 | -0.008 | 1209 |
| qwen25_15b__seed44__vocab_lora_r1 | hidden_lora | 1 | 1.46M | 1.120 | 0.007 | 1215 |
| qwen25_15b__seed44__vocab_lora_r2 | hidden_lora | 2 | 2.62M | 1.098 | -0.015 | 1244 |

```json
[
  {
    "run_id": "qwen25_15b__seed43__combined_inlmh_r1",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 43,
    "trainable": 1253888,
    "total": 1544968192,
    "pct": 0.08115947023976013,
    "eval_loss": 1.113,
    "train_runtime_s": 1181.0,
    "train_loss": 1.183
  },
  {
    "run_id": "qwen25_15b__seed43__combined_inlmh_r2",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 43,
    "trainable": 2407936,
    "total": 1546122240,
    "pct": 0.1557403378402991,
    "eval_loss": 1.095,
    "train_runtime_s": 1217.0,
    "train_loss": 1.166
  },
  {
    "run_id": "qwen25_15b__seed43__hidden_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 1154048,
    "total": 1544868352,
    "pct": 0.07470202872017925,
    "eval_loss": 1.126,
    "train_runtime_s": 1177.0,
    "train_loss": 1.199
  },
  {
    "run_id": "qwen25_15b__seed43__hidden_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 2308096,
    "total": 1546022400,
    "pct": 0.14929253288956226,
    "eval_loss": 1.103,
    "train_runtime_s": 1208.0,
    "train_loss": 1.177
  },
  {
    "run_id": "qwen25_15b__seed43__vocab_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 1460992,
    "total": 1545175296,
    "pct": 0.09455186112424101,
    "eval_loss": 1.121,
    "train_runtime_s": 1212.0,
    "train_loss": 1.193
  },
  {
    "run_id": "qwen25_15b__seed43__vocab_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 2615040,
    "total": 1546329344,
    "pct": 0.16911274497549728,
    "eval_loss": 1.099,
    "train_runtime_s": 1172.0,
    "train_loss": 1.172
  },
  {
    "run_id": "qwen25_15b__seed44__combined_inlmh_r1",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 44,
    "trainable": 1253888,
    "total": 1544968192,
    "pct": 0.08115947023976013,
    "eval_loss": 1.114,
    "train_runtime_s": 1188.0,
    "train_loss": 1.183
  },
  {
    "run_id": "qwen25_15b__seed44__combined_inlmh_r2",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 44,
    "trainable": 2407936,
    "total": 1546122240,
    "pct": 0.1557403378402991,
    "eval_loss": 1.096,
    "train_runtime_s": 1218.0,
    "train_loss": 1.165
  },
  {
    "run_id": "qwen25_15b__seed44__hidden_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 1154048,
    "total": 1544868352,
    "pct": 0.07470202872017925,
    "eval_loss": 1.127,
    "train_runtime_s": 1183.0,
    "train_loss": 1.199
  },
  {
    "run_id": "qwen25_15b__seed44__hidden_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 2308096,
    "total": 1546022400,
    "pct": 0.14929253288956226,
    "eval_loss": 1.105,
    "train_runtime_s": 1209.0,
    "train_loss": 1.177
  },
  {
    "run_id": "qwen25_15b__seed44__vocab_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 1460992,
    "total": 1545175296,
    "pct": 0.09455186112424101,
    "eval_loss": 1.12,
    "train_runtime_s": 1215.0,
    "train_loss": 1.192
  },
  {
    "run_id": "qwen25_15b__seed44__vocab_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 2615040,
    "total": 1546329344,
    "pct": 0.16911274497549728,
    "eval_loss": 1.098,
    "train_runtime_s": 1244.0,
    "train_loss": 1.169
  }
]
```
