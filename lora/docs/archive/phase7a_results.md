# Results: phase7a_20260520_035042

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| combined_inlmh_r1 | affine_input_lm_head_plus_hidden_lora | 1 | 2.76M | 0.957 | 0.000 | 3880 |
| combined_inlmh_r2 | affine_input_lm_head_plus_hidden_lora | 2 | 5.28M | 0.946 | -0.011 | 3945 |
| combined_inlmh_r4 | affine_input_lm_head_plus_hidden_lora | 4 | 10.33M | 0.931 | -0.027 | 3934 |
| hidden_lora_r1 | hidden_lora | 1 | 2.52M | 0.966 | 0.008 | 3874 |
| hidden_lora_r2 | hidden_lora | 2 | 5.05M | 0.952 | -0.005 | 3737 |
| hidden_lora_r4 | hidden_lora | 4 | 10.09M | 0.936 | -0.021 | 3932 |

```json
[
  {
    "run_id": "combined_inlmh_r1",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 2756096,
    "total": 7618372608,
    "pct": 0.03617696510545891,
    "eval_loss": 0.9574,
    "train_runtime_s": 3880.0,
    "train_loss": 1.019
  },
  {
    "run_id": "combined_inlmh_r2",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 5279232,
    "total": 7620895744,
    "pct": 0.06927311666947271,
    "eval_loss": 0.946,
    "train_runtime_s": 3945.0,
    "train_loss": 1.008
  },
  {
    "run_id": "combined_inlmh_r4",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 10325504,
    "total": 7625942016,
    "pct": 0.13539971820315505,
    "eval_loss": 0.9309,
    "train_runtime_s": 3934.0,
    "train_loss": 0.9956
  },
  {
    "run_id": "hidden_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 2523136,
    "total": 7618139648,
    "pct": 0.033120106962890895,
    "eval_loss": 0.9656,
    "train_runtime_s": 3874.0,
    "train_loss": 1.028
  },
  {
    "run_id": "hidden_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 5046272,
    "total": 7620662784,
    "pct": 0.06621828235983522,
    "eval_loss": 0.9523,
    "train_runtime_s": 3737.0,
    "train_loss": 1.015
  },
  {
    "run_id": "hidden_lora_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 10092544,
    "total": 7625709056,
    "pct": 0.13234892553445982,
    "eval_loss": 0.9362,
    "train_runtime_s": 3932.0,
    "train_loss": 1.0
  }
]
```
