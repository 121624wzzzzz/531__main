# Results: phase9d_20260520_234301

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_7b__hidden_lora_r1 | hidden_lora | 1 | 2.52M | 0.966 | 0.000 | 3696 |
| qwen25_7b__hidden_lora_r2 | hidden_lora | 2 | 5.05M | 0.952 | -0.013 | 3758 |
| qwen25_7b__input_lm_head_r1 | affine_input_lm_head_plus_hidden_lora | 1 | 2.76M | 0.957 | -0.008 | 3854 |
| qwen25_7b__input_lm_head_r2 | affine_input_lm_head_plus_hidden_lora | 2 | 5.28M | 0.946 | -0.020 | 3938 |
| qwen25_7b__input_only_r1 | affine_input_plus_hidden_lora | 1 | 2.64M | 0.965 | -0.001 | 3865 |
| qwen25_7b__input_only_r2 | affine_input_plus_hidden_lora | 2 | 5.16M | 0.951 | -0.015 | 3941 |
| qwen25_7b__lm_head_only_r1 | affine_lm_head_plus_hidden_lora | 1 | 2.64M | 0.959 | -0.007 | 3851 |
| qwen25_7b__lm_head_only_r2 | affine_lm_head_plus_hidden_lora | 2 | 5.16M | 0.946 | -0.019 | 3939 |

```json
[
  {
    "run_id": "qwen25_7b__hidden_lora_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 2523136,
    "total": 7618139648,
    "pct": 0.033120106962890895,
    "eval_loss": 0.9657,
    "train_runtime_s": 3696.0,
    "train_loss": 1.028
  },
  {
    "run_id": "qwen25_7b__hidden_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 5046272,
    "total": 7620662784,
    "pct": 0.06621828235983522,
    "eval_loss": 0.9524,
    "train_runtime_s": 3758.0,
    "train_loss": 1.015
  },
  {
    "run_id": "qwen25_7b__input_lm_head_r1",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 2756096,
    "total": 7618372608,
    "pct": 0.03617696510545891,
    "eval_loss": 0.9574,
    "train_runtime_s": 3854.0,
    "train_loss": 1.018
  },
  {
    "run_id": "qwen25_7b__input_lm_head_r2",
    "variant": "affine_input_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 5279232,
    "total": 7620895744,
    "pct": 0.06927311666947271,
    "eval_loss": 0.9457,
    "train_runtime_s": 3938.0,
    "train_loss": 1.008
  },
  {
    "run_id": "qwen25_7b__input_only_r1",
    "variant": "affine_input_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 2641408,
    "total": 7618257920,
    "pct": 0.03467207369109394,
    "eval_loss": 0.9646,
    "train_runtime_s": 3865.0,
    "train_loss": 1.026
  },
  {
    "run_id": "qwen25_7b__input_only_r2",
    "variant": "affine_input_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 5164544,
    "total": 7620781056,
    "pct": 0.06776922158042904,
    "eval_loss": 0.9509,
    "train_runtime_s": 3941.0,
    "train_loss": 1.013
  },
  {
    "run_id": "qwen25_7b__lm_head_only_r1",
    "variant": "affine_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 2637824,
    "total": 7618254336,
    "pct": 0.03462504510429619,
    "eval_loss": 0.9585,
    "train_runtime_s": 3851.0,
    "train_loss": 1.021
  },
  {
    "run_id": "qwen25_7b__lm_head_only_r2",
    "variant": "affine_lm_head_plus_hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 128.0,
    "seed": 42,
    "trainable": 5160960,
    "total": 7620777472,
    "pct": 0.06772222412952252,
    "eval_loss": 0.9463,
    "train_runtime_s": 3939.0,
    "train_loss": 1.009
  }
]
```
