# Results: phase5a_20260519_170658

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| hidden_lora_r16 | hidden_lora | 16 | 10.09M | 1.071 | -0.030 | 942 |
| hidden_lora_r2 | hidden_lora | 2 | 1.26M | 1.163 | 0.062 | 957 |
| hidden_lora_r32 | hidden_lora | 32 | 20.19M | 1.042 | -0.059 | 940 |
| hidden_lora_r4 | hidden_lora | 4 | 2.52M | 1.133 | 0.032 | 955 |
| hidden_lora_r8 | hidden_lora | 8 | 5.05M | 1.101 | 0.000 | 939 |

```json
[
  {
    "run_id": "hidden_lora_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 10092544,
    "total": 606142464,
    "pct": 1.6650448697156448,
    "eval_loss": 1.071,
    "train_runtime_s": 942.1,
    "train_loss": 1.139
  },
  {
    "run_id": "hidden_lora_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 1261568,
    "total": 597311488,
    "pct": 0.21120772416819816,
    "eval_loss": 1.163,
    "train_runtime_s": 956.8,
    "train_loss": 1.22
  },
  {
    "run_id": "hidden_lora_r32",
    "variant": "hidden_lora",
    "hidden_lora_rank": 32,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 20185088,
    "total": 616235008,
    "pct": 3.275550356269276,
    "eval_loss": 1.042,
    "train_runtime_s": 940.3,
    "train_loss": 1.118
  },
  {
    "run_id": "hidden_lora_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 2523136,
    "total": 598573056,
    "pct": 0.4215251546504626,
    "eval_loss": 1.133,
    "train_runtime_s": 955.2,
    "train_loss": 1.193
  },
  {
    "run_id": "hidden_lora_r8",
    "variant": "hidden_lora",
    "hidden_lora_rank": 8,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 5046272,
    "total": 601096192,
    "pct": 0.8395115569123418,
    "eval_loss": 1.101,
    "train_runtime_s": 939.3,
    "train_loss": 1.165
  }
]
```
