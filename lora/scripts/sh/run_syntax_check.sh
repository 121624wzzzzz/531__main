#!/usr/bin/env bash
set -euo pipefail

cd /home/wz/projects/mypro/im_exp/lora
source /home/wz/projects/mypro/im_exp/set >/dev/null
python -m py_compile \
  scripts/train_affine_vocab_lora.py \
  scripts/eval_base_loss.py \
  scripts/eval_merge_equivalence.py
