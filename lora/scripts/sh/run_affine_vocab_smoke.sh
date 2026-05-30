#!/usr/bin/env bash
set -eo pipefail

cd /home/wz/projects/mypro/im_exp/lora
source /home/wz/projects/mypro/im_exp/set
set -u
export PYTHONPATH=/home/wz/projects/mypro/im_exp/lora/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

python scripts/train_affine_vocab_lora.py \
  --model-path /home/wz/projects/mypro/im_exp/models/Qwen3-0.6B-Base \
  --train-data /home/wz/projects/mypro/im_exp/minimind/dataset/lora_exam.jsonl \
  --output-dir outputs/affine_vocab/smoke/qwen3_0_6b/affine_input \
  --variant affine_input \
  --max-seq-len 256 \
  --max-train-samples 8 \
  --max-steps 2 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 2e-4 \
  --affine-rank 4 \
  --affine-alpha 8 \
  --bf16 \
  --gradient-checkpointing
