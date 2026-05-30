#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

ROOT="/home/wz/projects/mypro/im_exp/minimind"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PY="/home/wz/anaconda3/envs/torch24/bin/python"
RUN_LOG="$ROOT/logs/backfill_s13_s5_tmux_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "[tmux-backfill] start $(date '+%F %T')"
echo "[tmux-backfill] log=$RUN_LOG"

echo "[tmux-backfill] MiniMind S13 remaining seeds"
for seed in 123 2026; do
  prefix="pretrain_v2_seed${seed}"
  log_dir="$ROOT/logs/s1-s12-pretrain-v2-seed${seed}"
  echo "[tmux-backfill] MiniMind seed=$seed prefix=$prefix log_dir=$log_dir $(date '+%F %T')"
  GPUS="$GPUS" VARIANTS=s13 SEED="$seed" WEIGHT_PREFIX="$prefix" LOG_DIR="$log_dir" \
    BATCH_SIZE=224 ACCUMULATION_STEPS=1 GRAD_SAVE_TENSORS=0 GRAD_LOG_INTERVAL=0 \
    bash train_s1_s12_pretrain.sh
  CUDA_VISIBLE_DEVICES="$GPUS" "$PY" results/eval_pretrain_loss.py \
    --variants s13 \
    --weight_prefix "$prefix" \
    --save_dir weights/final \
    --data_path dataset/minimind/pretrain_t2t.jsonl \
    --tokenizer_path model \
    --tail_ratio 0.01 \
    --max_samples 0 \
    --batch_size 64 \
    --num_workers 4 \
    --lm_head_bias 1 \
    --device cuda:1 \
    --output_csv "$log_dir/eval_pretrain_loss_s13.csv" \
    --output_json "$log_dir/eval_pretrain_loss_s13.json"
done

echo "[tmux-backfill] FineEdu/GPT-2 S13+S5 seeds"
for seed in 42 123 2026; do
  log_dir="$ROOT/logs/fineedu-gpt2-6b-seed${seed}"
  echo "[tmux-backfill] FineEdu seed=$seed log_dir=$log_dir $(date '+%F %T')"
  GPUS="$GPUS" VARIANTS=s13,s5 SEED="$seed" WEIGHT_PREFIX="fineedu_gpt2_6b_seed${seed}" \
    LOG_DIR="$log_dir" BATCH_SIZE=80 ACCUMULATION_STEPS=1 \
    GRAD_SAVE_TENSORS=0 GRAD_LOG_INTERVAL=1000 RUN_EVAL=1 EVAL_DEVICE=cuda:1 \
    bash train_fineedu_gpt2_pretrain.sh
done

echo "[tmux-backfill] done $(date '+%F %T')"
