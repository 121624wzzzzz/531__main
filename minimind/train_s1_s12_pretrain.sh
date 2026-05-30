#!/usr/bin/env bash
# ==============================================================================
# S1-S13 dense pretrain runner.
#
# Examples:
#   bash train_s1_s12_pretrain.sh
#   VARIANTS=s3,s4 MAX_STEPS=5 DATA_PATH=../dataset/minimind/pretrain_t2t_mini.jsonl bash train_s1_s12_pretrain.sh
#   START=s4 END=s13 FROM_RESUME=1 bash train_s1_s12_pretrain.sh
# ==============================================================================
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
ROOT=$(pwd)

TORCH24_PREFIX="/home/wz/anaconda3/envs/torch24"
if [[ ! -x "$TORCH24_PREFIX/bin/python" ]]; then
    echo "[s1-s12] 找不到 torch24 Python: $TORCH24_PREFIX/bin/python"
    exit 1
fi

export CONDA_PREFIX="$TORCH24_PREFIX"
export CONDA_DEFAULT_ENV="torch24"
export CUDA_HOME="$TORCH24_PREFIX"
export CUDA_PATH="$TORCH24_PREFIX"
export PATH="$TORCH24_PREFIX/bin:${PATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
DATA_PATH="${DATA_PATH:-../dataset/minimind/pretrain_t2t.jsonl}"
SAVE_DIR="${SAVE_DIR:-weights/final}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-weights/resume}"
WEIGHT_PREFIX="${WEIGHT_PREFIX:-pretrain_v2}"
SEED="${SEED:-42}"
RANK="${RANK:-32}"
BATCH_SIZE="${BATCH_SIZE:-224}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-340}"
EPOCHS="${EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FROM_RESUME="${FROM_RESUME:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
LM_HEAD_BIAS="${LM_HEAD_BIAS:-1}"
TRAIN_SPLIT_RATIO="${TRAIN_SPLIT_RATIO:-0.99}"
TOKENIZER_PATH="${TOKENIZER_PATH:-../model}"
GRAD_LOG_INTERVAL="${GRAD_LOG_INTERVAL:-0}"
GRAD_SAVE_TENSORS="${GRAD_SAVE_TENSORS:-0}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/s1-s12-pretrain-v2}"

mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONUNBUFFERED=1

all_variants=(s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13)

selected_variants=()
if [[ -n "${VARIANTS:-}" ]]; then
    IFS=',' read -r -a selected_variants <<< "$VARIANTS"
else
    take=0
    [[ -z "${START:-}" ]] && take=1
    for variant in "${all_variants[@]}"; do
        [[ -n "${START:-}" && "$variant" == "$START" ]] && take=1
        [[ "$take" == "1" ]] && selected_variants+=("$variant")
        [[ -n "${END:-}" && "$variant" == "$END" ]] && break
    done
fi

if [[ "${#selected_variants[@]}" -eq 0 ]]; then
    echo "[s1-s13] 没有匹配到要训练的变体，请检查 VARIANTS/START/END"
    exit 1
fi

echo "================================================================"
echo "[s1-s12] start at $(date '+%F %T')"
echo "  GPUS               = $GPUS  (DDP nproc=$NPROC)"
echo "  VARIANTS           = ${selected_variants[*]}"
echo "  DATA_PATH          = $DATA_PATH"
echo "  SAVE_DIR           = $SAVE_DIR"
echo "  CHECKPOINT_DIR     = $CHECKPOINT_DIR"
echo "  WEIGHT_PREFIX      = $WEIGHT_PREFIX"
echo "  SEED               = $SEED"
echo "  RANK               = $RANK"
echo "  LM_HEAD_BIAS       = $LM_HEAD_BIAS"
echo "  TRAIN_SPLIT_RATIO  = $TRAIN_SPLIT_RATIO"
echo "  TOKENIZER_PATH     = $TOKENIZER_PATH"
echo "  GRAD_LOG_INTERVAL  = $GRAD_LOG_INTERVAL"
echo "  GRAD_SAVE_TENSORS  = $GRAD_SAVE_TENSORS"
echo "  BATCH_SIZE         = $BATCH_SIZE"
echo "  ACCUMULATION_STEPS = $ACCUMULATION_STEPS"
echo "  effective_batch    = $((NPROC * BATCH_SIZE * ACCUMULATION_STEPS)) sequences"
echo "  MAX_SEQ_LEN        = $MAX_SEQ_LEN"
echo "  EPOCHS             = $EPOCHS"
echo "  LEARNING_RATE      = $LEARNING_RATE"
echo "  FROM_RESUME        = $FROM_RESUME"
echo "  MAX_STEPS          = $MAX_STEPS"
echo "  LOG_DIR            = $LOG_DIR"
echo "================================================================"

run_variant() {
    local variant=$1
    local save_weight="${WEIGHT_PREFIX}_${variant}"
    local logfile="$LOG_DIR/${variant}.log"
    local started=$(date +%s)

    echo
    echo "----------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')] >>> START variant: $variant"
    echo "  save_weight: $save_weight"
    echo "  log: $logfile"
    echo "----------------------------------------------------------------"

    cd "$ROOT/trainer"
    if torchrun --standalone --nproc_per_node="$NPROC" train_pretrain.py \
        --data_path "$DATA_PATH" \
        --save_dir "$SAVE_DIR" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --save_weight "$save_weight" \
        --embedding_variant "$variant" \
        --embedding_variant_rank "$RANK" \
        --seed "$SEED" \
        --lm_head_bias "$LM_HEAD_BIAS" \
        --train_split_ratio "$TRAIN_SPLIT_RATIO" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --grad_log_interval "$GRAD_LOG_INTERVAL" \
        --grad_log_path "$LOG_DIR/${variant}_grad_stats.jsonl" \
        --grad_save_tensors "$GRAD_SAVE_TENSORS" \
        --grad_tensor_dir "$LOG_DIR/${variant}_grad_tensors" \
        --batch_size "$BATCH_SIZE" \
        --accumulation_steps "$ACCUMULATION_STEPS" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --epochs "$EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --log_interval "$LOG_INTERVAL" \
        --save_interval "$SAVE_INTERVAL" \
        --num_workers "$NUM_WORKERS" \
        --from_resume "$FROM_RESUME" \
        --max_steps "$MAX_STEPS" \
        2>&1 | tee "$logfile"; then
        local elapsed=$(( $(date +%s) - started ))
        echo "[$(date '+%H:%M:%S')] <<< FINISH $variant in $((elapsed/60))m$((elapsed%60))s"
    else
        local elapsed=$(( $(date +%s) - started ))
        echo "[$(date '+%H:%M:%S')] !!! FAIL $variant after $((elapsed/60))m$((elapsed%60))s"
        exit 1
    fi
    cd "$ROOT"
}

for variant in "${selected_variants[@]}"; do
    run_variant "$variant"
done

echo
echo "================================================================"
echo "[s1-s12] ALL DONE at $(date '+%F %T')"
echo "权重产出列表:"
ls -lh "$ROOT/trainer/$SAVE_DIR"/"${WEIGHT_PREFIX}"_s*_768.pth 2>/dev/null || echo "  (无)"
echo "================================================================"
