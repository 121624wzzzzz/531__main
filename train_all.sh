#!/usr/bin/env bash
# ==============================================================================
# minimind 完整训练编排
#
# 任务流（必须串行，因为 from_weight 依赖前一阶段）:
#   1) pretrain   on pretrain_t2t.jsonl   (7.7 GB) -> out/pretrain<suf>_768<msuf>.pth
#   2) full_sft   on sft_t2t.jsonl       (13.1 GB) -> out/full_sft<suf>_768<msuf>.pth
#   3) lora_medical    -> out/lora_medical<suf>_768<msuf>.pth
#   4) lora_exam       -> out/lora_exam<suf>_768<msuf>.pth
#   5) lora_identity   -> out/lora_identity<suf>_768<msuf>.pth
#
# 其中:
#   <msuf> = "_moe"     若 MOE=1     （由 trainer_utils.py 自动添加）
#   <suf>  = "_untied"  若 UNTIED=1   （embed_tokens 与 lm_head 不绑权重）
#
# 用法:
#   GPUS=1,2,3,4,5,6 bash train_all.sh                      # 默认 dense + tied (本仓已跑过)
#   GPUS=1,2,3,4,5,6 MOE=1 bash train_all.sh                # MoE 版本
#   GPUS=1,2,3,4,5,6 UNTIED=1 bash train_all.sh             # Untied 版本
#   GPUS=1,2,3,4,5,6 MOE=1 UNTIED=1 bash train_all.sh       # 同时开 (MoE × Untied)
#   STAGE=pretrain bash train_all.sh                        # 只跑某阶段
# ==============================================================================
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
ROOT=$(pwd)

# ---- 1. 激活 torch24 环境 ----
source "$ROOT/../set" >/dev/null 2>&1 || {
    echo "[train_all] 无法激活 torch24，请检查 ../set"; exit 1;
}

# ---- 2. 配置 ----
GPUS="${GPUS:-1,2,3,4,5,6}"
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
STAGE="${STAGE:-all}"
MOE="${MOE:-0}"
UNTIED="${UNTIED:-0}"

# 后缀逻辑
SUFFIX=""
EXTRA=()
TAG="dense-tied"
if [[ "$MOE" == "1" ]]; then
    EXTRA+=(--use_moe 1)
    TAG="moe"
fi
if [[ "$UNTIED" == "1" ]]; then
    EXTRA+=(--tie_word_embeddings 0)
    SUFFIX="_untied"
    if [[ "$MOE" == "1" ]]; then TAG="moe-untied"; else TAG="dense-untied"; fi
fi

LOG_DIR="$ROOT/logs/${TAG}"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONUNBUFFERED=1

echo "================================================================"
echo "[train_all] start at $(date '+%F %T')"
echo "  TAG                = $TAG"
echo "  MOE                = $MOE   (suffix from trainer_utils: $([[ $MOE == 1 ]] && echo _moe || echo none))"
echo "  UNTIED             = $UNTIED (save_weight suffix: '${SUFFIX:-<empty>}')"
echo "  GPUS               = $GPUS  (DDP nproc=$NPROC)"
echo "  STAGE              = $STAGE"
echo "  EXTRA torchrun args= ${EXTRA[*]:-<none>}"
echo "  ROOT               = $ROOT"
echo "  LOG_DIR            = $LOG_DIR"
echo "================================================================"

# ---- 3. 训练 wrapper ----
run() {
    local name=$1; shift
    local logfile="$LOG_DIR/${name}.log"
    local started=$(date +%s)

    echo
    echo "----------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')] >>> START stage: $name  (tag=$TAG)"
    echo "  cmd: torchrun --standalone --nproc_per_node=$NPROC $*"
    echo "  log: $logfile"
    echo "----------------------------------------------------------------"

    cd "$ROOT/trainer"
    if torchrun --standalone --nproc_per_node="$NPROC" "$@" 2>&1 | tee "$logfile"; then
        local elapsed=$(( $(date +%s) - started ))
        echo "[$(date '+%H:%M:%S')] <<< FINISH $name in $((elapsed/60))m$((elapsed%60))s"
    else
        local elapsed=$(( $(date +%s) - started ))
        echo "[$(date '+%H:%M:%S')] !!! FAIL $name after $((elapsed/60))m$((elapsed%60))s; aborting pipeline"
        exit 1
    fi
    cd "$ROOT"
}

# ---- 4. 阶段执行 ----
if [[ "$STAGE" == "all" || "$STAGE" == "pretrain" ]]; then
    run "pretrain${SUFFIX}" train_pretrain.py \
        --data_path ../dataset/pretrain_t2t.jsonl \
        --save_weight "pretrain${SUFFIX}" \
        "${EXTRA[@]}"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "sft" ]]; then
    run "full_sft${SUFFIX}" train_full_sft.py \
        --data_path ../dataset/sft_t2t.jsonl \
        --save_weight "full_sft${SUFFIX}" \
        --from_weight "pretrain${SUFFIX}" \
        "${EXTRA[@]}"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "lora" ]]; then
    for ds in lora_medical lora_exam lora_identity; do
        run "${ds}${SUFFIX}" train_lora.py \
            --lora_name "${ds}${SUFFIX}" \
            --data_path "../dataset/${ds}.jsonl" \
            --from_weight "full_sft${SUFFIX}" \
            "${EXTRA[@]}"
    done
fi

echo
echo "================================================================"
echo "[train_all] ALL DONE at $(date '+%F %T')  (tag=$TAG)"
echo "权重产出列表:"
ls -lh "$ROOT/out"/*"${SUFFIX}"_768*.pth 2>/dev/null || echo "  (无)"
echo "================================================================"
