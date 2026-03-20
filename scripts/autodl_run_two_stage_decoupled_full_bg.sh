#!/usr/bin/env bash
# AutoDL 后台一键：完整二阶段解耦 baseline 实验（全数据集）+ 评测
# 用法:
#   bash scripts/autodl_run_two_stage_decoupled_full_bg.sh
#   bash scripts/autodl_run_two_stage_decoupled_full_bg.sh genia
#   DATASETS=genia,fewnerd,chemdner STAGE1_SIZE=10000 STAGE2_SIZE=100 N_EVAL=100 SCL_WEIGHT=0 \
#     bash scripts/autodl_run_two_stage_decoupled_full_bg.sh

set -euo pipefail

ROOT="/root/patent-extract-agent"
PY="/root/miniconda3/bin/python"

STAGE1_SIZE="${STAGE1_SIZE:-10000}"
STAGE2_SIZE="${STAGE2_SIZE:-100}"
N_EVAL="${N_EVAL:-100}"
# 阶段二 SCL：0=关闭（默认），>0 开启，如 0.5
SCL_WEIGHT="${SCL_WEIGHT:-0}"
if [ "${1:-}" != "" ]; then
  DATASETS="$1"
else
  DATASETS="${DATASETS:-genia,fewnerd,chemdner}"
fi

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/two_stage_decoupled_all_full_$(date +%Y%m%d_%H%M%S).log"

cd "$ROOT"

echo "[start] $(date '+%F %T') datasets=$DATASETS stage1=$STAGE1_SIZE stage2=$STAGE2_SIZE n_eval=$N_EVAL scl_weight=$SCL_WEIGHT"
echo "[log] $LOG_FILE"

# 仅清理训练相关进程，不匹配本脚本名，避免自杀
pkill -f 'train_span_entity_bce.py' 2>/dev/null || true
pkill -f 'train_stage2_projector_proto.py' 2>/dev/null || true
pkill -f 'run_two_stage_decoupled.py' 2>/dev/null || true
pkill -f 'run_gradient_isolate_unified.py' 2>/dev/null || true
pkill -f 'train_fewshot_proto_span.py' 2>/dev/null || true
pkill -f 'train_span_ner.py' 2>/dev/null || true
pkill -f 'train_mrc_ner.py' 2>/dev/null || true
pkill -f 'run_mrc_roberta' 2>/dev/null || true

{
  IFS=',' read -r -a DS_ARR <<< "$DATASETS"
  for DS in "${DS_ARR[@]}"; do
    DS="$(echo "$DS" | xargs)"
    [ -z "$DS" ] && continue
    echo ""
    echo "=================================================="
    echo "[dataset] $DS"
    echo "=================================================="
    echo "[1/3][$DS] two-stage decoupled train"
    "$PY" -u scripts/run_two_stage_decoupled.py \
      --dataset "$DS" \
      --stage1_size "$STAGE1_SIZE" \
      --stage2_size "$STAGE2_SIZE" \
      --build_subsets \
      --encoder roberta-base \
      --batch_stage1 32 \
      --batch_ep_stage2 8 \
      --scl_weight "$SCL_WEIGHT" \
      --num_workers 4 \
      --fp16

    echo "[2/3][$DS] eval on official test split"
    "$PY" -u scripts/eval_two_stage_decoupled_projector_proto.py \
      --dataset "$DS" \
      --split test \
      --encoder roberta-base \
      --stage2_ckpt "artifacts/two_stage_decoupled/${DS}/projector_stage2.pt" \
      --n_way 5 \
      --k_shot 5 \
      --n_eval "$N_EVAL"

    echo "[3/3][$DS] done"
    echo "[metrics][$DS] artifacts/two_stage_decoupled_eval_${DS}_test/metrics.json"
  done
  echo "[end] $(date '+%F %T')"
} 2>&1 | tee "$LOG_FILE"

