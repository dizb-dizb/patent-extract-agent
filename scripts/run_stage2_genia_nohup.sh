#!/usr/bin/env bash
# 二阶段 Genia 后台训练（脚本在 Linux 上执行，避免从 Windows 粘贴多行命令时粘坏）
# 用法:
#   bash scripts/run_stage2_genia_nohup.sh
#   SCL_WEIGHT=0.1 LR=3e-4 bash scripts/run_stage2_genia_nohup.sh

set -euo pipefail
ROOT="/root/patent-extract-agent"
PY="${PY:-/root/miniconda3/bin/python}"
LOG="${ROOT}/logs/stage2_genia.log"
SCL_WEIGHT="${SCL_WEIGHT:-0.0}"
LR="${LR:-3e-4}"

cd "$ROOT"
mkdir -p "$ROOT/logs"
pkill -f "train_stage2_projector_proto.py" 2>/dev/null || true
sleep 1

: > "$LOG"
nohup "$PY" -u scripts/train_stage2_projector_proto.py \
  --data "$ROOT/data/benchmarks/genia/train_100.jsonl" \
  --stage1_ckpt "$ROOT/artifacts/two_stage_decoupled/genia/encoder_span_proj.pt" \
  --output_dir "$ROOT/artifacts/two_stage_decoupled/genia" \
  --encoder roberta-base \
  --batch_episodes 8 \
  --lr "$LR" \
  --scl_weight "$SCL_WEIGHT" \
  --fp16 \
  >> "$LOG" 2>&1 &
echo "Started PID=$!  log=$LOG"
echo "  tail -f $LOG"
