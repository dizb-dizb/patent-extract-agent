#!/usr/bin/env bash
# 在 AutoDL 容器内执行：清理旧训练进程并后台启动「全数据集二阶段解耦 + 评测」
# 用法（SSH 登录后）: bash scripts/autodl_remote_start_two_stage_decoupled.sh

set -euo pipefail
ROOT="${ROOT:-/root/patent-extract-agent}"
cd "$ROOT"
mkdir -p logs
chmod +x scripts/autodl_run_two_stage_decoupled_full_bg.sh scripts/run_stage2_genia_nohup.sh 2>/dev/null || true

pkill -f train_span_entity_bce.py 2>/dev/null || true
pkill -f train_stage2_projector_proto.py 2>/dev/null || true
pkill -f run_two_stage_decoupled.py 2>/dev/null || true
sleep 2

LAUNCH_LOG="$ROOT/logs/run_two_stage_decoupled_full_bg_launcher.log"
nohup bash scripts/autodl_run_two_stage_decoupled_full_bg.sh > "$LAUNCH_LOG" 2>&1 &
echo "Started PID=$! (SCL off: SCL_WEIGHT=${SCL_WEIGHT:-0})"
echo "  tail -f $LAUNCH_LOG"
