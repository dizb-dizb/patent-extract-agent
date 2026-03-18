#!/usr/bin/env bash
# =============================================================================
#  AutoDL：修复主实验缺失/错误数据
#  等待梯度+隔离实验完成后自动开始
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "=================================================="
echo "  AutoDL 主实验修复  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  ROOT: $ROOT"
echo "=================================================="

# 等待梯度+隔离实验完成
if pgrep -f 'run_gradient_isolate_unified' > /dev/null 2>&1; then
    echo "[wait] 梯度+隔离实验仍在运行，等待完成..."
    while pgrep -f 'run_gradient_isolate_unified' > /dev/null 2>&1; do
        sleep 60
    done
    echo "[ok] 梯度+隔离实验已结束"
fi

mkdir -p "$ROOT/logs"

echo ""
echo "[step] 修复主实验"
python scripts/run_fix_main_experiments.py --multi-gpu --models-dir /root/models "$@"

echo ""
echo "[ok] 主实验修复完成"
echo "  查看: python _progress.py"
echo "=================================================="
