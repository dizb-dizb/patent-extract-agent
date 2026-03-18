#!/usr/bin/env bash
# =============================================================================
#  AutoDL 补充实验：终止旧进程 + 运行剩余实验 (B2r, B4r, B5, B-Span+Aug, Ours, Ours-r)
#  用法: bash scripts/autodl_run_supplementary.sh
#        bash scripts/autodl_run_supplementary.sh --datasets fewnerd
#        bash scripts/autodl_run_supplementary.sh --no-multi-gpu
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# HuggingFace 镜像 (AutoDL 国内)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "=================================================="
echo "  AutoDL 补充实验  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  ROOT: $ROOT"
echo "=================================================="

# 1. 终止旧训练进程
echo "[step 1] 终止旧训练进程..."
pkill -f 'train_bilstm_crf|train_seq_ner|train_fewshot_proto_span|train_span_ner|run_full_experiment|run_remaining' 2>/dev/null || true
python scripts/kill_training_processes.py 2>/dev/null || true
sleep 3
echo "  已清理"

# 2. 确保 logs 目录存在
mkdir -p "$ROOT/logs"

# 3. 运行剩余实验
echo ""
echo "[step 2] 运行剩余实验 (run_remaining_experiments.py)"
python scripts/run_remaining_experiments.py "$@"

echo ""
echo "[ok] 补充实验完成"
echo "  查看: bash _monitor.sh  或  python _progress.py"
echo "=================================================="
