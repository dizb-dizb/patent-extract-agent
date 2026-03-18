#!/usr/bin/env bash
# =============================================================================
#  补充运行实验：B-Span + B2r/B4r/B5/B-Span+Aug/Ours/Ours-r（与 run_remaining_experiments 对齐）
#  用法: bash scripts/run_supplementary.sh [--bspan-only] [--remaining]
#  --bspan-only: 仅跑 B-Span
#  --remaining: 跑完整剩余实验（B2r,B4r,B5,B-Span+Aug,Ours,Ours-r）
#  默认: 先终止旧进程，再跑 run_remaining_experiments.py
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

for CONDA_SH in /root/miniconda3/etc/profile.d/conda.sh /opt/conda/etc/profile.d/conda.sh ~/miniconda3/etc/profile.d/conda.sh; do
    [ -f "$CONDA_SH" ] && source "$CONDA_SH" && break
done
conda activate patent-agent 2>/dev/null || conda activate base 2>/dev/null || true

PY=$(command -v python 2>/dev/null || command -v python3 2>/dev/null)
[ -z "$PY" ] && PY="python"

# 1. 终止旧训练进程
echo "[phase] 终止旧训练进程..."
$PY scripts/kill_training_processes.py 2>/dev/null || pkill -f 'train_bilstm_crf|train_seq_ner|train_fewshot_proto_span|train_span_ner' 2>/dev/null || true
sleep 2

DO_BSPAN=0
DO_REMAINING=1
for arg in "$@"; do
    case "$arg" in
        --bspan-only) DO_REMAINING=0; DO_BSPAN=1 ;;
        --remaining) DO_REMAINING=1 ;;
    esac
done

echo "=================================================="
echo "  补充运行实验  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  B-Span: $DO_BSPAN   剩余实验: $DO_REMAINING"
echo "=================================================="

# ── B-Span: BERT Span 分类器（无原型网络）────────────────
if [ "$DO_BSPAN" -eq 1 ]; then
    echo ""
    echo "[phase] B-Span (fewnerd genia chemdner)"
    for DS in fewnerd genia chemdner; do
        echo "[run] B-Span $DS ..."
        $PY scripts/run_baseline.py --dataset "$DS" --mode supervised \
            --encoder-type transformer --encoder "${ENCODER:-bert-base-cased}" \
            --epochs 5 --n_way 5 --k_shot 5 --seed 42 --multi-gpu
        echo "[ok] B-Span $DS done"
    done
fi

# ── 完整剩余实验：B2r, B4r, B5, B-Span+Aug, Ours, Ours-r ────────────────
if [ "$DO_REMAINING" -eq 1 ]; then
    echo ""
    echo "[phase] 运行剩余实验 (run_remaining_experiments.py)"
    $PY scripts/run_remaining_experiments.py
fi

echo ""
echo "[ok] 补充实验完成. 查看: python _progress.py 或 bash _monitor.sh"
