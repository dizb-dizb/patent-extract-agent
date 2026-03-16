#!/usr/bin/env bash
# =============================================================================
#  完整实验 - 从零开始重新运行
#
#  用法:
#    bash scripts/run_from_scratch.sh              # 完整实验 (B1-B5+Ours + BWT + OOD)
#    bash scripts/run_from_scratch.sh --fast       # 快速模式 (B1-B4, 跳过 evidence)
#    bash scripts/run_from_scratch.sh --skip-data  # 已有数据，跳过下载/转换
#
#  环境变量 (可选):
#    RESET=1           清空 artifacts 后重跑
#    ENCODER           本地模型路径，如 /root/models/bert-base-cased
#    HF_ENDPOINT       HuggingFace 镜像，如 https://hf-mirror.com
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# HuggingFace 镜像 (AutoDL 国内环境)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "=================================================="
echo "  完整实验 - 从零开始  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  ROOT: $ROOT"
echo "=================================================="

# ── 1. 环境 ─────────────────────────────────────────────────────────────────
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | grep -q "patent-agent"; then
        conda activate patent-agent 2>/dev/null || conda activate base
    else
        conda activate base 2>/dev/null || true
    fi
fi

# ── 2. 可选：清空旧结果 ─────────────────────────────────────────────────────
RESET_ARGS=""
if [[ "${RESET:-0}" == "1" ]]; then
    RESET_ARGS="--reset"
    echo "[info] 将清空 artifacts 后重跑"
fi

# ── 3. 可选：本地模型路径 (AutoDL 预下载) ────────────────────────────────────
EXTRA=""
if [[ -n "${ENCODER:-}" ]]; then
    EXTRA="--encoder $ENCODER --roberta-encoder ${ROBERTA_ENCODER:-$ENCODER}"
    echo "[info] 使用本地模型: $ENCODER"
fi

# ── 4. 执行完整实验 ─────────────────────────────────────────────────────────
echo ""
echo "[run] python scripts/run_full_experiment.py --epochs 3 $RESET_ARGS $EXTRA $*"
echo ""

python scripts/run_full_experiment.py \
    --n_way 5 \
    --k_shot 5 \
    --datasets fewnerd,genia,chemdner \
    $RESET_ARGS \
    $EXTRA \
    "$@"

echo ""
echo "=================================================="
echo "  完成  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  汇总: artifacts/experiments/summary.json"
echo "  状态: python scripts/check_experiment_status.py"
echo "=================================================="
