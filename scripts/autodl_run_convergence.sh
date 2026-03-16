#!/usr/bin/env bash
# =============================================================================
#  AutoDL 收敛导向完整实验
#  使用合理 epoch / max_episodes 保证模型充分收敛，获取准确实验结果
#
#  用法:
#    bash scripts/autodl_run_convergence.sh              # 完整实验
#    bash scripts/autodl_run_convergence.sh --fast        # 快速模式 B1-B4
#    RESET=1 bash scripts/autodl_run_convergence.sh       # 清空后重跑
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "=================================================="
echo "  AutoDL 收敛导向实验  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  EPOCHS: B1(8/12/15) B2(5) Proto(8) BWT(3)"
echo "  MAX_EPISODES: Proto(1000/800/800) BWT(300)"
echo "=================================================="

# 环境
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    bash setup_autodl.sh 2>/dev/null || true
    conda activate patent-agent 2>/dev/null || conda activate base
fi

RESET_ARGS=""
[[ "${RESET:-0}" == "1" ]] && RESET_ARGS="--reset"

python scripts/run_full_experiment.py \
    --n_way 5 \
    --k_shot 5 \
    --datasets fewnerd,genia,chemdner \
    --multi-gpu \
    $RESET_ARGS \
    "$@"

echo ""
echo "[ok] 完成. 状态: python scripts/check_experiment_status.py"
