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

# 环境 - 激活 conda，自动查找 python
for CONDA_SH in /root/miniconda3/etc/profile.d/conda.sh /root/anaconda3/etc/profile.d/conda.sh /opt/conda/etc/profile.d/conda.sh; do
    [ -f "$CONDA_SH" ] && source "$CONDA_SH" && break
done
conda activate patent-agent 2>/dev/null || conda activate base 2>/dev/null || true

PY=$(command -v python 2>/dev/null \
    || command -v python3 2>/dev/null \
    || ls /root/miniconda3/bin/python 2>/dev/null \
    || ls /opt/conda/bin/python 2>/dev/null)
if [ -z "$PY" ]; then
    echo "[error] python not found"; exit 1
fi
echo "[info] Python: $($PY --version 2>&1)"
echo "[info] CUDA:   $($PY -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), "GPUs")' 2>/dev/null || echo 'torch not loaded')"

RESET_ARGS=""
[[ "${RESET:-0}" == "1" ]] && RESET_ARGS="--reset"

# 使用 /root/models/bert-base-cased (已预下载)
MODEL_PATH="/root/models/bert-base-cased"
ENCODER_ARG=""
if [ -d "$MODEL_PATH" ]; then
    ENCODER_ARG="--encoder $MODEL_PATH"
    echo "[info] 使用本地模型: $MODEL_PATH"
fi

$PY scripts/run_full_experiment.py \
    --n_way 5 \
    --k_shot 5 \
    --datasets fewnerd,genia,chemdner \
    --multi-gpu \
    --skip-roberta \
    $ENCODER_ARG \
    $RESET_ARGS \
    "$@"

echo ""
echo "[ok] 完成. 状态: $PY _progress.py"
