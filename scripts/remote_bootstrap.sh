#!/usr/bin/env bash
# 远程一键安装环境并执行命令
# 用法: bash scripts/remote_bootstrap.sh [command...]
# 示例: bash scripts/remote_bootstrap.sh python scripts/run_baseline.py --dataset fewnerd --mode fewshot

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

CUDA_VER="${CUDA_VER:-cu124}"
echo "[info] CUDA_VER=$CUDA_VER"

echo "[info] running setup_autodl.sh"
bash setup_autodl.sh

echo "[info] activating conda"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate patent-agent

echo "[info] verifying CUDA"
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'device_count:', torch.cuda.device_count())"

if [ $# -gt 0 ]; then
    echo "[info] executing: $*"
    exec "$@"
else
    echo "[ok] bootstrap done. run training manually."
fi
