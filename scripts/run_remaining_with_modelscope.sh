#!/bin/bash
# 使用 ModelScope 下载模型后运行剩余实验
# 解决 HuggingFace 网络不可达问题
set -e
cd "$(dirname "$0")/.."
MODELS_DIR="${MODELS_DIR:-/root/models}"

echo "============================================================"
echo "  ModelScope 下载 + 剩余实验"
echo "  模型目录: $MODELS_DIR"
echo "============================================================"

# 1. 安装 modelscope（若未安装）
python -c "import modelscope" 2>/dev/null || pip install modelscope -q

# 2. 通过 ModelScope 下载模型
echo ""
echo "[1] 下载模型 (ModelScope)..."
python scripts/download_models_modelscope.py --output-dir "$MODELS_DIR"

# 3. 运行剩余实验（使用本地模型路径）
echo ""
echo "[2] 运行剩余实验..."
python scripts/run_remaining_experiments.py --models-dir "$MODELS_DIR" --multi-gpu

echo ""
echo "[ok] 完成"
