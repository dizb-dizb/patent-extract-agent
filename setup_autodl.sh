#!/usr/bin/env bash
set -euo pipefail

# AutoDL quick setup script (Linux).
# Usage (on AutoDL):
#   bash setup_autodl.sh

ENV_NAME="${ENV_NAME:-patent-agent}"
PY_VER="${PY_VER:-3.10}"

# AutoDL 国内 HuggingFace 镜像（无GPU登录节点和有GPU计算节点均适用）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "[info] HF_ENDPOINT=${HF_ENDPOINT}"

echo "[info] creating conda env: ${ENV_NAME} (python=${PY_VER})"
if ! command -v conda >/dev/null 2>&1; then
  echo "[fail] conda not found. Please use AutoDL image with conda."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[info] env exists: ${ENV_NAME}"
else
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi

conda activate "${ENV_NAME}"

echo "[info] upgrading pip"
python -m pip install -U pip

CUDA_VER="${CUDA_VER:-cu124}"
echo "[info] installing PyTorch (CUDA_VER=${CUDA_VER})"
python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_VER}"

echo "[info] installing project requirements"
python -m pip install -r requirements-autodl.txt

echo "[ok] done. activate with: conda activate ${ENV_NAME}"

