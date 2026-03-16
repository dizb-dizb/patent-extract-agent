#!/usr/bin/env bash
# AutoDL 一键实验入口
#
# 用法:
#   bash scripts/autodl_run_all.sh --fast          # 快速验证: B1-B4 + 能力测试, 跳过 evidence
#   bash scripts/autodl_run_all.sh                 # 完整实验: B1-B5 + Ours + 能力测试
#   bash scripts/autodl_run_all.sh --fast --datasets fewnerd   # 只跑单个数据集
#
# 可选环境变量:
#   CUDA_VER=cu121  (默认 cu124, 对应 CUDA 12.4)
#   ENV_NAME=patent-agent  (默认 patent-agent)
#   EPOCHS=3               (默认 2)
#   N_WAY=5                (默认 5)
#   K_SHOT=5               (默认 5)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# AutoDL 国内 HuggingFace 镜像（绕过境外网络限制）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "[info] HF_ENDPOINT=${HF_ENDPOINT}"

echo "=================================================="
echo "  AutoDL 一键实验  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  ROOT: $ROOT"
echo "=================================================="

# ── 1. 环境安装 ─────────────────────────────────────────────────────────────
echo "[step 1/3] 安装/激活 Conda 环境"
CUDA_VER="${CUDA_VER:-cu124}"
ENV_NAME="${ENV_NAME:-patent-agent}"

if ! command -v conda &>/dev/null; then
    echo "[fail] 未找到 conda, 请使用带 conda 的 AutoDL 镜像"
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
bash setup_autodl.sh

conda activate "$ENV_NAME"

echo "[info] Python: $(python --version)"
echo "[info] PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "[info] CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "[info] GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
if python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    python -c 'import torch; print("[info] GPU:", torch.cuda.get_device_name(0))'
fi

# ── 2. 参数透传给 run_full_experiment.py ────────────────────────────────────
echo ""
echo "[step 2/3] 启动实验"
EPOCHS="${EPOCHS:-2}"
N_WAY="${N_WAY:-5}"
K_SHOT="${K_SHOT:-5}"
DATASETS="${DATASETS:-fewnerd,genia,chemdner}"

EXTRA_ARGS="--epochs $EPOCHS --n_way $N_WAY --k_shot $K_SHOT --datasets $DATASETS"

# Pass all script arguments through (e.g. --fast, --skip-data, --datasets ...)
python scripts/run_full_experiment.py $EXTRA_ARGS "$@"

# ── 3. 打印结果路径 ──────────────────────────────────────────────────────────
echo ""
echo "[step 3/3] 完成"
SUMMARY="$ROOT/artifacts/experiments/summary.json"
if [ -f "$SUMMARY" ]; then
    echo "[ok] 汇总结果: $SUMMARY"
    python - <<'PYEOF'
import json, pathlib, sys
p = pathlib.Path("artifacts/experiments/summary.json")
if not p.exists():
    sys.exit(0)
s = json.loads(p.read_text())
runs = s.get("runs", [])
print(f"\n{'Baseline':<8} {'Dataset':<12} {'F1':>6} {'FlatF1':>8} {'Note'}")
print("-" * 50)
for r in runs:
    ri = r.get("_run") or {}
    bl = r.get("baseline") or r.get("capability") or "?"
    ds = ri.get("dataset") or ri.get("test_ds") or "?"
    f1 = r.get("f1") or r.get("bwt") or 0.0
    ff = r.get("flat_f1") or r.get("f1") or 0.0
    note = ri.get("data_strategy", "")
    print(f"{bl:<8} {ds:<12} {f1:>6.4f} {ff:>8.4f}  {note}")
print(f"\nelapsed: {s.get('elapsed_seconds',0)/60:.1f} min")
PYEOF
else
    echo "[warn] 未找到汇总文件, 请检查实验日志"
fi
