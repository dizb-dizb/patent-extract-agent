#!/usr/bin/env bash
# =============================================================================
#  AutoDL：数据梯度+隔离统一实验
#  梯度与隔离在一起：每个实验 = n 样本 + meta-train/meta-test 类别隔离
#  用法: bash scripts/autodl_run_gradient_isolate.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# 0. 若有数据盘，将 data/artifacts/logs/models 放到数据盘，防止系统盘占满（含模型参数）
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
if [ -d "$AUTODL_TMP" ] && [ ! -L "$ROOT/data" ]; then
  echo "[step 0] 将 data/artifacts/logs/models 挂载到数据盘 $AUTODL_TMP ..."
  mkdir -p "$AUTODL_TMP/patent-extract-agent/data" "$AUTODL_TMP/patent-extract-agent/artifacts" "$AUTODL_TMP/patent-extract-agent/logs" "$AUTODL_TMP/patent-extract-agent/models"
  [ -d "$ROOT/data" ] && [ ! -L "$ROOT/data" ] && mv "$ROOT/data" "$AUTODL_TMP/patent-extract-agent/" || true
  [ -d "$ROOT/artifacts" ] && [ ! -L "$ROOT/artifacts" ] && mv "$ROOT/artifacts" "$AUTODL_TMP/patent-extract-agent/" || true
  [ -d "$ROOT/logs" ] && [ ! -L "$ROOT/logs" ] && mv "$ROOT/logs" "$AUTODL_TMP/patent-extract-agent/" || true
  if [ -d /root/models ] && [ ! -L /root/models ]; then mv /root/models "$AUTODL_TMP/patent-extract-agent/" && ln -sfn "$AUTODL_TMP/patent-extract-agent/models" /root/models; elif [ ! -e /root/models ]; then ln -sfn "$AUTODL_TMP/patent-extract-agent/models" /root/models; fi
  ln -sfn "$AUTODL_TMP/patent-extract-agent/data" "$ROOT/data"
  ln -sfn "$AUTODL_TMP/patent-extract-agent/artifacts" "$ROOT/artifacts"
  ln -sfn "$AUTODL_TMP/patent-extract-agent/logs" "$ROOT/logs"
  echo "  已挂载到数据盘（含模型参数），保留 metrics 后会自动释放"
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "=================================================="
echo "  AutoDL 梯度+隔离统一实验  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  ROOT: $ROOT"
echo "=================================================="

# 1. 终止旧训练进程（注意：不能匹配自身脚本名）
echo "[step 1] 终止旧训练进程..."
pkill -f 'train_bilstm_crf|train_seq_ner|train_fewshot_proto_span|train_span_ner' 2>/dev/null || true
pkill -f 'run_full_experiment|run_remaining_experiments|run_data_gradient_experiment' 2>/dev/null || true
python scripts/kill_training_processes.py 2>/dev/null || true
sleep 3
echo "  已清理"

mkdir -p "$ROOT/logs"

# 2. 运行统一实验
echo ""
echo "[step 2] 梯度+隔离统一实验 (n=10,100,1000 + meta-train/meta-test)"
python scripts/run_gradient_isolate_unified.py --multi-gpu "$@"

echo ""
echo "[ok] 梯度+隔离统一实验完成"
echo "  输出: artifacts/run_*_n{10,100,1000}[_isolate]/"
echo "  查看: python _progress.py"
echo "=================================================="
