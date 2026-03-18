#!/usr/bin/env bash
# =============================================================================
#  AutoDL：持续运行直到全部实验完成（剩余主实验 → 梯度+隔离）
#  用法: bash scripts/autodl_run_all_until_done.sh
#  与 nohup 配合可保持一直运行直到实验完成。
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "=================================================="
echo "  AutoDL 持续运行直到实验完成  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  ROOT: $ROOT"
echo "=================================================="

# 0. 数据盘挂载（含 data/artifacts/logs/models）
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
if [ -d "$AUTODL_TMP" ] && [ ! -L "$ROOT/data" ]; then
  echo "[step 0] 挂载 data/artifacts/logs/models 到数据盘..."
  mkdir -p "$AUTODL_TMP/patent-extract-agent/data" "$AUTODL_TMP/patent-extract-agent/artifacts" "$AUTODL_TMP/patent-extract-agent/logs" "$AUTODL_TMP/patent-extract-agent/models"
  [ -d "$ROOT/data" ] && [ ! -L "$ROOT/data" ] && mv "$ROOT/data" "$AUTODL_TMP/patent-extract-agent/" || true
  [ -d "$ROOT/artifacts" ] && [ ! -L "$ROOT/artifacts" ] && mv "$ROOT/artifacts" "$AUTODL_TMP/patent-extract-agent/" || true
  [ -d "$ROOT/logs" ] && [ ! -L "$ROOT/logs" ] && mv "$ROOT/logs" "$AUTODL_TMP/patent-extract-agent/" || true
  if [ -d /root/models ] && [ ! -L /root/models ]; then mv /root/models "$AUTODL_TMP/patent-extract-agent/" && ln -sfn "$AUTODL_TMP/patent-extract-agent/models" /root/models; elif [ ! -e /root/models ]; then ln -sfn "$AUTODL_TMP/patent-extract-agent/models" /root/models; fi
  ln -sfn "$AUTODL_TMP/patent-extract-agent/data" "$ROOT/data"
  ln -sfn "$AUTODL_TMP/patent-extract-agent/artifacts" "$ROOT/artifacts"
  ln -sfn "$AUTODL_TMP/patent-extract-agent/logs" "$ROOT/logs"
  echo "  已挂载到数据盘"
fi

# 1. 释放已有模型参数
echo "[step 1] 释放已有模型权重..."
python scripts/release_artifact_models.py --artifacts-dir "$ROOT/artifacts" || true

# 2. 终止旧进程
echo "[step 2] 终止旧训练进程..."
pkill -f 'train_bilstm_crf|train_seq_ner|train_fewshot_proto_span|train_span_ner' 2>/dev/null || true
pkill -f 'run_full_experiment|run_remaining_experiments|run_gradient_isolate' 2>/dev/null || true
python scripts/kill_training_processes.py 2>/dev/null || true
sleep 3
mkdir -p "$ROOT/logs"

# 3. 剩余主实验（B2r, B-Span+Aug）
echo ""
echo "[step 3] 运行剩余主实验..."
python scripts/run_remaining_experiments.py --multi-gpu || true

# 4. 梯度+隔离统一实验（持续到全部跑完）
echo ""
echo "[step 4] 运行梯度+隔离统一实验..."
python scripts/run_gradient_isolate_unified.py --multi-gpu "$@" || true

echo ""
echo "=================================================="
echo "[ok] 全部实验已跑完  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  查看: python _progress.py"
echo "=================================================="
