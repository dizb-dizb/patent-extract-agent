#!/usr/bin/env bash
# =============================================================================
#  AutoDL：两阶段隔离训练 (Two-Stage Decoupled) 拉满性能
#  环境：RTX 4090 24GB, CUDA 12.4, PyTorch 2.5, 90GB RAM
#  用法: bash scripts/autodl_run_two_stage_decoupled.sh
#        bash scripts/autodl_run_two_stage_decoupled.sh --dataset fewnerd
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# 0. 数据盘挂载（系统盘 30GB 易满，数据/模型/日志放数据盘）
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
if [ -d "$AUTODL_TMP" ] && [ ! -L "$ROOT/data" ]; then
  echo "[step 0] 挂载 data/artifacts/logs/models 到数据盘 $AUTODL_TMP ..."
  mkdir -p "$AUTODL_TMP/patent-extract-agent/data" "$AUTODL_TMP/patent-extract-agent/artifacts" \
           "$AUTODL_TMP/patent-extract-agent/logs" "$AUTODL_TMP/patent-extract-agent/models"
  [ -d "$ROOT/data" ] && [ ! -L "$ROOT/data" ] && mv "$ROOT/data" "$AUTODL_TMP/patent-extract-agent/" || true
  [ -d "$ROOT/artifacts" ] && [ ! -L "$ROOT/artifacts" ] && mv "$ROOT/artifacts" "$AUTODL_TMP/patent-extract-agent/" || true
  [ -d "$ROOT/logs" ] && [ ! -L "$ROOT/logs" ] && mv "$ROOT/logs" "$AUTODL_TMP/patent-extract-agent/" || true
  if [ -d /root/models ] && [ ! -L /root/models ]; then
    mv /root/models "$AUTODL_TMP/patent-extract-agent/" 2>/dev/null || true
    ln -sfn "$AUTODL_TMP/patent-extract-agent/models" /root/models
  elif [ ! -e /root/models ]; then
    ln -sfn "$AUTODL_TMP/patent-extract-agent/models" /root/models
  fi
  ln -sfn "$AUTODL_TMP/patent-extract-agent/data" "$ROOT/data"
  ln -sfn "$AUTODL_TMP/patent-extract-agent/artifacts" "$ROOT/artifacts"
  ln -sfn "$AUTODL_TMP/patent-extract-agent/logs" "$ROOT/logs"
  echo "  已挂载到数据盘"
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# RTX 4090 24GB 优化参数
BATCH_STAGE1=32          # 阶段一 batch（可试 48，OOM 则降为 24）
BATCH_EP_STAGE2=8        # 阶段二每批 episode 数
NUM_WORKERS=4            # DataLoader workers（阶段一）
FP16=1                   # 混合精度

echo "=================================================="
echo "  两阶段隔离训练 (AutoDL 拉满性能)  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  BATCH_STAGE1=$BATCH_STAGE1  BATCH_EP_STAGE2=$BATCH_EP_STAGE2  FP16=$FP16"
echo "=================================================="

# 1. 终止旧进程
echo "[step 1] 终止旧训练进程..."
pkill -f 'train_span_entity_bce|train_stage2_projector_proto|run_two_stage_decoupled' 2>/dev/null || true
sleep 2
echo "  已清理"

mkdir -p "$ROOT/logs"

# 2. 运行两阶段（带性能参数）
echo ""
DS="${1:-genia}"
EXTRA_ARGS=()
[ "$FP16" = "1" ] && EXTRA_ARGS+=(--fp16)

python scripts/run_two_stage_decoupled.py \
  --dataset "$DS" \
  --build_subsets \
  --encoder roberta-base \
  --batch_stage1 "$BATCH_STAGE1" \
  --batch_ep_stage2 "$BATCH_EP_STAGE2" \
  --num_workers "$NUM_WORKERS" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$ROOT/logs/run_two_stage_decoupled.log"

echo ""
echo "[ok] 两阶段训练完成"
echo "  输出: artifacts/two_stage_decoupled/$DS"
echo "  ckpt: encoder_span_proj.pt, projector_stage2.pt"
echo "=================================================="
