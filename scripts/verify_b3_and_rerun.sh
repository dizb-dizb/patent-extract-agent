#!/usr/bin/env bash
# =============================================================================
#  验证 B3 数据路径，若用了 augmented 则补跑 original
#  用法: bash scripts/verify_b3_and_rerun.sh [fewnerd|genia|chemdner]
#  不传参数则检查全部数据集
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PY=$(command -v python 2>/dev/null || command -v python3 2>/dev/null || true)
[ -z "$PY" ] && PY="python"

ENCODER="${ENCODER:-/root/models/bert-base-cased}"
[ ! -d "$ENCODER" ] && ENCODER="bert-base-cased"

check_and_rerun() {
    local ds=$1
    local cfg="$ROOT/artifacts/run_proto_span_bilstm/$ds/config.json"
    local wrong=0

    if [ -f "$cfg" ]; then
        local data_path
        data_path=$($PY -c "import json; m=json.load(open('$cfg')); print(m.get('data',''))" 2>/dev/null || echo "")
        if [[ "$data_path" == *"augmented"* ]] || [[ "$data_path" == *"_augmented"* ]]; then
            echo "[$ds] B3 数据路径错误: $data_path (应为 original)"
            wrong=1
        else
            echo "[$ds] B3 数据路径正确: $data_path"
        fi
    else
        echo "[$ds] 无 config.json，跳过"
        return 0
    fi

    if [ "$wrong" -eq 1 ]; then
        echo "[$ds] 补跑 B3 (original 数据)..."
        local ep max_ep
        case "$ds" in
            fewnerd)  ep=8; max_ep=1000; nw=5 ;;
            genia)    ep=8; max_ep=800;  nw=5 ;;
            chemdner) ep=8; max_ep=800;  nw=1 ;;
            *)        ep=8; max_ep=800;  nw=5 ;;
        esac
        $PY train_fewshot_proto_span.py \
            --data "data/benchmarks/$ds/train.jsonl" \
            --val "data/benchmarks/$ds/val.jsonl" \
            --output_dir "artifacts/run_proto_span_bilstm/$ds" \
            --encoder_type bilstm \
            --n_way $nw --k_shot 5 \
            --epochs $ep --max_episodes $max_ep --n_eval 80 \
            --max_len 256 \
            --encoder "$ENCODER" \
            --seed 42
        echo "[$ds] 补跑完成"
    fi
}

DATASETS="${1:-fewnerd genia chemdner}"
for ds in $DATASETS; do
    check_and_rerun "$ds"
done
echo "[ok] 验证完成"
