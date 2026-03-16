#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base 2>/dev/null

echo "=== GPU ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo ""
echo "=== 训练进程 ==="
ps aux | grep -E "train_(bilstm_crf|seq_ner|fewshot_proto)" | grep -v grep | awk '{print $11, $12, $13}' || echo "(无训练进程)"

echo ""
echo "=== 各任务进度 ==="
PROJ="/root/patent-extract-agent"
for name in B1_fewnerd B2_fewnerd B3_fewnerd B4_fewnerd B1_genia B2_genia B3_genia B4_genia B1_chemdner B2_chemdner B3_chemdner B4_chemdner; do
    LOG="$PROJ/logs/${name}.log"
    [ -f "$LOG" ] || continue
    PROGRESS=$(tail -3 "$LOG" 2>/dev/null | grep -oP 'ep\d+:\s+\d+%\|[^|]+\|\s+[\d/]+' | tail -1)
    LOSS=$(tail -3 "$LOG" 2>/dev/null | grep -oP 'loss=[\d.]+' | tail -1)
    [ -n "$PROGRESS" ] && echo "  $name: $PROGRESS $LOSS" || true
done

echo ""
echo "=== metrics.json (有结果的任务) ==="
for dir in \
    "$PROJ/artifacts/run_bilstm_crf/fewnerd" \
    "$PROJ/artifacts/run_bilstm_crf/genia" \
    "$PROJ/artifacts/run_bilstm_crf/chemdner" \
    "$PROJ/artifacts/run_seq_ner/fewnerd" \
    "$PROJ/artifacts/run_seq_ner/genia" \
    "$PROJ/artifacts/run_seq_ner/chemdner" \
    "$PROJ/artifacts/run_proto_span/fewnerd" \
    "$PROJ/artifacts/run_proto_span/genia" \
    "$PROJ/artifacts/run_proto_span/chemdner" \
    "$PROJ/artifacts/run_proto_span_bilstm/fewnerd" \
    "$PROJ/artifacts/run_proto_span_bilstm/genia" \
    "$PROJ/artifacts/run_proto_span_bilstm/chemdner"; do
    f="$dir/metrics.json"
    [ -f "$f" ] || continue
    label=$(echo "$dir" | sed 's|.*/artifacts/||;s|/| |')
    python -c "
import json, sys
m = json.load(open('$f'))
name = m.get('name','?')
ep   = m.get('epoch','?')
f1   = m.get('f1', 0)
ff   = m.get('flat_f1', f1)
enc  = m.get('encoder_type', 'transformer')
print(f'  {\"$label\":<35}  ep={ep}  F1={f1:.4f}  FlatF1={ff:.4f}  enc={enc}')
" 2>/dev/null
done

echo ""
echo "=== run_fast 主进程日志 (最后 10 行) ==="
tail -10 "$PROJ/logs/run_fast_main.log" 2>/dev/null || echo "(no log)"
