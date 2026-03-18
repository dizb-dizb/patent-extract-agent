#!/bin/bash
# =====================================================================
#  专利术语小样本NER 实验实时监控  v2
#  用法: bash /root/patent-extract-agent/_monitor.sh
#  每 20 秒自动刷新，Ctrl+C 退出
# =====================================================================
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate base 2>/dev/null
PY=$(which python 2>/dev/null || echo "python3")
PROJ="/root/patent-extract-agent"
LOG="$PROJ/logs"
INTERVAL=20

# ANSI 颜色
R='\033[0m'       # reset
BOLD='\033[1m'
GRN='\033[32m'
YLW='\033[33m'
RED='\033[31m'
CYN='\033[36m'
DIM='\033[2m'

bar() {   # bar <pct 0-100> <width>
    local p=$1 w=${2:-20}
    local filled=$(( w * p / 100 ))
    local empty=$(( w - filled ))
    printf "${GRN}"
    printf '%0.s█' $(seq 1 $filled 2>/dev/null) 2>/dev/null
    printf "${DIM}"
    printf '%0.s░' $(seq 1 $empty 2>/dev/null) 2>/dev/null
    printf "${R}"
}

read_f1() {   # read_f1 <metrics.json path> <field: f1|flat_f1|bwt>
    local f=$1 k=${2:-f1}
    [ -f "$f" ] || { echo "—"; return; }
    $PY -c "
import json,sys
try:
    m=json.load(open('$f'))
    v=m.get('$k',None)
    if v is None: print('—')
    elif '$k'=='bwt': print(f'{float(v):+.4f}')
    else: print(f'{float(v):.4f}')
except: print('err')
" 2>/dev/null || echo "err"
}

read_ep() {
    local f=$1
    [ -f "$f" ] || { echo "?"; return; }
    $PY -c "import json; m=json.load(open('$f')); print(m.get('epoch','?'))" 2>/dev/null || echo "?"
}

# 从日志里抓最新的 tqdm 进度行
progress_from_log() {
    local logf=$1
    [ -f "$logf" ] || { echo "(no log)"; return; }
    # 取最后一个包含进度的行
    local line
    line=$(tail -5 "$logf" 2>/dev/null | grep -oP 'ep\d+:\s+\d+%\|[^[]+\[\d+:\d+<\d+:\d+.*?it/s\]?' | tail -1)
    [ -n "$line" ] && echo "$line" || echo "—"
}

loss_from_log() {
    local logf=$1
    [ -f "$logf" ] || { echo ""; return; }
    tail -3 "$logf" 2>/dev/null | grep -oP 'loss=[\d.]+' | tail -1
}

# 进程状态
proc_alive() {
    local pidfile="$LOG/$1.pid"
    [ -f "$pidfile" ] || return 1
    local pid; pid=$(cat "$pidfile")
    kill -0 "$pid" 2>/dev/null
}

span_ner_running() {
    pgrep -f "train_span_ner" >/dev/null 2>&1
}

status_icon() {   # status_icon <log_name> <metrics_file>
    local name=$1 mfile=$2
    if proc_alive "$name"; then
        printf "${GRN}● RUN${R}"
    elif [ -f "$mfile" ]; then
        printf "${YLW}✓ DONE${R}"
    elif [[ "$name" == BSpan_* || "$name" == BSpanAug_* ]] && span_ner_running; then
        printf "${GRN}● RUN${R}"
    else
        printf "${DIM}○ WAIT${R}"
    fi
}

# 一行打印实验结果
result_row() {
    # $1=display_name $2=log_name $3=metrics_dir $4=n_epochs_expected $5=log_override(可选)
    local dname=$1 lname=$2 mdir=$3 epx=${4:-3} log_override=${5:-}
    local mfile="$mdir/metrics.json"
    local logf="$LOG/${lname}.log"
    [ -n "$log_override" ] && logf="$LOG/${log_override}.log"
    local f1; f1=$(read_f1 "$mfile" f1)
    local ff; ff=$(read_f1 "$mfile" flat_f1)
    local ep; ep=$(read_ep "$mfile")
    local icon; icon=$(status_icon "$lname" "$mfile")
    local prog; prog=$(progress_from_log "$logf")
    local loss; loss=$(loss_from_log "$logf")

    printf "  %-22s  %-9s  ep:%-4s  F1:%-8s FlatF1:%-8s  %s\n" \
        "$dname" "$icon" "${ep}/${epx}" "$f1" "$ff" "$loss"
    [ -n "$prog" ] && printf "    ${DIM}%s${R}\n" "$prog"
}

# ─────────────────────────────────────────────────────────────────────
while true; do
    printf '\033[2J\033[H'   # clear screen

    NOW=$(date '+%Y-%m-%d %H:%M:%S')
    printf "${BOLD}╔═══════════════════════════════════════════════════════════════════╗${R}\n"
    printf "${BOLD}║  专利术语小样本NER 实验监控   %-36s║${R}\n" "$NOW"
    printf "${BOLD}╚═══════════════════════════════════════════════════════════════════╝${R}\n"

    # ── GPU ─────────────────────────────────────────────────────────────────
    printf "\n${CYN}▶ GPU 状态${R}\n"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read idx name util mem_used mem_total temp; do
        idx=$(echo $idx|xargs); util=$(echo $util|xargs)
        mu=$(echo $mem_used|xargs); mt=$(echo $mem_total|xargs); t=$(echo $temp|xargs)
        name=$(echo $name|xargs|cut -c1-20)
        printf "  GPU%-1s %-21s " "$idx" "$name"
        bar "$util" 18
        printf " %3s%%  %5sMB/%-5sMB  %s°C\n" "$util" "$mu" "$mt" "$t"
    done

    # ── 进程统计 ────────────────────────────────────────────────────────────
    NPROC=$(pgrep -c -f "train_bilstm_crf\|train_seq_ner\|train_fewshot_proto_span\|train_span_ner" 2>/dev/null || echo "0")
    NPROC=$(echo "$NPROC" | head -1 | tr -cd '0-9')
    NPROC=${NPROC:-0}
    printf "\n${CYN}▶ 训练进程: ${R}"
    if [ "$NPROC" -gt 0 ] 2>/dev/null; then
        printf "${GRN}${NPROC} 个运行中${R}\n"
    else
        printf "${YLW}0 个（全部完成或等待）${R}\n"
    fi

    # ── B1: BiLSTM-CRF ──────────────────────────────────────────────────────
    printf "\n${CYN}▶ B1  BiLSTM-CRF (BIO 序列标注)${R}\n"
    result_row "B1 fewnerd"  "B1_fewnerd"       "$PROJ/artifacts/run_bilstm_crf/fewnerd"  3
    result_row "B1 genia"    "B1_genia_real"     "$PROJ/artifacts/run_bilstm_crf/genia"    5
    result_row "B1 chemdner" "B1_chemdner_real"  "$PROJ/artifacts/run_bilstm_crf/chemdner" 5

    # ── B2: BERT-CRF ────────────────────────────────────────────────────────
    printf "\n${CYN}▶ B2  BERT-CRF (BIO序列标注, multi-GPU, bert-base-cased)${R}\n"
    result_row "B2 fewnerd"  "B2_fewnerd"       "$PROJ/artifacts/run_seq_ner/fewnerd"  5
    result_row "B2 genia"    "B2_genia_real"     "$PROJ/artifacts/run_seq_ner/genia"    5
    result_row "B2 chemdner" "B2_chemdner_real"  "$PROJ/artifacts/run_seq_ner/chemdner" 5

    # ── B2r: RoBERTa-CRF ────────────────────────────────────────────────────
    printf "\n${CYN}▶ B2r RoBERTa-CRF (BIO序列标注, multi-GPU, roberta-base)${R}\n"
    result_row "B2r fewnerd"  "B2r_fewnerd"       "$PROJ/artifacts/run_seq_ner_roberta/fewnerd"  5
    result_row "B2r genia"    "B2r_genia"         "$PROJ/artifacts/run_seq_ner_roberta/genia"    5
    result_row "B2r chemdner" "B2r_chemdner"      "$PROJ/artifacts/run_seq_ner_roberta/chemdner" 5

    # ── B-Span: BERT Span 分类器（无原型网络，有监督全量对照）────────────────
    printf "\n${CYN}▶ B-Span  BERT Span分类器（无原型网络，有监督全量训练）${R}\n"
    result_row "B-Span fewnerd"  "BSpan_fewnerd"  "$PROJ/artifacts/run_span_ner/fewnerd"  5 "supplementary"
    result_row "B-Span genia"    "BSpan_genia"    "$PROJ/artifacts/run_span_ner/genia"    5 "supplementary"
    result_row "B-Span chemdner" "BSpan_chemdner" "$PROJ/artifacts/run_span_ner/chemdner" 5 "supplementary"

    # ── B3: BiLSTM-Proto ────────────────────────────────────────────────────
    printf "\n${CYN}▶ B3  BiLSTM Proto-Span (original data)${R}\n"
    result_row "B3 fewnerd"  "B3_fewnerd"       "$PROJ/artifacts/run_proto_span_bilstm/fewnerd"  3
    result_row "B3 genia"    "B3_genia_real"     "$PROJ/artifacts/run_proto_span_bilstm/genia"    5
    result_row "B3 chemdner" "B3_chemdner_real"  "$PROJ/artifacts/run_proto_span_bilstm/chemdner" 5

    # ── B4: BERT-Proto ──────────────────────────────────────────────────────
    printf "\n${CYN}▶ B4  BERT Proto-Span (original data, multi-GPU, bert-base-cased)${R}\n"
    result_row "B4 fewnerd"  "B4_fewnerd"       "$PROJ/artifacts/run_proto_span/fewnerd"  8
    result_row "B4 genia"    "B4_genia_real"     "$PROJ/artifacts/run_proto_span/genia"    8
    result_row "B4 chemdner" "B4_chemdner_real"  "$PROJ/artifacts/run_proto_span/chemdner" 8

    # ── B4r: RoBERTa-Proto ──────────────────────────────────────────────────
    printf "\n${CYN}▶ B4r RoBERTa Proto-Span (original data, multi-GPU, roberta-base)${R}\n"
    result_row "B4r fewnerd"  "B4r_fewnerd"       "$PROJ/artifacts/run_proto_span_roberta/fewnerd"  8
    result_row "B4r genia"    "B4r_genia"         "$PROJ/artifacts/run_proto_span_roberta/genia"    8
    result_row "B4r chemdner" "B4r_chemdner"      "$PROJ/artifacts/run_proto_span_roberta/chemdner" 8

    # ── B4f: BERT-Proto + encoder 冻结（未微调+原型网络）──────────────────────
    printf "\n${CYN}▶ B4f  BERT Proto-Span (encoder 冻结, 未微调模型+原型)${R}\n"
    result_row "B4f fewnerd"  "B4f_fewnerd"       "$PROJ/artifacts/run_proto_span_frozen/fewnerd"  8
    result_row "B4f genia"    "B4f_genia"         "$PROJ/artifacts/run_proto_span_frozen/genia"    8
    result_row "B4f chemdner" "B4f_chemdner"      "$PROJ/artifacts/run_proto_span_frozen/chemdner" 8

    # ── B4rf: RoBERTa-Proto + encoder 冻结（未微调+原型网络）───────────────────
    printf "\n${CYN}▶ B4rf RoBERTa Proto-Span (encoder 冻结, 未微调模型+原型)${R}\n"
    result_row "B4rf fewnerd"  "B4rf_fewnerd"     "$PROJ/artifacts/run_proto_span_roberta_frozen/fewnerd"  8
    result_row "B4rf genia"    "B4rf_genia"       "$PROJ/artifacts/run_proto_span_roberta_frozen/genia"    8
    result_row "B4rf chemdner" "B4rf_chemdner"    "$PROJ/artifacts/run_proto_span_roberta_frozen/chemdner" 8

    # ── B5: BiLSTM-Proto + augmented ────────────────────────────────────────
    printf "\n${CYN}▶ B5  BiLSTM Proto-Span + 增强数据 (反向对照)${R}\n"
    result_row "B5 fewnerd"  "B5_fewnerd"   "$PROJ/artifacts/run_proto_span_bilstm_aug/fewnerd"  3
    result_row "B5 genia"    "B5_genia"     "$PROJ/artifacts/run_proto_span_bilstm_aug/genia"    3
    result_row "B5 chemdner" "B5_chemdner"  "$PROJ/artifacts/run_proto_span_bilstm_aug/chemdner" 3

    # ── B-Span+Aug: BERT Span 无原型 + 增强数据（对照：证明原型网络优势）────
    printf "\n${CYN}▶ B-Span+Aug  BERT Span(无原型) + 增强数据${R}\n"
    result_row "B-Span+Aug fewnerd"  "BSpanAug_fewnerd"  "$PROJ/artifacts/run_span_ner_aug/fewnerd"  5
    result_row "B-Span+Aug genia"    "BSpanAug_genia"    "$PROJ/artifacts/run_span_ner_aug/genia"    5
    result_row "B-Span+Aug chemdner" "BSpanAug_chemdner" "$PROJ/artifacts/run_span_ner_aug/chemdner" 5

    # ── Ours: BERT-Proto + augmented ─────────────────────────────────────────
    printf "\n${CYN}▶ Ours  BERT-Proto + 增强数据 (核心方法-BERT, multi-GPU)${R}\n"
    result_row "Ours fewnerd"  "Ours_fewnerd"   "$PROJ/artifacts/run_proto_span_aug/fewnerd"  8
    result_row "Ours genia"    "Ours_genia"     "$PROJ/artifacts/run_proto_span_aug/genia"    8
    result_row "Ours chemdner" "Ours_chemdner"  "$PROJ/artifacts/run_proto_span_aug/chemdner" 8

    # ── Ours-r: RoBERTa-Proto + augmented ─────────────────────────────────────
    printf "\n${CYN}▶ Ours-r  RoBERTa-Proto + 增强数据 (核心方法-RoBERTa, multi-GPU)${R}\n"
    result_row "Ours-r fewnerd"  "Oursr_fewnerd"   "$PROJ/artifacts/run_proto_span_roberta_aug/fewnerd"  8
    result_row "Ours-r genia"    "Oursr_genia"     "$PROJ/artifacts/run_proto_span_roberta_aug/genia"    8
    result_row "Ours-r chemdner" "Oursr_chemdner"  "$PROJ/artifacts/run_proto_span_roberta_aug/chemdner" 8

    # ── 战役三: BWT ──────────────────────────────────────────────────────────
    printf "\n${CYN}▶ 战役三  BWT 持续学习（遗忘率）${R}\n"
    BWT_F="$PROJ/artifacts/continual/metrics.json"
    if [ -f "$BWT_F" ]; then
        $PY -c "
import json
m=json.load(open('$BWT_F'))
t1a1=m.get('f1_t1_after_t1',0); t1a2=m.get('f1_t1_after_t2',0)
t2a2=m.get('f1_t2_after_t2',0); bwt=m.get('bwt',0)
rh=m.get('rehearsal_ratio',0)
print(f'  chemdner after T1: {t1a1:.4f}  |  after T2: {t1a2:.4f}  |  genia: {t2a2:.4f}')
color='\033[32m' if bwt>=-0.02 else '\033[31m'
print(f'  BWT = {color}{bwt:+.4f}\033[0m  (rehearsal={rh})')
" 2>/dev/null
    else
        printf "  ${DIM}(等待 B1/B2 fewnerd 完成后自动执行)${R}\n"
    fi

    # ── 战役四: OOD ──────────────────────────────────────────────────────────
    printf "\n${CYN}▶ 战役四  Zero/One-shot OOD 泛化${R}\n"
    OOD_F="$PROJ/artifacts/ood_oneshot/metrics.json"
    if [ -f "$OOD_F" ]; then
        $PY -c "
import json
m=json.load(open('$OOD_F'))
p,r,f=m.get('precision',0),m.get('recall',0),m.get('f1',0)
k=m.get('k_shot','?')
td=m.get('test_data','').split('/')[-2] if '/' in m.get('test_data','') else '?'
print(f'  {k}-shot  fewnerd → {td}:  P={p:.4f}  R={r:.4f}  F1={f:.4f}')
" 2>/dev/null
    else
        printf "  ${DIM}(等待主实验完成后自动执行)${R}\n"
    fi

    # ── 实验矩阵快速汇总 ─────────────────────────────────────────────────────
    printf "\n${CYN}▶ 实验矩阵  F1 快速对比${R}\n"
    printf "  ${BOLD}%-12s  %8s  %8s  %8s${R}\n" "Baseline" "fewnerd" "genia" "chemdner"
    printf "  %s\n" "──────────────────────────────────────────"
    for PAIR in \
        "B1(BiLSTM-CRF):run_bilstm_crf" \
        "B2(BERT-CRF):run_seq_ner" \
        "B2r(RoBERTa-CRF):run_seq_ner_roberta" \
        "B-Span(BERT-Span,NoProto):run_span_ner" \
        "B3(BiLSTM-Proto):run_proto_span_bilstm" \
        "B4(BERT-Proto):run_proto_span" \
        "B4r(RoBERTa-Proto):run_proto_span_roberta" \
        "B4f(BERT-Proto,frozen):run_proto_span_frozen" \
        "B4rf(RoBERTa-Proto,frozen):run_proto_span_roberta_frozen" \
        "B5(BiLSTM+Aug):run_proto_span_bilstm_aug" \
        "B-Span+Aug(BERT-Span,NoProto+Aug):run_span_ner_aug" \
        "Ours(BERT-Proto+Aug):run_proto_span_aug" \
        "Ours-r(RoBERTa-Proto+Aug):run_proto_span_roberta_aug"; do
        LABEL="${PAIR%%:*}"
        DIR_BASE="${PAIR##*:}"
        FN=$(read_f1 "$PROJ/artifacts/$DIR_BASE/fewnerd/metrics.json" f1)
        GN=$(read_f1 "$PROJ/artifacts/$DIR_BASE/genia/metrics.json" f1)
        CN=$(read_f1 "$PROJ/artifacts/$DIR_BASE/chemdner/metrics.json" f1)
        printf "  %-28s  %8s  %8s  %8s\n" "$LABEL" "$FN" "$GN" "$CN"
    done

    # ── 页脚 ─────────────────────────────────────────────────────────────────
    printf "\n${DIM}  ↻ 每 ${INTERVAL}s 自动刷新  |  Ctrl+C 退出${R}\n"
    sleep $INTERVAL
done
