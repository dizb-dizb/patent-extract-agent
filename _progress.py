import warnings; warnings.filterwarnings("ignore")
import paramiko, json, re

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.cqa1.seetacloud.com", 38815, "root", "CQtlwjJT2xIF", timeout=15)

def run(cmd):
    _, out, _ = c.exec_command(cmd, timeout=20)
    return out.read().decode(errors="replace").strip()

print("=" * 65)
print("  AutoDL 实验进度  " + run("date '+%Y-%m-%d %H:%M:%S'"))
print("=" * 65)

# GPU
print("\n[GPU]")
print(run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader"))

# 进程
print("\n[训练进程]")
procs = run("ps aux | grep -E 'train_|run_full' | grep -v grep | awk '{print $2, $3\"%\", $11, $12, $13}'")
print(procs if procs else "(无训练进程)")

# 已完成 metrics
print("\n[已完成基线]")
files = run("find /root/patent-extract-agent/artifacts -name 'metrics.json' 2>/dev/null | sort")
results = []
for f in files.split("\n"):
    f = f.strip()
    if not f:
        continue
    content = run(f"cat {f} 2>/dev/null")
    try:
        m = json.loads(content)
        path = f.replace("/root/patent-extract-agent/artifacts/", "")
        parts = path.split("/")
        baseline = parts[0] if parts else "?"
        dataset = parts[1] if len(parts) > 1 else "?"
        ep = m.get("epoch", "?")
        f1 = m.get("f1", 0)
        best = m.get("best_f1", 0)
        p = m.get("precision", 0)
        r = m.get("recall", 0)
        results.append((baseline, dataset, ep, p, r, f1, best))
        print(f"  {baseline:35s} {dataset:12s}  ep={ep:>2}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}  best={best:.4f}")
    except:
        print(f"  {f}  (解析失败)")

# 当前进度
print("\n[当前训练进度 - 最新 tqdm]")
log = run("tail -c 8000 /root/patent-extract-agent/logs/run_convergence.log 2>/dev/null")
speeds = [float(x) for x in re.findall(r'(\d+\.\d+)it/s', log) if float(x) < 1000]
# 当前 epoch
ep_matches = re.findall(r'ep(\d+):\s+(\d+)%', log)
if ep_matches:
    last_ep, last_pct = ep_matches[-1]
    speed = f"{speeds[-1]:.1f} it/s" if speeds else "?"
    print(f"  当前: ep{last_ep}  进度 {last_pct}%  速度 {speed}")
else:
    print("  (等待下一 epoch 开始)")

# 当前正在跑哪个脚本
current_script = run("ps aux | grep -E 'train_(bilstm|seq|fewshot)' | grep -v grep | awk '{print $11, $12, $13}' | head -1")
if current_script:
    print(f"  脚本: {current_script}")

# 日志最后几行有效信息
print("\n[日志最后有效行]")
log_lines = run("grep -a 'eval\\|epoch\\|info\\|phase\\|=====\\|B1\\|B2\\|B3\\|B4\\|run\\]' /root/patent-extract-agent/logs/run_convergence.log 2>/dev/null | tail -10")
for line in log_lines.split("\n"):
    if line.strip():
        print(f"  {line.strip()[:100]}")

# 预估进度
print("\n[实验矩阵完成情况]")
baselines = [
    # 主实验（全数据集，无原型网络/无 SCL）
    ("run_bilstm_crf",             "B1 BiLSTM-CRF"),
    ("run_seq_ner",                "B2 BERT-CRF"),
    ("run_seq_ner_roberta",        "B2r RoBERTa-CRF"),
    ("run_proto_span_bilstm",      "B3 BiLSTM-Proto"),
    ("run_proto_span",             "B4 BERT-Proto"),
    ("run_span_ner",               "B-Span BERT-Span"),
    ("run_span_ner_aug",           "B-Span+Aug"),
]
# 梯度+隔离统一实验（n=10,100,1000, proto→_isolate, span→仅梯度）
gradient_isolate = []
for n in [10, 100, 1000]:
    for base, label in [
        ("run_proto_span",                f"B4 n={n}"),
        ("run_proto_span_roberta",        f"B4r n={n}"),
        ("run_proto_span_frozen",         f"B4f n={n}"),
        ("run_proto_span_roberta_frozen", f"B4rf n={n}"),
        ("run_proto_span_bilstm_aug",     f"B5 n={n}"),
        ("run_proto_span_aug",            f"Ours n={n}"),
        ("run_proto_span_roberta_aug",    f"Ours-r n={n}"),
    ]:
        gradient_isolate.append((f"{base}_n{n}_isolate", f"{label}+iso"))
    for base, label in [
        ("run_span_ner",     f"B-Span n={n}"),
        ("run_span_ner_aug", f"B-Span+Aug n={n}"),
    ]:
        gradient_isolate.append((f"{base}_n{n}", label))
# 原型网络 n=10/100 无类别隔离（仅 fewnerd、genia 有独立目录；chemdner 仍看上行 +iso 同 n）
gradient_no_isolate = []
for n in [10, 100]:
    for base, short in [
        ("run_proto_span",                "B4"),
        ("run_proto_span_roberta",        "B4r"),
        ("run_proto_span_frozen",         "B4f"),
        ("run_proto_span_roberta_frozen", "B4rf"),
        ("run_proto_span_bilstm_aug",     "B5"),
        ("run_proto_span_aug",            "Ours"),
        ("run_proto_span_roberta_aug",    "Ours-r"),
    ]:
        gradient_no_isolate.append((f"{base}_n{n}", f"{short} n={n} 无隔"))
datasets = ["fewnerd", "genia", "chemdner"]
done = 0
total = len(baselines) * len(datasets)
for dir_name, label in baselines:
    row = f"  {label:40s}"
    for ds in datasets:
        mf = f"/root/patent-extract-agent/artifacts/{dir_name}/{ds}/metrics.json"
        content = run(f"cat {mf} 2>/dev/null")
        if content:
            try:
                m = json.loads(content)
                f1 = m.get("best_f1", m.get("f1", 0))
                row += f"  {ds[:5]}={f1:.3f}"
                done += 1
            except:
                row += f"  {ds[:5]}=err"
        else:
            row += f"  {ds[:5]}=----"
    print(row)

# 梯度+隔离统一实验（含原型网络 B4/B4r/B4f/B4rf/B5/Ours/Ours-r + 无原型对照 B-Span/B-Span+Aug）
print("\n[梯度+隔离统一实验]  (原型网络: B4/B4r/B4f/B4rf/B5/Ours/Ours-r | 无原型: B-Span/B-Span+Aug)")
gi_done = 0
gi_total = 0
for dir_name, label in gradient_isolate:
    row = f"  {label:40s}"
    for ds in datasets:
        gi_total += 1
        # Proto isolate: chemdner 没有 _isolate, 退化为 _n{n}
        actual_dir = dir_name
        if ds == "chemdner" and "_isolate" in dir_name:
            actual_dir = dir_name.replace("_isolate", "")
        mf = f"/root/patent-extract-agent/artifacts/{actual_dir}/{ds}/metrics.json"
        content = run(f"cat {mf} 2>/dev/null")
        if content:
            try:
                m = json.loads(content)
                f1 = m.get("best_f1", m.get("f1", 0))
                row += f"  {ds[:5]}={f1:.3f}"
                gi_done += 1
            except:
                row += f"  {ds[:5]}=err"
        else:
            row += f"  {ds[:5]}=----"
    print(row)

gni_done = 0
gni_total = 0
print("\n[原型 无隔离 n=10/100]  (fewnerd / genia；与上表 _isolate 对照)")
for dir_name, label in gradient_no_isolate:
    row = f"  {label:40s}"
    for ds in ["fewnerd", "genia"]:
        gni_total += 1
        mf = f"/root/patent-extract-agent/artifacts/{dir_name}/{ds}/metrics.json"
        content = run(f"cat {mf} 2>/dev/null")
        if content:
            try:
                m = json.loads(content)
                f1 = m.get("best_f1", m.get("f1", 0))
                row += f"  {ds[:5]}={f1:.3f}"
                gni_done += 1
            except Exception:
                row += f"  {ds[:5]}=err"
        else:
            row += f"  {ds[:5]}=----"
    print(row)

total += gi_total + gni_total
done += gi_done + gni_done
print(f"\n  总进度: {done}/{total} ({100*done//total if total else 0}%)")
print("=" * 65)
c.close()
