"""
数据梯度+类别隔离统一实验：梯度与隔离在一起，不分开。

每次实验同时满足：
  - 梯度：仅用 n=10/100/1000 条训练样本
  - 隔离：训练仅用 meta-train 类别，评估用 meta-test 类别（类别无交集）

重要：所有实验均使用 base 预训练编码器（bert-base-cased / roberta-base / 随机初始化
BiLSTM）在限定样本上从头训练，不加载任何已训练 NER/Proto 模型 checkpoint。即：
  encoder = AutoModel.from_pretrained("bert-base-cased")  # 仅 base LM 权重
  而非 encoder = load("artifacts/run_xxx/best_model.pt")   # 无此逻辑

实验矩阵（三因素：有无原型、有无增强、有无冻结）：
  Proto 模型（fewshot 支持 isolate）→ artifacts/run_*_n{n}_isolate/
  Span 模型（supervised 不支持 isolate）→ artifacts/run_span_ner[_aug]_n{n}/

特殊：
  - chemdner 仅 1 类实体，跳过 isolate，Proto 模型退化为仅梯度 _n{n}
  - fewnerd/genia：除 n=10/100/1000+隔离外，另跑 n=10、n=100 无类别隔离 Proto（输出 _n10/_n100，与 _n10_isolate 对照）
  - B-Span/B-Span+Aug 为 supervised 模式，仅做梯度（本就无隔离）
  - SCL 已从代码中移除
  - 全数据集时仅用 B-Span（无原型），不跑 B4-Proto

用法:
  python scripts/run_gradient_isolate_unified.py --multi-gpu
  python scripts/run_gradient_isolate_unified.py --sizes 10,100 --datasets fewnerd
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ROOT / "data" / "benchmarks"
SPLIT_DIR = ROOT / "data" / "dataset" / "split"

DATASETS = ["fewnerd", "genia", "chemdner"]
DATA_SIZES = [10, 100, 1000]
# 无类别隔离的小样本 Proto 仅跑 n=10、100（与隔离实验对照）
NO_ISOLATE_PROTO_SIZES = (10, 100)

EPOCHS_SUPERVISED = 5
EPOCHS_PROTO = 8
MAX_EPISODES_PROTO = {"fewnerd": 500, "genia": 400, "chemdner": 400}
N_EVAL = 50
N_WAY_PROTO = {"fewnerd": 5, "genia": 5, "chemdner": 1}
K_SHOT = 5
MAX_LEN_PROTO = 256


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def build_label_to_spans(samples: list[dict]) -> dict[str, list]:
    lab_to: dict[str, list] = {}
    for s in samples:
        ctx = str(s.get("context") or "")
        if not ctx:
            continue
        for sp in s.get("spans") or []:
            if not isinstance(sp, dict):
                continue
            lab = str(sp.get("label") or "term").strip()
            if not lab:
                continue
            if lab not in lab_to:
                lab_to[lab] = []
            lab_to[lab].append(ctx)
    return lab_to


def split_labels_for_isolate(
    dataset: str,
    data_strategy: str,
    k_shot: int,
    train_ratio: float = 0.6,
) -> tuple[str, str] | None:
    if data_strategy == "augmented":
        path = SPLIT_DIR / f"{dataset}_train_augmented.jsonl"
    else:
        path = BENCHMARKS / dataset / "train.jsonl"
    samples = load_jsonl(path)
    if not samples:
        return None
    lab_to = build_label_to_spans(samples)
    valid = sorted(l for l, items in lab_to.items() if len(items) >= k_shot)
    if len(valid) < 4:
        return None
    n_train = max(2, int(len(valid) * train_ratio))
    n_train = min(n_train, len(valid) - 2)
    return (",".join(valid[:n_train]), ",".join(valid[n_train:]))


def run(cmd: list[str], desc: str) -> int:
    print(f"\n{'='*60}")
    print(f"[run] {desc}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*60}")
    ret = subprocess.run(cmd, cwd=str(ROOT))
    if ret.returncode != 0:
        print(f"[warn] {desc} exited with code {ret.returncode}")
    return ret.returncode


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="数据梯度+隔离统一实验")
    ap.add_argument("--datasets", type=str, default="fewnerd,genia,chemdner")
    ap.add_argument("--sizes", type=str, default="10,100,1000",
                    help="训练样本数梯度，逗号分隔")
    ap.add_argument("--models-dir", type=str, default="",
                    help="本地模型目录（如 /root/models）")
    ap.add_argument("--multi-gpu", action="store_true", default=True)
    ap.add_argument("--no-multi-gpu", action="store_false", dest="multi_gpu")
    ap.add_argument("--train-ratio", type=float, default=0.6,
                    help="meta-train 类别占比")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]

    models_dir = Path(args.models_dir) if args.models_dir else None
    encoder_bert = str(models_dir / "bert-base-cased") if models_dir else "bert-base-cased"
    encoder_roberta = str(models_dir / "roberta-base") if models_dir else "roberta-base"
    if models_dir:
        print(f"[config] BERT={encoder_bert}, RoBERTa={encoder_roberta}")

    multi = ["--multi-gpu"] if args.multi_gpu else []
    base_kw = ["--seed", "42"]
    proto_kw = ["--k_shot", str(K_SHOT), "--max-len", str(MAX_LEN_PROTO)]

    print("=" * 60)
    print("  数据梯度+隔离统一实验")
    print(f"  梯度: n={sizes}")
    print(f"  数据集: {datasets}")
    print(f"  隔离: meta-train {args.train_ratio:.0%} / meta-test {1-args.train_ratio:.0%}")
    print("=" * 60)

    # Proto experiments: gradient + isolate combined
    PROTO_EXPERIMENTS = [
        # (name, encoder_key, data_strategy, encoder_type, freeze, extra_desc)
        ("B4",    "bert",    "original",  "transformer", False, "BERT-Proto"),
        ("B4r",   "roberta", "original",  "transformer", False, "RoBERTa-Proto"),
        ("B4f",   "bert",    "original",  "transformer", True,  "BERT-Proto(冻结)"),
        ("B4rf",  "roberta", "original",  "transformer", True,  "RoBERTa-Proto(冻结)"),
        ("B5",    "bilstm",  "augmented", "bilstm",      False, "BiLSTM-Proto+Aug"),
        ("Ours",  "bert",    "augmented", "transformer", False, "BERT-Proto+Aug"),
        ("Oursr", "roberta", "augmented", "transformer", False, "RoBERTa-Proto+Aug"),
    ]

    # Span experiments: gradient only (supervised, no isolate support)
    SPAN_EXPERIMENTS = [
        # (name, data_strategy, extra_desc)
        ("BSpan",    "original",  "BERT-Span"),
        ("BSpanAug", "augmented", "BERT-Span+Aug"),
    ]

    for n in sizes:
        print(f"\n{'#'*60}")
        print(f"# n = {n}")
        print(f"{'#'*60}")

        for ds in datasets:
            max_ep = MAX_EPISODES_PROTO.get(ds, 400)
            n_way = str(N_WAY_PROTO.get(ds, 5))

            # Check data availability
            train_orig = BENCHMARKS / ds / "train.jsonl"
            if not train_orig.exists():
                print(f"[skip] {ds} train.jsonl 不存在")
                continue
            n_orig = count_lines(train_orig)
            if n_orig < n:
                print(f"[skip] {ds} original 仅 {n_orig} 条，不足 {n}")
                continue

            # Determine isolate labels (fewnerd/genia support, chemdner skips)
            split_orig = split_labels_for_isolate(ds, "original", K_SHOT, args.train_ratio)
            split_aug = split_labels_for_isolate(ds, "augmented", K_SHOT, args.train_ratio)
            can_isolate = split_orig is not None

            if can_isolate:
                train_labels_orig, test_labels_orig = split_orig
                print(f"\n[isolate] {ds} original: "
                      f"train={len(train_labels_orig.split(','))} 类, "
                      f"test={len(test_labels_orig.split(','))} 类")
                if split_aug:
                    train_labels_aug, test_labels_aug = split_aug
                    print(f"[isolate] {ds} augmented: "
                          f"train={len(train_labels_aug.split(','))} 类, "
                          f"test={len(test_labels_aug.split(','))} 类")
                else:
                    train_labels_aug = test_labels_aug = None
            else:
                print(f"\n[skip isolate] {ds}: 类别不足，仅做梯度")

            # --- Proto experiments: gradient + isolate ---
            for name, enc_key, data_strat, enc_type, freeze, desc in PROTO_EXPERIMENTS:
                if enc_key == "bert":
                    encoder = encoder_bert
                elif enc_key == "roberta":
                    encoder = encoder_roberta
                else:
                    encoder = "unused"

                # Check augmented data exists
                if data_strat == "augmented":
                    aug_path = SPLIT_DIR / f"{ds}_train_augmented.jsonl"
                    if not aug_path.exists():
                        print(f"[skip] {desc} n={n} {ds}: augmented 数据不存在")
                        continue
                    n_aug = count_lines(aug_path)
                    if n_aug < n:
                        print(f"[skip] {desc} n={n} {ds}: augmented 仅 {n_aug} 条")
                        continue

                # Build output suffix and isolate args
                if can_isolate:
                    out_suffix = f"_n{n}_isolate"
                    if data_strat == "augmented" and split_aug:
                        iso_args = ["--train-labels", train_labels_aug,
                                    "--test-labels", test_labels_aug]
                    elif data_strat == "original":
                        iso_args = ["--train-labels", train_labels_orig,
                                    "--test-labels", test_labels_orig]
                    else:
                        iso_args = []
                        out_suffix = f"_n{n}"
                else:
                    out_suffix = f"_n{n}"
                    iso_args = []

                cmd = [
                    sys.executable, "scripts/run_baseline.py",
                    "--dataset", ds,
                    "--mode", "fewshot",
                    "--data-strategy", data_strat,
                    "--encoder-type", enc_type,
                    "--encoder", encoder,
                    "--epochs", str(EPOCHS_PROTO),
                    "--max-episodes", str(max_ep),
                    "--n-eval", str(N_EVAL),
                    "--n_way", n_way,
                    "--batch-episodes", "8",
                    *proto_kw, *multi, *base_kw,
                    "--max-train-samples", str(n),
                    "--output-suffix", out_suffix,
                    *iso_args,
                ]
                if freeze:
                    cmd.append("--freeze-encoder")

                iso_tag = "+isolate" if iso_args else ""
                run(cmd, f"{desc} n={n}{iso_tag} {ds}")

            # --- Proto 无隔离：仅 n=10/100，且仅多类数据集（chemdner 已在上面用 _n{n} 跑过）---
            if n in NO_ISOLATE_PROTO_SIZES and can_isolate:
                for name, enc_key, data_strat, enc_type, freeze, desc in PROTO_EXPERIMENTS:
                    if enc_key == "bert":
                        encoder = encoder_bert
                    elif enc_key == "roberta":
                        encoder = encoder_roberta
                    else:
                        encoder = "unused"
                    if data_strat == "augmented":
                        aug_path = SPLIT_DIR / f"{ds}_train_augmented.jsonl"
                        if not aug_path.exists():
                            print(f"[skip] {desc} n={n} 无隔离 {ds}: augmented 不存在")
                            continue
                        n_aug = count_lines(aug_path)
                        if n_aug < n:
                            print(f"[skip] {desc} n={n} 无隔离 {ds}: augmented 仅 {n_aug} 条")
                            continue
                    out_suffix = f"_n{n}"
                    cmd = [
                        sys.executable, "scripts/run_baseline.py",
                        "--dataset", ds,
                        "--mode", "fewshot",
                        "--data-strategy", data_strat,
                        "--encoder-type", enc_type,
                        "--encoder", encoder,
                        "--epochs", str(EPOCHS_PROTO),
                        "--max-episodes", str(max_ep),
                        "--n-eval", str(N_EVAL),
                        "--n_way", n_way,
                        "--batch-episodes", "8",
                        *proto_kw, *multi, *base_kw,
                        "--max-train-samples", str(n),
                        "--output-suffix", out_suffix,
                    ]
                    if freeze:
                        cmd.append("--freeze-encoder")
                    run(cmd, f"{desc} n={n} 无隔离 {ds}")

            # --- Span experiments: gradient only ---
            for name, data_strat, desc in SPAN_EXPERIMENTS:
                if data_strat == "augmented":
                    aug_path = SPLIT_DIR / f"{ds}_train_augmented.jsonl"
                    if not aug_path.exists():
                        print(f"[skip] {desc} n={n} {ds}: augmented 数据不存在")
                        continue
                    n_aug = count_lines(aug_path)
                    if n_aug < n:
                        print(f"[skip] {desc} n={n} {ds}: augmented 仅 {n_aug} 条")
                        continue

                out_suffix = f"_n{n}"
                cmd = [
                    sys.executable, "scripts/run_baseline.py",
                    "--dataset", ds,
                    "--mode", "supervised",
                    "--data-strategy", data_strat,
                    "--encoder-type", "transformer",
                    "--encoder", encoder_bert,
                    "--epochs", str(EPOCHS_SUPERVISED),
                    *multi, *base_kw,
                    "--max-train-samples", str(n),
                    "--output-suffix", out_suffix,
                    "--batch-size", "24",
                    "--num-workers", "8",
                ]
                run(cmd, f"{desc} n={n} {ds}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("[汇总] 梯度+隔离统一实验 F1")
    print("=" * 60)

    proto_arts = [
        ("B4",     "run_proto_span"),
        ("B4r",    "run_proto_span_roberta"),
        ("B4f",    "run_proto_span_frozen"),
        ("B4rf",   "run_proto_span_roberta_frozen"),
        ("B5",     "run_proto_span_bilstm_aug"),
        ("Ours",   "run_proto_span_aug"),
        ("Ours-r", "run_proto_span_roberta_aug"),
    ]
    span_arts = [
        ("B-Span",     "run_span_ner"),
        ("B-Span+Aug", "run_span_ner_aug"),
    ]

    for ds in datasets:
        print(f"\n  {ds}:")
        can_iso = split_labels_for_isolate(ds, "original", K_SHOT, args.train_ratio) is not None
        for n_val in sizes:
            row = f"    n={n_val:>4}:"
            for label, base in proto_arts:
                sfx = f"_n{n_val}_isolate" if can_iso else f"_n{n_val}"
                mf = ROOT / "artifacts" / f"{base}{sfx}" / ds / "metrics.json"
                if mf.exists():
                    try:
                        m = json.loads(mf.read_text(encoding="utf-8"))
                        f1 = m.get("best_f1", m.get("f1", 0))
                        row += f"  {label}={f1:.3f}"
                    except Exception:
                        row += f"  {label}=err"
                else:
                    row += f"  {label}=----"
            for label, base in span_arts:
                mf = ROOT / "artifacts" / f"{base}_n{n_val}" / ds / "metrics.json"
                if mf.exists():
                    try:
                        m = json.loads(mf.read_text(encoding="utf-8"))
                        f1 = m.get("best_f1", m.get("f1", 0))
                        row += f"  {label}={f1:.3f}"
                    except Exception:
                        row += f"  {label}=err"
                else:
                    row += f"  {label}=----"
            print(row)

    print("\n" + "=" * 60)
    print("[汇总] Proto 无隔离（仅 n=10/100，fewnerd/genia；输出 run_*_n{n}/）")
    print("=" * 60)
    for ds in datasets:
        if not split_labels_for_isolate(ds, "original", K_SHOT, args.train_ratio):
            continue
        print(f"\n  {ds}:")
        for n_val in NO_ISOLATE_PROTO_SIZES:
            row = f"    n={n_val:>4} 无隔离:"
            for label, base in proto_arts:
                mf = ROOT / "artifacts" / f"{base}_n{n_val}" / ds / "metrics.json"
                if mf.exists():
                    try:
                        m = json.loads(mf.read_text(encoding="utf-8"))
                        f1 = m.get("best_f1", m.get("f1", 0))
                        row += f"  {label}={f1:.3f}"
                    except Exception:
                        row += f"  {label}=err"
                else:
                    row += f"  {label}=----"
            print(row)

    print("\n" + "=" * 60)
    print("[ok] 梯度+隔离统一实验完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
