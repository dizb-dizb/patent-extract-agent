"""
消融实验调度：遍历消融矩阵，调用 run_baseline，收集 metrics 到 artifacts/ablations/summary.json。

消融维度：模型(BERT/RoBERTa)、数据量(1-shot/5-shot/全量)、数据集
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ABLATIONS_DIR = ROOT / "artifacts" / "ablations"
BENCHMARKS = ROOT / "data" / "benchmarks"
UNIFIED_DIR = ROOT / "data" / "dataset" / "unified"
SPLIT_DIR = ROOT / "data" / "dataset" / "split"

# 消融矩阵
DATASETS = ["fewnerd", "genia", "chemdner"]
ENCODERS = {
    "bert": "bert-base-cased",
    "roberta": "bert-base-cased",  # 英文用 bert-base-cased，中文用 roberta
}
DATASET_ENCODERS = {
    "fewnerd": ["bert-base-cased"],
    "genia": ["bert-base-cased"],
    "chemdner": ["bert-base-cased"],
    "unified": ["bert-base-cased"],
}


def run_one(dataset: str, mode: str, encoder: str, n_way: int, k_shot: int, data_strategy: str = "original") -> dict | None:
    """运行单次实验，返回 metrics 或 None。"""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_baseline.py"),
        "--dataset", dataset,
        "--mode", mode,
        "--encoder", encoder,
        "--data-strategy", data_strategy,
    ]
    if mode == "fewshot":
        cmd.extend(["--n_way", str(n_way), "--k_shot", str(k_shot)])
    try:
        subprocess.run(cmd, cwd=str(ROOT), check=True, capture_output=True, timeout=7200)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    if mode == "supervised":
        out_dir = ROOT / "artifacts" / "run_span_ner" / dataset
    elif mode == "seq":
        out_dir = ROOT / "artifacts" / "run_seq_ner" / dataset
    else:
        out_dir = ROOT / "artifacts" / "run_proto_span" / dataset
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    data["_run"] = {
        "dataset": dataset,
        "mode": mode,
        "encoder": encoder,
        "n_way": n_way,
        "k_shot": k_shot,
        "data_strategy": data_strategy,
    }
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default="fewnerd,genia,chemdner",
                    help="Comma-separated: fewnerd,genia,chemdner,unified")
    ap.add_argument("--modes", type=str, default="supervised,fewshot",
                    help="Comma-separated: supervised,fewshot,seq")
    ap.add_argument("--data-strategy", type=str, default="original", choices=["original", "augmented"],
                    help="original=基准数据; augmented=Agent增强 (Ours)")
    ap.add_argument("--k_shots", type=str, default="1,5")
    ap.add_argument("--n_way", type=int, default=5)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    k_shots = [int(k) for k in args.k_shots.split(",") if k.strip().isdigit()]

    ABLATIONS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for dataset in datasets:
        if dataset == "unified":
            data_dir = UNIFIED_DIR / "train.jsonl"
        elif args.data_strategy == "augmented":
            data_dir = SPLIT_DIR / f"{dataset}_train_augmented.jsonl"
        else:
            data_dir = BENCHMARKS / dataset / "train.jsonl"
        if not data_dir.exists():
            print(f"[skip] {dataset} (no data: {data_dir})")
            continue
        encoders = DATASET_ENCODERS.get(dataset, ["bert-base-cased"])
        for mode in modes:
            if mode == "supervised":
                for enc in encoders:
                    if args.dry_run:
                        print(f"[dry] {dataset} supervised {enc} {args.data_strategy}")
                        continue
                    r = run_one(dataset, mode, enc, args.n_way, 0, args.data_strategy)
                    if r:
                        results.append(r)
                        print(f"[ok] {dataset} supervised {enc} F1={r.get('f1', 0):.4f}")
            elif mode == "seq":
                for enc in encoders:
                    if args.dry_run:
                        print(f"[dry] {dataset} seq {enc} {args.data_strategy}")
                        continue
                    r = run_one(dataset, mode, enc, args.n_way, 0, args.data_strategy)
                    if r:
                        results.append(r)
                        print(f"[ok] {dataset} seq {enc} F1={r.get('f1', 0):.4f}")
            else:
                for k in k_shots:
                    for enc in encoders:
                        if args.dry_run:
                            print(f"[dry] {dataset} fewshot {args.n_way}w{k}k {enc} {args.data_strategy}")
                            continue
                        r = run_one(dataset, mode, enc, args.n_way, k, args.data_strategy)
                        if r:
                            results.append(r)
                            print(f"[ok] {dataset} fewshot {args.n_way}w{k}k {enc} F1={r.get('f1', 0):.4f}")
                        time.sleep(2)

    summary_path = ABLATIONS_DIR / "summary.json"
    summary = {
        "runs": results,
        "created_at": int(time.time()),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] summary: {summary_path} ({len(results)} runs)")


if __name__ == "__main__":
    main()
