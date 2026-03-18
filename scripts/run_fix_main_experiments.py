"""
修复主实验缺失/错误数据问题：

1. B4 genia + chemdner：原先误用 augmented 数据，应用 original 重跑
2. B3 chemdner：原先误用 augmented 数据，应用 original 重跑
3. B3 genia：仅跑了 ep=2，需完整 8 epoch 重跑
4. B2r chemdner：从未执行
5. B-Span+Aug：3 个数据集从未执行

用法:
  python scripts/run_fix_main_experiments.py --multi-gpu
  python scripts/run_fix_main_experiments.py --models-dir /root/models --multi-gpu
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EPOCHS_SEQ = 5
EPOCHS_SPAN = 5
EPOCHS_PROTO = 8
MAX_EPISODES_PROTO = {"fewnerd": 1000, "genia": 800, "chemdner": 800}
N_EVAL = 80
N_WAY_PROTO = {"fewnerd": 5, "genia": 5, "chemdner": 1}
MAX_LEN_PROTO = 256
DATASETS = ["fewnerd", "genia", "chemdner"]

results: list[tuple[str, int]] = []


def run(cmd: list[str], desc: str) -> int:
    print(f"\n{'='*60}")
    print(f"[run] {desc}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*60}")
    ret = subprocess.run(cmd, cwd=str(ROOT))
    results.append((desc, ret.returncode))
    if ret.returncode != 0:
        print(f"[warn] {desc} exited with code {ret.returncode}")
    return ret.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=str, default="")
    ap.add_argument("--multi-gpu", action="store_true", default=True)
    ap.add_argument("--no-multi-gpu", action="store_false", dest="multi_gpu")
    args = ap.parse_args()

    models_dir = Path(args.models_dir) if args.models_dir else None
    encoder_bert = str(models_dir / "bert-base-cased") if models_dir else "bert-base-cased"
    encoder_roberta = str(models_dir / "roberta-base") if models_dir else "roberta-base"
    if models_dir:
        print(f"[config] BERT={encoder_bert}, RoBERTa={encoder_roberta}")

    multi = ["--multi-gpu"] if args.multi_gpu else []
    base_kw = ["--seed", "42"]
    proto_kw = ["--max-len", str(MAX_LEN_PROTO)]

    print("=" * 60)
    print("  主实验修复：缺失/错误数据重跑")
    print("=" * 60)

    # ── Fix 1: B4 genia + chemdner 用 ORIGINAL 数据重跑 ────────
    for ds in ["genia", "chemdner"]:
        n_way = str(N_WAY_PROTO.get(ds, 5))
        max_ep = MAX_EPISODES_PROTO.get(ds, 800)
        run([
            sys.executable, "scripts/run_baseline.py",
            "--dataset", ds, "--mode", "fewshot",
            "--data-strategy", "original",
            "--encoder-type", "transformer",
            "--encoder", encoder_bert,
            "--epochs", str(EPOCHS_PROTO),
            "--max-episodes", str(max_ep),
            "--n-eval", str(N_EVAL),
            "--n_way", n_way,
            *proto_kw, *multi, *base_kw,
        ], f"B4 BERT-Proto(original) {ds} [修复:之前误用augmented]")

    # ── Fix 2: B3 chemdner 用 ORIGINAL 数据重跑 ────────
    n_way = str(N_WAY_PROTO.get("chemdner", 1))
    max_ep = MAX_EPISODES_PROTO.get("chemdner", 800)
    run([
        sys.executable, "scripts/run_baseline.py",
        "--dataset", "chemdner", "--mode", "fewshot",
        "--data-strategy", "original",
        "--encoder-type", "bilstm",
        "--encoder", "unused",
        "--epochs", str(EPOCHS_PROTO),
        "--max-episodes", str(max_ep),
        "--n-eval", str(N_EVAL),
        "--n_way", n_way,
        *proto_kw, *base_kw,
    ], "B3 BiLSTM-Proto(original) chemdner [修复:之前误用augmented]")

    # ── Fix 3: B3 genia 完整 8 epoch 重跑 ────────
    n_way = str(N_WAY_PROTO.get("genia", 5))
    max_ep = MAX_EPISODES_PROTO.get("genia", 800)
    run([
        sys.executable, "scripts/run_baseline.py",
        "--dataset", "genia", "--mode", "fewshot",
        "--data-strategy", "original",
        "--encoder-type", "bilstm",
        "--encoder", "unused",
        "--epochs", str(EPOCHS_PROTO),
        "--max-episodes", str(max_ep),
        "--n-eval", str(N_EVAL),
        "--n_way", n_way,
        *proto_kw, *base_kw,
    ], "B3 BiLSTM-Proto(original) genia [修复:之前仅ep=2]")

    # ── Fix 4: B2r chemdner 缺失 ────────
    run([
        sys.executable, "scripts/run_baseline.py",
        "--dataset", "chemdner", "--mode", "seq",
        "--encoder-type", "transformer",
        "--encoder", encoder_roberta,
        "--epochs", str(EPOCHS_SEQ),
        *multi, *base_kw,
    ], "B2r RoBERTa-CRF chemdner [缺失]")

    # ── Fix 5: B-Span+Aug 3 个数据集 ────────
    for ds in DATASETS:
        run([
            sys.executable, "scripts/run_baseline.py",
            "--dataset", ds, "--mode", "supervised",
            "--data-strategy", "augmented",
            "--encoder-type", "transformer",
            "--encoder", encoder_bert,
            "--epochs", str(EPOCHS_SPAN),
            *multi, *base_kw,
        ], f"B-Span+Aug BERT-Span+Aug {ds} [缺失]")

    # ── Summary ────────
    print("\n" + "=" * 60)
    print("[汇总] 主实验修复结果")
    print("=" * 60)
    ok = sum(1 for _, rc in results if rc == 0)
    fail = sum(1 for _, rc in results if rc != 0)
    for desc, rc in results:
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"  [{status:>7s}] {desc}")
    print(f"\n  共 {len(results)} 项: {ok} 成功, {fail} 失败")
    print("=" * 60)


if __name__ == "__main__":
    main()
