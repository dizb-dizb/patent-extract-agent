"""
运行剩余主实验（全数据集，无原型网络）：B2r, B-Span+Aug
原型网络实验（B4r/B4f/B4rf/B5/Ours/Ours-r）已移至 run_gradient_isolate_unified.py。

跳过已完成的 B1, B2, B-Span, B3, B4。
AutoDL 用法: bash scripts/autodl_run_supplementary.sh
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["fewnerd", "genia", "chemdner"]
EPOCHS_B2 = 5


def run(cmd: list[str], desc: str) -> int:
    print(f"\n{'='*60}")
    print(f"[run] {desc}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*60}")
    ret = subprocess.run(cmd, cwd=str(ROOT))
    if ret.returncode != 0:
        print(f"[warn] {desc} exited with code {ret.returncode}")
    return ret.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default="fewnerd,genia,chemdner")
    ap.add_argument("--models-dir", type=str, default="",
                    help="本地模型目录，如 /root/models")
    ap.add_argument("--multi-gpu", action="store_true", default=True)
    ap.add_argument("--no-multi-gpu", action="store_false", dest="multi_gpu")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    models_dir = Path(args.models_dir) if args.models_dir else None
    encoder_bert = str(models_dir / "bert-base-cased") if models_dir else "bert-base-cased"
    encoder_roberta = str(models_dir / "roberta-base") if models_dir else "roberta-base"
    if models_dir:
        print(f"[config] BERT={encoder_bert}, RoBERTa={encoder_roberta}")

    multi = ["--multi-gpu"] if args.multi_gpu else []
    base_kw = ["--seed", "42"]

    for ds in datasets:
        # B2r: RoBERTa-CRF
        run([
            sys.executable, "scripts/run_baseline.py",
            "--dataset", ds, "--mode", "seq", "--encoder-type", "transformer",
            "--encoder", encoder_roberta, "--epochs", str(EPOCHS_B2),
            *multi, *base_kw
        ], f"B2r RoBERTa-CRF {ds}")

        # B-Span+Aug: BERT-Span(无原型) + 增强数据
        run([
            sys.executable, "scripts/run_baseline.py",
            "--dataset", ds, "--mode", "supervised", "--data-strategy", "augmented",
            "--encoder-type", "transformer", "--encoder", encoder_bert,
            "--epochs", str(EPOCHS_B2), *multi, *base_kw
        ], f"B-Span+Aug BERT-Span+Aug {ds}")

    print("\n" + "="*60)
    print("[ok] 剩余主实验完成")
    print("  实验: B2r, B-Span+Aug (全数据集，无原型网络)")
    print("  原型网络实验见: python scripts/run_gradient_isolate_unified.py")
    print("  查看: python _progress.py")
    print("="*60)


if __name__ == "__main__":
    main()
