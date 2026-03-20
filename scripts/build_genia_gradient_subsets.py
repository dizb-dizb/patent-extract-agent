#!/usr/bin/env python3
"""
GENIA 梯度训练子集构建（学术规范版）.

数据来源：仅使用官方 train.jsonl，不从 train 中切分 holdout。
  - 全局测试集：使用官方 test.jsonl（本脚本不修改）
  - 验证集：使用官方 val.jsonl（convert 后即 dev）
  - 梯度子集：对官方 train.jsonl 做一次随机打乱（seed=42），再按“俄罗斯套娃”截取

输出（默认写入 data/benchmarks/genia/）：
  - train_100.jsonl   = 打乱后前 100 条
  - train_1000.jsonl  = 打乱后前 1000 条（包含前 100）
  - train_10000.jsonl = 打乱后前 10000 条（包含前 1000）

用法:
  python scripts/build_genia_gradient_subsets.py
  python scripts/build_genia_gradient_subsets.py --sizes 10,100,1000,10000 --out_dir data/benchmarks/genia
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_GENIA = ROOT / "data" / "benchmarks" / "genia"


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
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


def write_jsonl(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="从官方 train 构建嵌套梯度子集，不使用 holdout")
    ap.add_argument("--train", type=str, default="", help="官方 train.jsonl 路径，默认 data/benchmarks/genia/train.jsonl")
    ap.add_argument("--out_dir", type=str, default="", help="输出目录，默认与 train 同目录")
    ap.add_argument("--sizes", type=str, default="10,100,1000,10000", help="逗号分隔的样本数，如 10,100,1000,10000")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_path = Path(args.train) if args.train else BENCH_GENIA / "train.jsonl"
    if not train_path.is_absolute():
        train_path = ROOT / train_path
    if not train_path.exists():
        raise FileNotFoundError(f"官方 train 不存在: {train_path}。请先运行: python scripts/download_benchmarks.py genia && python scripts/convert_genia.py")

    out_dir = Path(args.out_dir) if args.out_dir else train_path.parent
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    sizes.sort()

    samples = load_jsonl(train_path)
    if not samples:
        raise RuntimeError("train.jsonl 为空")

    random.seed(args.seed)
    shuffled = list(samples)
    random.shuffle(shuffled)

    for n in sizes:
        if n > len(shuffled):
            print(f"[skip] n={n} 超过 train 样本数 {len(shuffled)}")
            continue
        subset = shuffled[:n]
        out_path = out_dir / f"train_{n}.jsonl"
        write_jsonl(out_path, subset)
        print(f"[ok] {out_path} ({len(subset)} 条，来自官方 train 打乱后前 {n} 条)")

    print("\n[说明] 全局测试集请使用官方 test.jsonl，勿从 train 切分。验证集使用 val.jsonl。")
    test_p = out_dir / "test.jsonl"
    val_p = out_dir / "val.jsonl"
    if test_p.exists():
        print(f"  官方测试集: {test_p}")
    if val_p.exists():
        print(f"  官方验证集: {val_p}")


if __name__ == "__main__":
    main()
