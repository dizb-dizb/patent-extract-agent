#!/usr/bin/env python3
"""
两阶段隔离训练方案 (Two-Stage Decoupled Training) 完整流程

阶段一：领域通用跨度预训练 (Class-Agnostic Span Pre-training)
  - 数据：Subset-10000
  - 模块：RoBERTa + Span 提议层
  - 目标：二分类（实体 vs 非实体），BCE Loss
  - 输出：encoder_span_proj.pt

阶段二：冻结底座的度量微调 (Frozen Metric Fine-tuning)
  - 数据：Subset-100 / Subset-1000
  - 模块：冻结 encoder+span_proj，仅训练 Projector
  - 目标：N-way K-shot 原型网络 CE（默认不加 SCL，可用 --scl_weight>0 开启）
  - 输出：projector_stage2.pt

用法:
  python scripts/run_two_stage_decoupled.py --dataset genia
  python scripts/run_two_stage_decoupled.py --dataset genia --stage2_size 1000
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ROOT / "data" / "benchmarks"


def main() -> None:
    ap = argparse.ArgumentParser(description="两阶段隔离训练完整流程")
    ap.add_argument("--dataset", type=str, default="genia", choices=["genia", "fewnerd", "chemdner"])
    ap.add_argument("--stage1_size", type=int, default=10000)
    ap.add_argument("--stage2_size", type=int, default=100, help="100 或 1000")
    ap.add_argument("--build_subsets", action="store_true", help="先构建梯度子集")
    ap.add_argument("--skip_stage1", action="store_true", help="跳过阶段一（使用已有 ckpt）")
    ap.add_argument("--output_dir", type=str, default="")
    ap.add_argument("--encoder", type=str, default="roberta-base")
    ap.add_argument("--batch_stage1", type=int, default=0, help="阶段一 batch_size，0=默认 8")
    ap.add_argument("--batch_ep_stage2", type=int, default=0, help="阶段二 batch_episodes，0=默认 4")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument(
        "--scl_weight",
        type=float,
        default=0.0,
        help="阶段二 SCL 权重；0=不使用 SCL（默认）",
    )
    args = ap.parse_args()

    ds_dir = BENCHMARKS / args.dataset
    train_path = ds_dir / "train.jsonl"
    if not train_path.exists():
        print(f"[err] {train_path} 不存在。请先运行: python scripts/download_benchmarks.py {args.dataset}")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "artifacts" / "two_stage_decoupled" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1_data = ds_dir / f"train_{args.stage1_size}.jsonl"
    stage2_data = ds_dir / f"train_{args.stage2_size}.jsonl"
    stage1_ckpt = out_dir / "encoder_span_proj.pt"

    if args.build_subsets or not stage1_data.exists() or not stage2_data.exists():
        sizes = sorted(set([args.stage1_size, args.stage2_size] + [10, 100, 1000, 10000]))
        sizes = [s for s in sizes if s <= 100000]
        cmd = [
            sys.executable,
            "scripts/build_genia_gradient_subsets.py",
            "--train", str(train_path),
            "--out_dir", str(ds_dir),
            "--sizes", ",".join(map(str, sizes)),
        ]
        print("[1/4] 构建梯度子集...")
        ret = subprocess.run(cmd, cwd=ROOT)
        if ret.returncode != 0:
            print("[warn] 构建失败，请确保 train.jsonl 存在。fewnerd/chemdner 可手动复制 train.jsonl 为 train_100.jsonl 等")

    if not stage1_data.exists():
        print(f"[err] 阶段一数据不存在: {stage1_data}")
        sys.exit(1)
    if not stage2_data.exists():
        print(f"[err] 阶段二数据不存在: {stage2_data}")
        sys.exit(1)

    if not args.skip_stage1:
        print("[2/4] 阶段一：BCE 跨度预训练...")
        cmd = [
            sys.executable,
            "scripts/train_span_entity_bce.py",
            "--data", str(stage1_data),
            "--output_dir", str(out_dir),
            "--encoder", args.encoder,
        ]
        subprocess.run(cmd, cwd=ROOT, check=True)
    else:
        if not stage1_ckpt.exists():
            print(f"[err] --skip_stage1 但 {stage1_ckpt} 不存在")
            sys.exit(1)
        print("[2/4] 跳过阶段一，使用已有 ckpt")

    print("[3/4] 阶段二：Projector 度量微调...")
    cmd = [
        sys.executable,
        "scripts/train_stage2_projector_proto.py",
        "--data", str(stage2_data),
        "--stage1_ckpt", str(stage1_ckpt),
        "--output_dir", str(out_dir),
        "--encoder", args.encoder,
    ]
    if args.batch_ep_stage2 > 0:
        cmd += ["--batch_episodes", str(args.batch_ep_stage2)]
    cmd += ["--scl_weight", str(args.scl_weight)]
    if args.fp16:
        cmd += ["--fp16"]
    subprocess.run(cmd, cwd=ROOT, check=True)

    print("[4/4] 完成")
    print(f"  阶段一 ckpt: {stage1_ckpt}")
    print(f"  阶段二 ckpt: {out_dir / 'projector_stage2.pt'}")


if __name__ == "__main__":
    main()
