"""
释放已有实验结果目录中的模型参数文件，仅保留 metrics.json、labels.json 等小文件，防止磁盘占满。

删除：model.pt, pytorch_model.bin, tokenizer.json（大文件）
保留：metrics.json, labels.json, config.json, tokenizer_config.json 等

用法:
  python scripts/release_artifact_models.py
  python scripts/release_artifact_models.py --artifacts-dir /root/patent-extract-agent/artifacts
  python scripts/release_artifact_models.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REMOVE_GLOBS = ["model.pt", "pytorch_model.bin", "tokenizer.json", "training_args.bin"]


def main() -> None:
    ap = argparse.ArgumentParser(description="释放 artifacts 下已完成实验的模型权重")
    ap.add_argument("--artifacts-dir", type=str, default="", help="artifacts 目录")
    ap.add_argument("--dry-run", action="store_true", help="仅打印不删除")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else ROOT / "artifacts"
    if not artifacts_dir.exists():
        print(f"[skip] 目录不存在: {artifacts_dir}")
        return

    total_freed = 0
    removed_count = 0

    for subdir in sorted(artifacts_dir.rglob("*")):
        if not subdir.is_dir():
            continue
        if not (subdir / "metrics.json").exists():
            continue
        for name in REMOVE_GLOBS:
            f = subdir / name
            if not f.exists():
                continue
            size = f.stat().st_size
            if args.dry_run:
                print(f"  [dry-run] 将删除 {f.relative_to(artifacts_dir)} ({size / (1024*1024):.2f} MiB)")
            else:
                try:
                    f.unlink()
                    print(f"  已删除 {f.relative_to(artifacts_dir)} ({size / (1024*1024):.2f} MiB)")
                except OSError as e:
                    print(f"  [warn] 删除失败 {f}: {e}", file=sys.stderr)
                    continue
            total_freed += size
            removed_count += 1

    if removed_count == 0:
        print("[ok] 无可释放的模型文件")
    else:
        print(f"\n[ok] 共释放 {removed_count} 个文件，约 {total_freed / (1024*1024):.2f} MiB")
    if args.dry_run and removed_count > 0:
        print("  以上为 dry-run，未实际删除。")


if __name__ == "__main__":
    main()
