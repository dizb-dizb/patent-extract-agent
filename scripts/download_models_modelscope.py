"""
通过 ModelScope 下载 BERT/RoBERTa 模型（解决 HuggingFace 网络不可达问题）

用法：
  python scripts/download_models_modelscope.py
  python scripts/download_models_modelscope.py --output-dir /root/models

依赖：pip install modelscope
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "bert-base-cased": "AI-ModelScope/bert-base-cased",
    "roberta-base": "AI-ModelScope/roberta-base",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="通过 ModelScope 下载 BERT/RoBERTa 模型")
    ap.add_argument("--output-dir", type=str, default="",
                    help="模型保存目录，默认 ROOT/models")
    ap.add_argument("--models", type=str, default="bert-base-cased,roberta-base",
                    help="逗号分隔的模型名")
    args = ap.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ModelScope] 模型将保存到: {out_dir}")

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        print("[fail] 请先安装 modelscope: pip install modelscope")
        sys.exit(1)

    for name in args.models.split(","):
        name = name.strip()
        if name not in MODELS:
            print(f"[skip] 未知模型: {name}")
            continue
        ms_id = MODELS[name]
        target = out_dir / name
        if (target / "config.json").exists():
            print(f"[skip] 已存在: {target}")
            continue
        print(f"\n[download] {name} <- {ms_id}")
        try:
            path = snapshot_download(ms_id, local_dir=str(target))
            print(f"  -> {path}")
        except Exception as e:
            print(f"[fail] {e}")
            sys.exit(1)

    print("\n[ok] 下载完成")
    print(f"  BERT:   --encoder {out_dir / 'bert-base-cased'}")
    print(f"  RoBERTa: --encoder {out_dir / 'roberta-base'}")


if __name__ == "__main__":
    main()
