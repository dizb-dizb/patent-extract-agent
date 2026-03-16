"""
GENIA: raw span JSONL -> data/benchmarks/genia/{train,val,test}.jsonl
Download script already outputs span format to raw/. This copies to final output.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "benchmarks" / "genia" / "raw"
OUT = ROOT / "data" / "benchmarks" / "genia"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for fname, out_f in [("train", "train"), ("dev", "val"), ("test", "test")]:
        in_p = RAW / f"{fname}.jsonl"
        out_p = OUT / f"{out_f}.jsonl"
        if in_p.exists():
            shutil.copy2(in_p, out_p)
            n = sum(1 for _ in open(out_p, encoding="utf-8"))
            print(f"[ok] {out_p} ({n} samples)")
        else:
            print(f"[skip] {in_p} not found. Run: python scripts/download_benchmarks.py genia")


if __name__ == "__main__":
    main()
