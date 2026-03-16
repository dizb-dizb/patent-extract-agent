"""
Evidence-based data augmentation (no LLM):

Input:
  - train_spans_enhanced.jsonl (preferred)
  - train_spans.jsonl (fallback)
  - *_with_evidence.jsonl (unified/split format, via --input)

For each span that has an evidence snippet, we:
  - search the term text inside the snippet
  - if found, create a new training sample where:
      context = snippet
      spans   = single span covering that occurrence

Output:
  - train_spans_augmented.jsonl (default)
  - or path specified by --output

用法：
  python augment_from_evidence.py
  python augment_from_evidence.py --input data/dataset/split/fewnerd_train_with_evidence.jsonl --output data/dataset/split/fewnerd_train_augmented.jsonl
  python augment_from_evidence.py --input-dir data/dataset/split --output-dir data/dataset/split  # 批量处理 *_with_evidence.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SPANS_ENH = ROOT / "train_spans_enhanced.jsonl"
SPANS = ROOT / "train_spans.jsonl"
OUT = ROOT / "train_spans_augmented.jsonl"
SPLIT_DIR = ROOT / "data" / "dataset" / "split"


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


def make_aug_sample_from_snippet(term: str, label: str, snippet: str) -> dict | None:
    t = (term or "").strip()
    s = (snippet or "").strip()
    if not t or not s:
        return None
    idx = s.find(t)
    if idx < 0:
        return None
    return {
        "context": s,
        "spans": [
            {
                "start": idx,
                "end": idx + len(t),
                "label": label or "term",
                "text": t,
            }
        ],
    }


def augment_data(data: list[dict]) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for s in data:
        augmented.append(s)
        for sp in s.get("spans") or []:
            if not isinstance(sp, dict):
                continue
            ev = sp.get("evidence") or {}
            snippet = (ev.get("snippet") or "").strip()
            if not snippet:
                continue
            term = (sp.get("text") or "").strip()
            label = (sp.get("label") or "term").strip()
            aug = make_aug_sample_from_snippet(term, label, snippet)
            if aug is not None:
                augmented.append(aug)
    return augmented


def run_single(input_path: Path, output_path: Path) -> int:
    data = load_jsonl(input_path)
    augmented = augment_data(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in augmented:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return len(augmented) - len(data)


def run_batch(input_dir: Path, output_dir: Path) -> dict[str, int]:
    """处理 input_dir 下所有 *_with_evidence.jsonl，输出到 output_dir 的 *_augmented.jsonl"""
    pattern = re.compile(r"^(.+)_(train|val|test)_with_evidence\.jsonl$")
    results: dict[str, int] = {}
    for p in sorted(input_dir.iterdir()):
        if not p.is_file() or not p.name.endswith("_with_evidence.jsonl"):
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        prefix, split = m.group(1), m.group(2)
        out_name = f"{prefix}_{split}_augmented.jsonl"
        out_path = output_dir / out_name
        added = run_single(p, out_path)
        results[out_name] = added
        print(f"  {p.name} -> {out_name} (+{added})")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Evidence-based span augmentation")
    ap.add_argument("--input", type=str, default="", help="输入 JSONL 路径")
    ap.add_argument("--output", type=str, default="", help="输出 JSONL 路径")
    ap.add_argument("--input-dir", type=str, default="", help="批量输入目录（处理 *_with_evidence.jsonl）")
    ap.add_argument("--output-dir", type=str, default="", help="批量输出目录")
    args = ap.parse_args()

    if args.input_dir and args.output_dir:
        inp = Path(args.input_dir)
        out = Path(args.output_dir)
        if not inp.exists():
            print(f"[fail] 输入目录不存在: {inp}")
            return
        print(f"[info] 批量处理: {inp} -> {out}")
        run_batch(inp, out)
        print("[ok] 批量增强完成")
        return

    if args.input and args.output:
        src = Path(args.input)
        dst = Path(args.output)
        if not src.exists():
            print(f"[fail] 输入文件不存在: {src}")
            return
        added = run_single(src, dst)
        print(f"[ok] {dst.name} (added {added} augmented)")
        return

    # 默认：使用 train_spans_enhanced / train_spans
    src = SPANS_ENH if SPANS_ENH.exists() else SPANS
    if not src.exists():
        print(f"[fail] missing dataset: {src}")
        return

    data = load_jsonl(src)
    augmented = augment_data(data)
    with open(OUT, "w", encoding="utf-8") as f:
        for s in augmented:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    added = len(augmented) - len(data)
    print(f"[ok] wrote {len(augmented)} samples to {OUT.name} (added {added} augmented)")


if __name__ == "__main__":
    main()

