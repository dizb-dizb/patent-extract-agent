"""
Utilities to convert span-based samples into token-level BIO format
for sequence labelling models (BiLSTM-CRF / BERT / RoBERTa).

Input jsonl format (train_spans*.jsonl):
  {"context": "...", "spans": [{"start":0,"end":2,"label":"Condition","text":".."}, ...]}

Export:
  - .bio.txt : one sentence per block, "char label" per line, blank line between sentences.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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


def spans_to_bio(context: str, spans: list[dict]) -> list[tuple[str, str]]:
    """
    Very simple char-level BIO scheme.
    WARNING: does not handle crossing spans; overlapping resolved by
    preferring longer spans (applied greedily).
    """
    s = context or ""
    n = len(s)
    labels = ["O"] * n

    # sort spans by length desc to prefer longer ones
    norm_spans: list[tuple[int, int, str]] = []
    for e in spans or []:
        if not isinstance(e, dict):
            continue
        start = int(e.get("start", 0))
        end = int(e.get("end", 0))
        if start < 0 or end <= start or end > n:
            continue
        lab = str(e.get("label", "TERM")).strip() or "TERM"
        norm_spans.append((start, end, lab))
    norm_spans.sort(key=lambda x: (x[1] - x[0]), reverse=True)

    for start, end, lab in norm_spans:
        # if any position already tagged by another longer span, we skip
        if any(lbl != "O" for lbl in labels[start:end]):
            continue
        labels[start] = f"B-{lab}"
        for i in range(start + 1, end):
            labels[i] = f"I-{lab}"

    return [(ch, labels[i]) for i, ch in enumerate(s)]


def export_bio(jsonl_path: Path, out_path: Path) -> None:
    data = load_jsonl(jsonl_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in data:
            ctx = sample.get("context") or ""
            spans = sample.get("spans") or []
            seq = spans_to_bio(ctx, spans)
            for ch, tag in seq:
                # use underscore for spaces to keep alignment
                token = ch if ch != " " else "_"
                f.write(f"{token} {tag}\n")
            f.write("\n")


def main() -> None:
    root = Path(__file__).resolve().parent
    src = root / "train_spans_augmented.jsonl"
    if not src.exists():
        src = root / "train_spans_enhanced.jsonl"
    if not src.exists():
        src = root / "train_spans.jsonl"
    if not src.exists():
        print("[fail] no spans dataset found")
        return
    out = root / "train.bio.txt"
    export_bio(src, out)
    print(f"[ok] exported BIO to {out} from {src.name}")


if __name__ == "__main__":
    main()

