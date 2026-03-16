"""
Few-NERD: tokens + ner_tags -> {context, spans} span format.
Reads data/benchmarks/fewnerd/raw/*.jsonl, writes data/benchmarks/fewnerd/{train,val,test}.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "benchmarks" / "fewnerd" / "raw"
OUT = ROOT / "data" / "benchmarks" / "fewnerd"


def load_label_names() -> dict[int, str]:
    path = RAW / "label_names.json"
    if not path.exists():
        return {}
    names = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {}


def tokens_tags_to_spans(tokens: list[str], tags: list[int], label_names: dict[int, str]) -> list[dict]:
    """Convert token-level tags to character-level spans. IO scheme: O=0, entity types 1+."""
    if not tokens or not tags:
        return []
    text = " ".join(tokens)
    token_starts = []
    pos = 0
    for t in tokens:
        token_starts.append(pos)
        pos += len(t) + 1
    spans = []
    i = 0
    while i < len(tags):
        tid = tags[i]
        if tid == 0 or (label_names and str(label_names.get(tid, "O")) == "O"):
            i += 1
            continue
        lab = label_names.get(tid, f"type_{tid}")
        start_char = token_starts[i]
        j = i + 1
        while j < len(tags) and tags[j] == tid:
            j += 1
        end_char = token_starts[j - 1] + len(tokens[j - 1])
        span_text = text[start_char:end_char]
        spans.append({"start": start_char, "end": end_char, "label": lab, "text": span_text})
        i = j
    return spans


def convert_file(in_path: Path, out_path: Path, label_names: dict[int, str]) -> int:
    count = 0
    with open(out_path, "w", encoding="utf-8") as outf:
        for line in in_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tokens = obj.get("tokens") or []
            tags = obj.get("fine_ner_tags") or obj.get("ner_tags") or []
            if len(tokens) != len(tags):
                continue
            spans = tokens_tags_to_spans(tokens, tags, label_names)
            text = " ".join(tokens)
            outf.write(json.dumps({"context": text, "spans": spans}, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    label_names = load_label_names()
    for fname, out_fname in [("train", "train"), ("dev", "val"), ("test", "test")]:
        in_p = RAW / f"{fname}.jsonl"
        out_p = OUT / f"{out_fname}.jsonl"
        if not in_p.exists():
            print(f"[skip] {in_p} not found")
            continue
        n = convert_file(in_p, out_p, label_names)
        print(f"[ok] {out_p} ({n} samples)")


if __name__ == "__main__":
    main()
