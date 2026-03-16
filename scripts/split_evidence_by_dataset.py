"""
按数据集拆分证据增强数据（无需重跑）。

基于 build_dataset 的合并顺序（fewnerd → genia → chemdner）与 manifest 中的 samples_per_dataset，
按行范围将 train/val/test_with_evidence.jsonl 拆分为 fewnerd、genia、chemdner 三个数据集。
同时拆分 no_evidence_*.jsonl（按 sample_idx 映射到数据集）。

用法：
  python scripts/split_evidence_by_dataset.py
  python scripts/split_evidence_by_dataset.py --input-dir data/dataset/unified --output-dir data/dataset/split
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "dataset" / "unified"
DEFAULT_OUTPUT = ROOT / "data" / "dataset" / "split"


def _load_manifest(input_dir: Path) -> dict:
    path = input_dir / "manifest.json"
    if not path.exists():
        print(f"[fail] manifest.json 不存在: {path}")
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_ranges(samples_per_dataset: dict, datasets: list[str]) -> dict[str, dict[str, tuple[int, int]]]:
    """
    计算每个数据集在每个 split 下的 (start_line, end_line) 范围（含 start，不含 end）。
    行号 0-based。
    """
    ranges: dict[str, dict[str, tuple[int, int]]] = {}
    for split in ("train", "val", "test"):
        ranges[split] = {}
        pos = 0
        for ds in datasets:
            n = samples_per_dataset.get(ds, {}).get(split, 0)
            ranges[split][ds] = (pos, pos + n)
            pos += n
    return ranges


def _split_with_evidence(
    input_dir: Path,
    output_dir: Path,
    ranges: dict[str, dict[str, tuple[int, int]]],
    datasets: list[str],
) -> dict[str, dict[str, int]]:
    """拆分 *_with_evidence.jsonl，返回各文件实际写入行数。"""
    written: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        inp_path = input_dir / f"{split}_with_evidence.jsonl"
        if not inp_path.exists():
            print(f"[skip] {inp_path} 不存在")
            continue
        written[split] = {}
        lines: list[str] = []
        with open(inp_path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        for ds in datasets:
            start, end = ranges[split][ds]
            chunk = lines[start:end]
            out_path = output_dir / f"{ds}_{split}_with_evidence.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for ln in chunk:
                    f.write(ln if ln.endswith("\n") else ln + "\n")
            written[split][ds] = len(chunk)
            print(f"  {ds}/{split}: {len(chunk)} -> {out_path.name}")
    return written


def _split_no_evidence(
    input_dir: Path,
    output_dir: Path,
    ranges: dict[str, dict[str, tuple[int, int]]],
    datasets: list[str],
) -> dict[str, dict[str, int]]:
    """
    拆分 no_evidence_*.jsonl。每条记录的 sample_idx 对应 unified 中的行号（0-based），
    据此判断所属数据集。
    """
    written: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        inp_path = input_dir / f"no_evidence_{split}.jsonl"
        if not inp_path.exists():
            continue
        by_ds: dict[str, list[dict]] = {ds: [] for ds in datasets}
        with open(inp_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                idx = int(rec.get("sample_idx", -1))
                if idx < 0:
                    continue
                for ds in datasets:
                    start, end = ranges[split][ds]
                    if start <= idx < end:
                        by_ds[ds].append(rec)
                        break
        written[split] = {}
        for ds in datasets:
            items = by_ds[ds]
            if not items:
                continue
            out_path = output_dir / f"{ds}_no_evidence_{split}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in items:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written[split][ds] = len(items)
            print(f"  {ds}/no_evidence_{split}: {len(items)} -> {out_path.name}")
    return written


def _validate(
    written_ev: dict[str, dict[str, int]],
    written_no: dict[str, dict[str, int]],
    samples_per_dataset: dict,
    datasets: list[str],
) -> bool:
    """校验拆分后行数与 manifest 一致。"""
    ok = True
    for split in ("train", "val", "test"):
        for ds in datasets:
            expected = samples_per_dataset.get(ds, {}).get(split, 0)
            actual = written_ev.get(split, {}).get(ds, 0)
            if actual != expected:
                print(f"[warn] {ds}/{split}_with_evidence: 期望 {expected}, 实际 {actual}")
                ok = False
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="按数据集拆分证据增强数据")
    ap.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT))
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    manifest = _load_manifest(input_dir)
    datasets = manifest.get("datasets", ["fewnerd", "genia", "chemdner"])
    samples_per_dataset = manifest.get("samples_per_dataset", {})

    print(f"[info] 数据集顺序: {datasets}")
    ranges = _compute_ranges(samples_per_dataset, datasets)

    print("[info] 拆分 *_with_evidence.jsonl")
    written_ev = _split_with_evidence(input_dir, output_dir, ranges, datasets)

    print("[info] 拆分 no_evidence_*.jsonl")
    written_no = _split_no_evidence(input_dir, output_dir, ranges, datasets)

    if _validate(written_ev, written_no, samples_per_dataset, datasets):
        print("[ok] 校验通过")
    else:
        print("[warn] 部分文件行数与 manifest 不一致，请检查")

    print(f"[ok] 输出目录: {output_dir}")


if __name__ == "__main__":
    main()
