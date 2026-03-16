"""
构建转化完后的 span 原型网络训练数据集。

将 data/benchmarks/{fewnerd,genia,chemdner}/ 下已转化的 train/val/test.jsonl
合并为 data/dataset/unified/ 供 run_baseline --dataset unified 使用。

用法：
  python scripts/build_dataset.py
  python scripts/build_dataset.py --datasets fewnerd,genia
  python scripts/build_dataset.py --output data/dataset/unified
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ROOT / "data" / "benchmarks"
DEFAULT_OUTPUT = ROOT / "data" / "dataset" / "unified"
DEFAULT_DATASETS = ["fewnerd", "genia", "chemdner"]


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [warn] {path}:{i} JSON 解析失败: {e}")
                continue
            if not isinstance(obj, dict):
                continue
            ctx = obj.get("context")
            spans = obj.get("spans")
            if ctx is None or spans is None:
                print(f"  [warn] {path}:{i} 缺少 context 或 spans，跳过")
                continue
            if not isinstance(spans, list):
                continue
            out.append(obj)
    return out


def _validate_span(sp: dict) -> bool:
    if not isinstance(sp, dict):
        return False
    start = sp.get("start")
    end = sp.get("end")
    if start is None or end is None:
        return False
    try:
        s, e = int(start), int(end)
        if e <= s:
            return False
    except (TypeError, ValueError):
        return False
    return True


def _write_jsonl(path: Path, samples: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_dataset(
    datasets: list[str],
    output_dir: Path,
    write_manifest: bool = True,
) -> dict:
    """
    合并指定数据集的 train/val/test，输出到 output_dir。
    返回 manifest 统计信息。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "datasets": datasets,
        "splits": {},
        "label_counts": Counter(),
        "samples_per_dataset": {},
    }

    for split in ("train", "val", "test"):
        all_samples: list[dict] = []
        for ds in datasets:
            bench_dir = BENCHMARKS / ds
            # val 可能叫 val.jsonl 或 dev.jsonl
            if split == "val":
                p = bench_dir / "val.jsonl"
                if not p.exists():
                    p = bench_dir / "dev.jsonl"
            else:
                p = bench_dir / f"{split}.jsonl"
            if not p.exists():
                print(f"[skip] {ds}/{split}.jsonl 不存在")
                continue
            samples = _load_jsonl(p)
            valid = []
            for s in samples:
                spans = s.get("spans") or []
                ok_spans = [sp for sp in spans if _validate_span(sp)]
                if ok_spans != spans:
                    s = {**s, "spans": ok_spans}
                if s.get("context"):
                    valid.append(s)
                    for sp in s.get("spans") or []:
                        lab = str(sp.get("label") or "term").strip()
                        if lab:
                            manifest["label_counts"][lab] += 1
            n = len(valid)
            if ds not in manifest["samples_per_dataset"]:
                manifest["samples_per_dataset"][ds] = {}
            manifest["samples_per_dataset"][ds][split] = n
            all_samples.extend(valid)
            print(f"  {ds}/{split}: {n} 条")

        out_path = output_dir / f"{split}.jsonl"
        written = _write_jsonl(out_path, all_samples)
        manifest["splits"][split] = written
        print(f"  -> {out_path} ({written} 条)")

    if write_manifest:
        # 转为可序列化
        manifest_out = {
            "datasets": manifest["datasets"],
            "splits": manifest["splits"],
            "samples_per_dataset": manifest["samples_per_dataset"],
            "label_counts": dict(manifest["label_counts"].most_common(50)),
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest_out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  -> {manifest_path}")

    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="构建转化完后的 span 训练数据集")
    ap.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="逗号分隔的数据集名，如 fewnerd,genia,chemdner",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="输出目录",
    )
    ap.add_argument("--no-manifest", action="store_true", help="不写入 manifest.json")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not datasets:
        datasets = DEFAULT_DATASETS

    print(f"[info] 构建数据集: {datasets}")
    print(f"[info] 输出: {args.output}")
    build_dataset(
        datasets=datasets,
        output_dir=Path(args.output),
        write_manifest=not args.no_manifest,
    )
    print("[ok] 完成")


if __name__ == "__main__":
    main()
