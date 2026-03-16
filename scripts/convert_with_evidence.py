"""
对 {context, spans} JSONL 逐 span 做证据链检索，有证据的附加到 span，无证据的写入 no_evidence_for_review.jsonl 供人工查验。
支持并行加速、进度条、断点保存与恢复。

用法：
  python scripts/convert_with_evidence.py
  python scripts/convert_with_evidence.py --input data/dataset/unified/train.jsonl --workers 16
  python scripts/convert_with_evidence.py --input data/dataset/unified/train.jsonl --limit 100
  python scripts/convert_with_evidence.py --no-resume  # 忽略已有断点，重新开始
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from patent_agent_pipeline import EvidenceChunk, retrieve_evidence_for_term

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable  # 无 tqdm 时退化为普通迭代


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or obj.get("context") is None:
                continue
            spans = obj.get("spans")
            if spans is None:
                obj["spans"] = []
            out.append(obj)
    return out


def _chunk_to_dict(c: EvidenceChunk) -> dict:
    return {
        "source": c.source,
        "url": c.url,
        "title": c.title,
        "snippet": c.snippet,
        "confidence": c.confidence,
        "retrieved_at": c.retrieved_at,
    }


def _chunk_from_dict(term: str, d: dict | None) -> EvidenceChunk | None:
    """从 checkpoint 恢复 EvidenceChunk。"""
    if not d or not isinstance(d, dict):
        return None
    try:
        return EvidenceChunk(
            term=term,
            source=d.get("source", ""),
            url=d.get("url", ""),
            title=d.get("title", ""),
            snippet=d.get("snippet", ""),
            retrieved_at=int(d.get("retrieved_at", 0)),
            confidence=float(d.get("confidence", 0)),
        )
    except (TypeError, ValueError):
        return None


def _load_checkpoint(checkpoint_path: Path, input_path: Path) -> dict[str, EvidenceChunk | None]:
    """加载断点，若 input_path 不匹配则返回空。"""
    if not checkpoint_path.exists():
        return {}
    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        stored_input = data.get("_input_path")
        if stored_input and str(Path(stored_input).resolve()) != str(input_path.resolve()):
            return {}
        cache = data.get("term_to_chunk", data)
        out: dict[str, EvidenceChunk | None] = {}
        for t, v in cache.items():
            if t.startswith("_"):
                continue
            if v is None:
                out[t] = None
            else:
                out[t] = _chunk_from_dict(t, v)
        return out
    except Exception:
        return {}


def _save_checkpoint(checkpoint_path: Path, input_path: Path, term_to_chunk: dict[str, EvidenceChunk | None]) -> None:
    """保存断点。"""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict | None] = {}
    for t, c in term_to_chunk.items():
        cache[t] = _chunk_to_dict(c) if c else None
    data = {"_input_path": str(input_path.resolve()), "term_to_chunk": cache}
    checkpoint_path.write_text(json.dumps(data, ensure_ascii=False, indent=None), encoding="utf-8")


def convert_with_evidence(
    input_path: Path,
    output_enhanced: Path,
    output_no_evidence: Path,
    limit: int | None = None,
    workers: int = 8,
    resume: bool = True,
    checkpoint_interval: int = 100,
) -> dict:
    """
    读取 {context, spans} JSONL，对每个 span 做证据检索，输出增强 JSONL 和待人工查验列表。
    返回统计信息。
    """
    samples = _load_jsonl(input_path)
    if limit is not None and limit > 0:
        samples = samples[:limit]

    # 1) 收集唯一 term -> 检索证据
    term_to_chunk: dict[str, EvidenceChunk | None] = {}
    unique_terms: list[str] = []
    seen: set[str] = set()
    for rec in samples:
        for sp in rec.get("spans") or []:
            if not isinstance(sp, dict):
                continue
            t = (sp.get("text") or "").strip()
            if not t or t in seen:
                continue
            seen.add(t)
            unique_terms.append(t)

    checkpoint_path = output_enhanced.parent / (output_enhanced.stem + "_checkpoint.json")
    if resume:
        loaded = _load_checkpoint(checkpoint_path, input_path)
        if loaded:
            # 仅恢复当前 run 涉及的术语
            relevant = {k: v for k, v in loaded.items() if k in seen}
            term_to_chunk.update(relevant)
            unique_terms = [t for t in unique_terms if t not in relevant]
            print(f"[info] 从断点恢复 {len(relevant)} 条，剩余 {len(unique_terms)} 条待检索")
    if not unique_terms:
        print("[info] 无待检索术语，直接构建输出")
    else:
        print(f"[info] 唯一 span 术语数: {len(unique_terms)} workers={workers}")

        def _retrieve(t: str) -> tuple[str, EvidenceChunk | None]:
            return (t, retrieve_evidence_for_term(t))

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_retrieve, t): t for t in unique_terms}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="证据检索", unit="term"):
                t, chunk = fut.result()
                term_to_chunk[t] = chunk
                done += 1
                if done % checkpoint_interval == 0:
                    _save_checkpoint(checkpoint_path, input_path, term_to_chunk)
        _save_checkpoint(checkpoint_path, input_path, term_to_chunk)

    # 2) 构建增强样本 + 无证据列表
    enhanced: list[dict] = []
    no_evidence_items: list[dict] = []

    for line_idx, rec in enumerate(samples):
        ctx = rec.get("context") or ""
        spans_in = rec.get("spans") or []
        spans_out: list[dict] = []

        for sp in spans_in:
            if not isinstance(sp, dict):
                continue
            t = (sp.get("text") or "").strip()
            s_out = {
                "start": sp.get("start", 0),
                "end": sp.get("end", 0),
                "label": sp.get("label", "term"),
                "text": t,
                "evidence": None,
            }
            chunk = term_to_chunk.get(t)
            if chunk is not None:
                s_out["evidence"] = _chunk_to_dict(chunk)
            else:
                no_evidence_items.append({
                    "term": t,
                    "label": s_out["label"],
                    "context": ctx,
                    "start": s_out["start"],
                    "end": s_out["end"],
                    "sample_idx": line_idx,
                    "line_idx": line_idx + 1,
                })
            spans_out.append(s_out)

        enhanced.append({"context": ctx, "spans": spans_out})

    # 3) 写入
    output_enhanced.parent.mkdir(parents=True, exist_ok=True)
    output_no_evidence.parent.mkdir(parents=True, exist_ok=True)

    with open(output_enhanced, "w", encoding="utf-8") as f:
        for rec in enhanced:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(output_no_evidence, "w", encoding="utf-8") as f:
        for item in no_evidence_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    all_terms_in_samples = {t for rec in samples for sp in (rec.get("spans") or []) if isinstance(sp, dict) and (t := (sp.get("text") or "").strip())}
    total_terms = len(all_terms_in_samples)
    n_with_ev = sum(1 for t in all_terms_in_samples if term_to_chunk.get(t) is not None)
    return {
        "samples": len(samples),
        "unique_terms": total_terms,
        "terms_with_evidence": n_with_ev,
        "terms_no_evidence": total_terms - n_with_ev,
        "no_evidence_span_occurrences": len(no_evidence_items),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="对 span 数据集做证据链检索，无证据写入待人工查验")
    ap.add_argument(
        "--input",
        type=str,
        default=str(ROOT / "data" / "dataset" / "unified" / "train.jsonl"),
        help="输入 {context, spans} JSONL 路径",
    )
    ap.add_argument(
        "--output-enhanced",
        type=str,
        default=str(ROOT / "data" / "dataset" / "unified" / "train_with_evidence.jsonl"),
        help="输出增强 JSONL 路径",
    )
    ap.add_argument(
        "--output-no-evidence",
        type=str,
        default=str(ROOT / "data" / "dataset" / "unified" / "no_evidence_for_review.jsonl"),
        help="无证据 span 列表，供人工查验",
    )
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 条样本（调试用）")
    ap.add_argument("--workers", type=int, default=8, help="并行 worker 数（默认 8，API 限速时适当降低）")
    ap.add_argument("--no-resume", action="store_true", help="忽略已有断点，重新开始")
    ap.add_argument("--checkpoint-interval", type=int, default=100, help="每 N 条保存一次断点（默认 100）")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"[fail] 输入文件不存在: {inp}")
        sys.exit(1)

    stats = convert_with_evidence(
        input_path=inp,
        output_enhanced=Path(args.output_enhanced),
        output_no_evidence=Path(args.output_no_evidence),
        limit=args.limit,
        workers=args.workers,
        resume=not args.no_resume,
        checkpoint_interval=args.checkpoint_interval,
    )

    print(f"[ok] 增强数据: {args.output_enhanced}")
    print(f"[ok] 待人工查验: {args.output_no_evidence}")
    print(f"[stats] 样本数={stats['samples']} 唯一术语={stats['unique_terms']} "
          f"有证据={stats['terms_with_evidence']} 无证据={stats['terms_no_evidence']} "
          f"无证据出现次数={stats['no_evidence_span_occurrences']}")


if __name__ == "__main__":
    main()
