"""
对已有的 train_ready.jsonl 做「嵌套子实体展开」：在每条记录的 entities 中，
为复合实体内部出现的子术语补充 span，使训练数据同时包含父实体与子实体。
不调用标注 API，仅基于现有实体与词典做展开。
"""
import json
from pathlib import Path

from patent_annotator.nested_expand import expand_entities_with_subentities
from patent_annotator.span_format import sentence_record_to_span_sample


def build_lexicon_from_jsonl(records: list[dict]) -> dict[str, str]:
    """从 (sentence, entities) 记录中收集 term -> label，用作子实体 label。"""
    lexicon: dict[str, str] = {}
    for rec in records:
        for e in rec.get("entities") or []:
            if not isinstance(e, dict):
                continue
            t = (e.get("text") or "").strip()
            if t:
                lexicon[t] = (e.get("label") or "term").strip()
    return lexicon


def main() -> None:
    base = Path(__file__).resolve().parent
    input_path = base / "train_ready.jsonl"
    output_path = base / "train_ready.jsonl"
    spans_path = base / "train_spans.jsonl"

    if not input_path.exists():
        print(f"未找到 {input_path.name}，请先运行标注生成该文件。")
        return

    records: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        print("没有有效记录。")
        return

    lexicon = build_lexicon_from_jsonl(records)
    print(f"从现有标注构建词典共 {len(lexicon)} 条，用于子实体匹配与 label。")

    expanded_count = 0
    for rec in records:
        sentence = rec.get("sentence") or ""
        entities = rec.get("entities") or []
        expanded = expand_entities_with_subentities(sentence, entities, lexicon)
        if len(expanded) > len(entities):
            expanded_count += 1
        rec["entities"] = expanded

    print(f"已为 {expanded_count} 条句子补充了嵌套子实体 span。")

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"已写回: {output_path.name}")

    with open(spans_path, "w", encoding="utf-8") as f:
        for rec in records:
            sample = sentence_record_to_span_sample(rec["sentence"], rec["entities"])
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"已写入 Span 格式: {spans_path.name}")


if __name__ == "__main__":
    main()
