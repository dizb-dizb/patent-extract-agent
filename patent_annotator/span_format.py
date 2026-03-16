"""将实体列表转为 Span-based 训练格式。"""


def entities_to_spans(entities: list[dict]) -> list[dict]:
    """将扁平实体列表转为 span 列表，每个实体对应一条 span。"""
    spans: list[dict] = []
    for e in entities or []:
        if not isinstance(e, dict):
            continue
        text = (e.get("text") or "").strip()
        if not text:
            continue
        spans.append({
            "start": int(e.get("start", 0)),
            "end": int(e.get("end", 0)),
            "label": e.get("label", "term"),
            "text": text,
        })
    return spans


def sentence_record_to_span_sample(sentence: str, entities: list[dict]) -> dict:
    """单条 (sentence, entities) 转为 {context, spans} 训练样本。"""
    return {
        "context": sentence,
        "spans": entities_to_spans(entities),
    }
