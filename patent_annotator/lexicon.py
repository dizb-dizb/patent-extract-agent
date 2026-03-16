"""专利术语词典：从标注结果整合所有术语并生成 label，再填回实体。"""

from collections import OrderedDict


def collect_terms_from_entities(entities: list[dict]) -> list[str]:
    """递归收集所有实体的 text，去重并保持顺序。"""
    seen: set[str] = set()
    order: list[str] = []

    def visit(es: list[dict]) -> None:
        for e in es:
            if not isinstance(e, dict):
                continue
            t = (e.get("text") or "").strip()
            if t and t not in seen:
                seen.add(t)
                order.append(t)
            for c in e.get("children") or []:
                visit([c])

    visit(entities)
    return order


def build_lexicon_from_records(
    all_sentence_records: list[tuple[str, list[dict]]],
) -> dict[str, str]:
    """从所有 (sentence, entities) 中收集术语，生成词典并返回 term_text -> label 映射。"""
    order: OrderedDict[str, None] = OrderedDict()
    for _, entities in all_sentence_records:
        for term in collect_terms_from_entities(entities):
            order[term] = None
    # 生成 label：term_0, term_1, ... 或直接用术语文本作为 label 键
    return {term: f"term_{i}" for i, term in enumerate(order)}


def fill_entity_labels(entity: dict, lexicon: dict[str, str]) -> None:
    """递归将 entity 及其 children 的 label 按词典填回（原地修改）。"""
    text = (entity.get("text") or "").strip()
    if text and text in lexicon:
        entity["label"] = lexicon[text]
    for c in entity.get("children") or []:
        fill_entity_labels(c, lexicon)


def fill_records_with_lexicon(
    all_sentence_records: list[tuple[str, list[dict]]],
    lexicon: dict[str, str],
) -> None:
    """对所有 sentence_records 中的 entities 按 lexicon 填回 label（原地修改）。"""
    for _, entities in all_sentence_records:
        for e in entities:
            fill_entity_labels(e, lexicon)
