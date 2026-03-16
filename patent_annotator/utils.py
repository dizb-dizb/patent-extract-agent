"""专利/论文标注工具：使用 re.finditer 对模型返回的 text 在原文中重定位，修正 start/end 索引偏差。"""
import re
from .schema import NestedEntity


def find_text_span(source: str, text: str, start_hint: int | None = None, end_hint: int | None = None) -> tuple[int, int] | None:
    if not text:
        return None
    pattern = re.escape(text)
    start_search = start_hint if start_hint is not None else 0
    end_search = end_hint if end_hint is not None else len(source)
    search_region = source[start_search:end_search]
    for m in re.finditer(pattern, search_region):
        return (start_search + m.start(), start_search + m.end())
    if start_hint is not None or end_hint is not None:
        for m in re.finditer(pattern, source):
            return (m.start(), m.end())
    return None


def recalculate_entity_span(
    source: str,
    entity: NestedEntity,
    start_hint: int | None = None,
    end_hint: int | None = None,
) -> NestedEntity:
    """在原文上重算单条实体的 start/end。"""
    span = find_text_span(source, entity.text, start_hint, end_hint)
    new_start, new_end = span if span else (entity.start, entity.end)
    return NestedEntity(text=entity.text, label=entity.label, start=new_start, end=new_end)


def recalculate_nested_spans(source: str, entities: list[NestedEntity]) -> list[NestedEntity]:
    """对扁平实体列表在原文上重算 start/end。"""
    return [recalculate_entity_span(source, e) for e in entities]
