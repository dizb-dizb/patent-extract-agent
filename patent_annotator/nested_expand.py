"""将复合实体展开为「父实体 + 子实体」的嵌套标注，便于模型同时学习整体与内部片段。"""
from __future__ import annotations


def _collect_subterm_candidates(entities: list[dict], lexicon: dict[str, str]) -> list[tuple[str, str]]:
    """收集可用于子实体的 (text, label)：来自词典 + 同句其他实体的 text。去重且优先词典 label。"""
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for e in entities or []:
        if not isinstance(e, dict):
            continue
        t = (e.get("text") or "").strip()
        if not t or len(t) < 2:  # 过短不当作子片段
            continue
        if t in seen:
            continue
        seen.add(t)
        label = (e.get("label") or "").strip() or lexicon.get(t, "term")
        candidates.append((t, label))
    for term, label in lexicon.items():
        term = (term or "").strip()
        if not term or len(term) < 2 or term in seen:
            continue
        seen.add(term)
        candidates.append((term, label))
    return candidates


def _find_all_substring_offsets(haystack: str, needle: str) -> list[int]:
    """返回 needle 在 haystack 中所有出现的起始下标。"""
    if not needle or needle not in haystack:
        return []
    out: list[int] = []
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i < 0:
            break
        out.append(i)
        start = i + 1
    return out


def expand_entities_with_subentities(
    sentence: str,
    entities: list[dict],
    lexicon: dict[str, str],
    min_sub_len: int = 2,
) -> list[dict]:
    """
    在现有实体列表上，为每个实体内部出现的「子术语」补充 span，得到支持嵌套的实体列表。
    - 子术语来源：当前句内其他实体的 text、以及 lexicon 中的术语（在实体 text 内出现即视为子实体）。
    - 不改变原有实体，仅追加子实体；若某 (start, end) 已存在则不再追加。
    - 返回按 start 再 end 排序的实体列表（含重叠/嵌套 span）。
    """
    if not entities:
        return []
    candidates = _collect_subterm_candidates(entities, lexicon)
    existing_spans: set[tuple[int, int]] = set()
    for e in entities:
        if isinstance(e, dict) and "start" in e and "end" in e:
            existing_spans.add((int(e["start"]), int(e["end"])))

    added: list[dict] = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        text = (e.get("text") or "").strip()
        s_start = int(e.get("start", 0))
        s_end = int(e.get("end", 0))
        if not text:
            continue
        # 在实体文本内查找所有子术语出现位置
        for sub_text, sub_label in candidates:
            if sub_text == text or len(sub_text) < min_sub_len:
                continue
            if sub_text not in text:
                continue
            for off in _find_all_substring_offsets(text, sub_text):
                g_start = s_start + off
                g_end = g_start + len(sub_text)
                if (g_start, g_end) in existing_spans:
                    continue
                # 确保不超出句子且与原文一致
                if g_end > len(sentence) or g_start < 0:
                    continue
                if sentence[g_start:g_end] != sub_text:
                    continue
                existing_spans.add((g_start, g_end))
                added.append({
                    "text": sub_text,
                    "label": sub_label,
                    "start": g_start,
                    "end": g_end,
                })

    result = list(entities) + added
    result.sort(key=lambda x: (int(x.get("start", 0)), int(x.get("end", 0))))
    return result
