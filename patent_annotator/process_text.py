"""文本清洗、分句与分批逻辑。"""
import re

HEADER_FOOTER_PATTERN = re.compile(
    r"^(?:\d{1,5}\s*[|]\s*)?"
    r"(?:[A-Za-z][\w\s\-\.]+(?:Journal|Review|Science|Nature|Proceedings|Acta|IEEE)\b[^.]*\.?)?"
    r"(?:\s*[|]\s*\d{1,5})?\s*$",
    re.MULTILINE,
)
PAGE_NUMBER_ONLY = re.compile(r"^\s*\d{1,5}\s*$", re.MULTILINE)
DOI_LINE = re.compile(r"^\s*(?:https?://)?(?:doi\.org/|doi:)\s*\S+\s*$", re.MULTILINE | re.IGNORECASE)
HYPHEN_LINE_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)", re.MULTILINE)
MULTI_NEWLINE = re.compile(r"\n\s*\n\s*")


def clean_pdf_text(text: str) -> str:
    if not text or not text.strip():
        return text
    s = text.strip()
    lines = s.split("\n")
    cleaned_lines = []
    for line in lines:
        ls = line.strip()
        if HEADER_FOOTER_PATTERN.fullmatch(ls) or PAGE_NUMBER_ONLY.fullmatch(ls) or DOI_LINE.fullmatch(ls):
            continue
        cleaned_lines.append(line)
    s = "\n".join(cleaned_lines)
    s = HYPHEN_LINE_BREAK.sub(r"\1\2", s)
    s = MULTI_NEWLINE.sub(" ", s)
    s = re.sub(r"\n", " ", s)
    s = re.sub(r" +", " ", s).strip()
    return s


def split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    try:
        import nltk
        nltk.data.find("tokenizers/punkt")
    except (ImportError, LookupError):
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except Exception:
        pass
    parts = re.split(r"(?<=[。！？.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def batch_sentences(sentences: list[str], batch_size: int = 15) -> list[list[str]]:
    return [sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)]


def canonical_text_and_batch_offsets(sentences: list[str], batch_size: int, sep: str = "\n") -> tuple[str, list[int]]:
    if not sentences:
        return "", []
    offsets = [0]
    for s in sentences:
        offsets.append(offsets[-1] + len(s) + len(sep))
    canonical = sep.join(sentences)
    batch_starts = [offsets[i * batch_size] for i in range((len(sentences) + batch_size - 1) // batch_size)]
    return canonical, batch_starts


def merge_batch_results(batch_entities_list: list[list[dict]], batch_starts: list[int]) -> list[dict]:
    merged = []

    def add_offset(e: dict, base: int) -> dict:
        return {
            "text": e.get("text", ""),
            "label": e.get("label", ""),
            "start": e.get("start", 0) + base,
            "end": e.get("end", 0) + base,
        }

    for batch_idx, entities in enumerate(batch_entities_list):
        if not entities:
            continue
        base = batch_starts[batch_idx] if batch_idx < len(batch_starts) else 0
        for e in entities:
            merged.append(add_offset(e, base))
    return merged


def assign_entities_to_sentences(batch_sentences: list[str], batch_entities: list[dict], sep: str = "\n") -> list[tuple[str, list[dict]]]:
    if not batch_sentences:
        return []
    starts = [0]
    for s in batch_sentences:
        starts.append(starts[-1] + len(s) + len(sep))

    def to_sentence_local(e: dict, sent_start: int, sent_end: int) -> dict:
        loc_start = max(0, e.get("start", 0) - sent_start)
        loc_end = min(sent_end - sent_start, e.get("end", 0) - sent_start)
        if loc_end <= loc_start:
            loc_end = loc_start + len(e.get("text", ""))
        return {
            "text": e.get("text", ""),
            "label": e.get("label", ""),
            "start": loc_start,
            "end": loc_end,
        }

    result = [(s, []) for s in batch_sentences]
    for e in batch_entities:
        s_start = e.get("start", 0)
        sent_idx = 0
        for j in range(len(starts) - 1):
            if starts[j] <= s_start < starts[j + 1]:
                sent_idx = j
                break
            if s_start >= starts[j + 1]:
                sent_idx = min(j + 1, len(batch_sentences) - 1)
        if sent_idx >= len(batch_sentences):
            sent_idx = len(batch_sentences) - 1
        sent_start = starts[sent_idx]
        sent_end = starts[sent_idx + 1] - len(sep) if sent_idx + 1 < len(starts) else sent_start + len(batch_sentences[sent_idx])
        result[sent_idx][1].append(to_sentence_local(e, sent_start, sent_end))
    return result
