"""论文术语嵌套标注：支持 Gemini、DeepSeek、FAST_MODEL（OpenAI 兼容）。"""
import json
import os
import re
from pathlib import Path

from .schema import PaperAnnotationResult

# 支持 GEMINI_API_KEY（优先）、DEEPSEEK_API_KEY、FAST_MODEL_API_KEY

MAX_INPUT_TOKENS = 10000
# 中英文混合约 2 字符/token，20 句通常不会超 10000 token，此处做上限截断
CHARS_PER_TOKEN = 2


def _truncate_to_max_tokens(text: str, max_tokens: int = MAX_INPUT_TOKENS) -> str:
    """将输入文本截断到合理 token 范围内（按字符数估算）。"""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()

ANNOTATION_SYSTEM = """你是一名专利术语标注专家。对文本中的专利/专业术语做扁平标注：

1. 识别每个专利或专业术语的完整片段，每个术语单独一条，不嵌套。
2. 不需要对术语做类型分类，label 统一填 "term" 即可。
3. 输出且仅输出一个合法 JSON：{"entities": [{"text":"...","label":"term","start":0,"end":0}, ...]}，start/end 为本组文本内的字符偏移（从 0 开始）。"""

BATCH_ANNOTATION_USER = """请对本组句子做专利术语的扁平标注：识别每个术语的 text 及在文中的 start/end。label 一律填 "term"。只输出一个 JSON 对象，不要其他说明。

句子组：
---
{paper_text}
---"""


def _chat(messages: list[dict], max_tokens: int = 8192, tier: str = "performance") -> str | None:
    """统一 chat 调用，支持 Gemini / DeepSeek / FAST_MODEL。tier=performance 用于普通分割/标注。"""
    try:
        from llm_client import chat_completion, get_client_and_model_for_openai_style
    except ImportError:
        return None
    content = chat_completion(messages, temperature=0, max_tokens=max_tokens, tier=tier)
    if content is not None:
        return content
    pair = get_client_and_model_for_openai_style()
    if not pair:
        return None
    from openai import OpenAI
    client, model = pair
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        temperature=0,
        max_tokens=max_tokens,
    )
    if resp.choices:
        c = resp.choices[0].message.content
        if c:
            return c.strip()
    return None


def _parse_json_in_content(content: str | None) -> PaperAnnotationResult | None:
    """从模型返回的 content 中解析 JSON，转为 PaperAnnotationResult。"""
    if not content or not content.strip():
        return None
    s = content.strip()
    # 1. 尝试提取 ```json ... ``` 或 ``` ... ```
    if "```" in s:
        for mark in ("```json", "```"):
            idx = s.find(mark)
            if idx >= 0:
                start = idx + len(mark)
                end = s.find("```", start)
                if end < 0:
                    end = len(s)
                s = s[start:end].strip()
                break
    # 2. 尝试从文本中提取首个 { ... } JSON 块（支持多层嵌套）
    start_brace = s.find("{")
    if start_brace >= 0:
        depth, i = 0, start_brace
        for i in range(start_brace, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    s = s[start_brace : i + 1]
                    break

    def _normalize_entity(e: dict) -> dict:
        return {
            "text": str(e.get("text", "")),
            "label": str(e.get("label", "term"))[:50],
            "start": int(e.get("start", 0)),
            "end": int(e.get("end", 0)),
        }

    try:
        data = json.loads(s)
        if isinstance(data, dict):
            ents = data.get("entities", data.get("entity", []))
        elif isinstance(data, list):
            ents = data
        else:
            return None
        if not isinstance(ents, list):
            return None
        normalized = [_normalize_entity(e) for e in ents if isinstance(e, dict)]
        return PaperAnnotationResult(entities=normalized)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # 尝试修复截断的 JSON（模型输出被 max_tokens 截断时）
    for suffix in ("]", "}", "]}", '"]}', "]}", "}" * 5):
        try:
            s2 = s.rstrip()
            if not s2.endswith("}"):
                s2 = s2 + suffix
            data = json.loads(s2)
            if isinstance(data, dict):
                ents = data.get("entities", data.get("entity", []))
            elif isinstance(data, list):
                ents = data
            else:
                continue
            if isinstance(ents, list):
                normalized = [_normalize_entity(e) for e in ents if isinstance(e, dict)]
                if normalized:
                    return PaperAnnotationResult(entities=normalized)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return None


def _invoke_annotate(
    paper_text: str,
    max_retries: int = 2,
    batch_index: int | None = None,
) -> PaperAnnotationResult | None:
    """调用 API 进行标注。输入控制在 MAX_INPUT_TOKENS 以内，temperature=0。"""
    paper_text = _truncate_to_max_tokens(paper_text)
    user_content = BATCH_ANNOTATION_USER.format(paper_text=paper_text)
    messages = [
        {"role": "system", "content": ANNOTATION_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    last_content: str | None = None
    for attempt in range(max_retries + 1):
        try:
            content = _chat(messages, max_tokens=8192)
            last_content = content
            result = _parse_json_in_content(content)
            if result is not None:
                return result
        except Exception:
            if attempt == max_retries:
                raise
    if os.environ.get("DEBUG_ANNOTATION") and last_content is not None:
        _save_failed_response(batch_index, last_content)
    return None


def _save_failed_response(batch_index: int | None, content: str) -> None:
    """将解析失败的原始返回写入文件，便于排查。"""
    try:
        path = Path(__file__).resolve().parent.parent / "annotation_debug_failures.txt"
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== 批次 {batch_index} 原始返回 ===\n{content}\n")
    except Exception:
        pass


def create_batch_annotation_chain():
    """返回可调用的标注链，兼容 main.py 的 chain.invoke({"paper_text": ...}) 接口。"""

    class Chain:
        def invoke(self, inputs: dict):
            paper_text = inputs.get("paper_text", "")
            batch_index = inputs.get("batch_index")
            return _invoke_annotate(paper_text, batch_index=batch_index)

    return Chain()


# ---------- 为专利术语统一生成语义 label（供 Span 训练格式使用）---------

LABEL_GEN_SYSTEM = """你为专利术语分配简短的英文语义标签。给定术语列表，为每个术语输出一个标签，如 Method, System, Component, Material, Device, Process, Technique 等。只输出一个 JSON 对象，格式：{"术语1": "Label1", "术语2": "Label2", ...}，不要其他说明。"""


def generate_labels_for_terms(terms: list[str]) -> dict[str, str]:
    """用 AI 为所有专利术语统一生成语义 label，返回 {术语: label}。"""
    if not terms:
        return {}
    batch_size = 200
    result: dict[str, str] = {}
    for i in range(0, len(terms), batch_size):
        chunk = terms[i : i + batch_size]
        user_content = "请为以下专利术语分别分配一个简短英文标签：\n" + "\n".join(f"- {t}" for t in chunk)
        try:
            content = _chat(
                [
                    {"role": "system", "content": LABEL_GEN_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=4096,
            )
            if not content:
                continue
            # 解析 JSON
            s = content.strip()
            if "```" in s:
                for mark in ("```json", "```"):
                    idx = s.find(mark)
                    if idx >= 0:
                        start = idx + len(mark)
                        end = s.find("```", start)
                        if end < 0:
                            end = len(s)
                        s = s[start:end].strip()
                        break
            start_brace = s.find("{")
            if start_brace >= 0:
                depth, j = 0, start_brace
                for j in range(start_brace, len(s)):
                    if s[j] == "{":
                        depth += 1
                    elif s[j] == "}":
                        depth -= 1
                        if depth == 0:
                            s = s[start_brace : j + 1]
                            break
            data = json.loads(s)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, str) and k.strip():
                        result[k.strip()] = str(v).strip()[:50]
        except Exception:
            continue
    return result
