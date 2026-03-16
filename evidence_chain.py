"""
证据知识链 (Evidence Knowledge Chain)：多步循环思考框架。

让大模型不仅是"路由"，更是"逻辑推导机"：
1. 动态句块重组与假设生成 (Hypothesis Generation)
2. 关联相似度检索 (Associative Retrieval)
3. 构词规则校验与佐证 (Morphological Verification)
4. 知识块沉淀与模型微调弹药生成 (Knowledge Consolidation)

核心结构：假设 -> 找相似佐证 -> 扣构词规则 -> 形成定论
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# 构词规则：化学前缀/后缀、生物模式
_CHEM_PREFIXES = ("聚", "共聚", "嵌段", "接枝", "超支化", "超", "多", "线型", "支化")
_CHEM_SUFFIXES = ("硅氧烷", "烯", "烷", "醇", "酸", "酯", "胺", "酮", "醚", "苯", "基")
_BIO_PATTERNS = ("蛋白", "基因", "受体", "抗体", "因子", "酶", "细胞")


@dataclass
class EvidenceChainNode:
    """
    证据链节点：结构化输出，供 RoBERTa 等模型微调使用。
    对应「假设 -> 找相似佐证 -> 扣构词规则 -> 形成定论」铁三角。
    """
    hypothesis_segmentation: list[str] = field(default_factory=list)
    associative_evidence: str = ""
    morphological_rule: str = ""
    final_conclusion: str = ""
    confidence_score: float = 0.0
    # 扩展：各词根检索到的知识块摘要，便于审计
    root_snippets: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_segmentation": self.hypothesis_segmentation,
            "associative_evidence": self.associative_evidence,
            "morphological_rule": self.morphological_rule,
            "final_conclusion": self.final_conclusion,
            "confidence_score": self.confidence_score,
            "root_snippets": self.root_snippets,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ------------------------- Step 1: Hypothesis Generation -------------------------


def generate_hypotheses(term: str, domain: str) -> list[list[str]]:
    """
    将长句切分为不同的 N-gram 组合，生成多种分割假设。
    示例：「超支化聚硅氧烷」-> [["超支化", "聚硅氧烷"], ["超", "支化聚硅氧烷"]]
    """
    t = (term or "").strip()
    if not t or len(t) < 3:
        return [[t]] if t else []

    hypotheses: list[list[str]] = []

    if domain == "chem":
        # 化学：优先按前缀+主体拆分
        for prefix in _CHEM_PREFIXES:
            if t.startswith(prefix) and len(t) > len(prefix):
                rest = t[len(prefix):]
                if rest:
                    hypotheses.append([prefix, rest])
        for suffix in _CHEM_SUFFIXES:
            if suffix in t and t != suffix:
                idx = t.find(suffix)
                if idx > 0:
                    before = t[:idx]
                    after = t[idx:]
                    if before and len(before) <= 8:
                        hypotheses.append([before, after])
                elif t.endswith(suffix):
                    before = t[:-len(suffix)]
                    if before:
                        hypotheses.append([before, suffix])
        # 二元切分：在合理位置尝试
        for i in range(1, min(len(t), 8)):
            hypotheses.append([t[:i], t[i:]])

    elif domain == "bio":
        for pat in _BIO_PATTERNS:
            if pat in t and t != pat:
                idx = t.find(pat)
                before = t[:idx]
                after = t[idx + len(pat):]
                if idx > 0 and after:
                    hypotheses.append([before, pat, after])
                elif idx > 0:
                    hypotheses.append([before, pat])
                elif after:
                    hypotheses.append([pat, after])
        for i in range(1, min(len(t), 6)):
            hypotheses.append([t[:i], t[i:]])

    else:
        # phy / unknown: 简单二元切分
        for i in range(1, min(len(t), 8)):
            hypotheses.append([t[:i], t[i:]])

    # 去重并保证完整覆盖
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    for h in hypotheses:
        key = tuple(h)
        if key not in seen and "".join(h) == t:
            seen.add(key)
            out.append(h)
    if not out:
        out = [[t]]
    return out


# ------------------------- Step 2: Associative Retrieval -------------------------


def associative_retrieve(
    roots: list[str],
    domain: str,
    retrieve_fn: Callable[[str], Any],
) -> tuple[str, dict[str, str]]:
    """
    退化搜索：对每个词根单独检索，聚合相似知识块。
    retrieve_fn(term) -> EvidenceChunk | None (或具有 .snippet 的对象)
    """
    combined: list[str] = []
    root_snippets: dict[str, str] = {}
    for r in roots:
        r = (r or "").strip()
        if not r or len(r) < 2:
            continue
        chunk = retrieve_fn(r)
        if chunk is not None and hasattr(chunk, "snippet") and chunk.snippet:
            snip = (chunk.snippet or "")[:400]
            combined.append(f"[{r}]: {snip}")
            root_snippets[r] = snip
        time.sleep(0.15)  # 礼貌延迟
    return "\n\n".join(combined), root_snippets


# ------------------------- Step 3: Morphological Verification -------------------------


def _get_chem_rule(components: list[str]) -> str:
    """化学构词规则描述。"""
    if not components:
        return "无构词拆解"
    pre = [c for c in components if c in _CHEM_PREFIXES]
    suf = [c for c in components if any(s in c for s in _CHEM_SUFFIXES)]
    if pre and suf:
        return f"化学命名：前缀{'+'.join(pre)}修饰主体{'+'.join(suf)}，符合高分子/聚合物构词逻辑"
    if pre:
        return f"化学前缀{'+'.join(pre)}用于修饰聚合物结构"
    if suf:
        return f"化学后缀/基团{'+'.join(suf)}符合命名规范"
    return "构词拆解符合化学命名习惯"


def _get_bio_rule(components: list[str]) -> str:
    """生物构词规则描述。"""
    if not components:
        return "无构词拆解"
    for pat in _BIO_PATTERNS:
        if any(pat in c for c in components):
            return f"生物命名：含{pat}等标准术语，符合靶点/蛋白命名规范"
    return "构词拆解符合生物学术语习惯"


def _get_phy_rule(components: list[str]) -> str:
    """物理构词规则描述。"""
    if not components:
        return "无构词拆解"
    return "物理/工程术语，量纲与单位符合规范"


def morphological_verify(term: str, domain: str, components: list[str]) -> tuple[str, float]:
    """
    将检索到的相似知识块与特定领域命名规则对齐。
    返回 (规则描述, 置信度 0.0-1.0)。
    """
    if not components:
        return ("无拆解", 0.2)
    if domain == "chem":
        rule = _get_chem_rule(components)
        # 有前缀+主体时置信度更高
        has_prefix = any(c in _CHEM_PREFIXES for c in components)
        has_suffix = any(any(s in c for s in _CHEM_SUFFIXES) for c in components)
        conf = 0.5
        if has_prefix and has_suffix:
            conf = 0.65
        elif has_prefix or has_suffix:
            conf = 0.55
        return (rule, conf)
    if domain == "bio":
        rule = _get_bio_rule(components)
        return (rule, 0.5)
    if domain == "phy":
        rule = _get_phy_rule(components)
        return (rule, 0.45)
    return ("通用构词校验", 0.35)


# ------------------------- Step 4: Knowledge Consolidation -------------------------


def _llm_fill_chain(term: str, domain: str, node: EvidenceChainNode) -> EvidenceChainNode:
    """可选：用 LLM 补全 final_conclusion，提升信息密度。"""
    try:
        from llm_client import chat_completion
    except ImportError:
        return node
    comp_str = "+".join(node.hypothesis_segmentation)
    evidence_preview = (node.associative_evidence or "")[:500]
    content = chat_completion(
        messages=[
            {"role": "system", "content": "根据构词拆解与检索佐证，输出一句标准专利术语的最终推导结论（如：超支化聚硅氧烷符合高分子化学命名，为超支化拓扑的聚硅氧烷聚合物）。只输出结论，不要解释。"},
            {"role": "user", "content": f"术语：{term}\n拆解：{comp_str}\n佐证摘要：{evidence_preview}\n构词规则：{node.morphological_rule}"},
        ],
        temperature=0,
        max_tokens=200,
        tier="quality",
    )
    if content:
        node.final_conclusion = content.strip()[:400]
    return node


def build_evidence_chain(
    term: str,
    domain: str,
    retrieve_fn: Callable[[str], Any],
    use_llm_conclusion: bool = True,
) -> EvidenceChainNode:
    """
    主入口：假设 -> 找相似佐证 -> 扣构词规则 -> 形成定论。

    retrieve_fn(term) -> EvidenceChunk | None，由 pipeline 注入（含 Wikipedia/DuckDuckGo/学术 API）。
    """
    t = (term or "").strip()
    if not t:
        return EvidenceChainNode(confidence_score=0.0)

    # Step 1: 假设生成
    hypotheses = generate_hypotheses(t, domain)
    if not hypotheses:
        return EvidenceChainNode(
            hypothesis_segmentation=[t],
            final_conclusion=f"术语「{t}」过短，无法拆分验证。",
            confidence_score=0.2,
        )

    # 优先选择最「合理」的假设：词根数适中、长度均衡
    def score_hypothesis(h: list[str]) -> float:
        if len(h) <= 1:
            return 0.0
        lens = [len(x) for x in h]
        if any(l < 2 for l in lens):
            return 0.3
        return 1.0 / (1 + abs(len(h) - 2))  # 偏好 2 段

    best = max(hypotheses, key=score_hypothesis)
    if len(best) <= 1:
        best = hypotheses[0]

    # Step 2: 关联检索
    associative_evidence, root_snippets = associative_retrieve(best, domain, retrieve_fn)

    # Step 3: 构词规则校验
    morphological_rule, morph_conf = morphological_verify(t, domain, best)

    # Step 4: 知识沉淀
    node = EvidenceChainNode(
        hypothesis_segmentation=best,
        associative_evidence=associative_evidence,
        morphological_rule=morphological_rule,
        final_conclusion="",  # 待 LLM 或规则填充
        confidence_score=morph_conf,
        root_snippets=root_snippets,
    )
    if not node.final_conclusion:
        if associative_evidence:
            node.final_conclusion = f"根据构词拆解「{'+'.join(best)}」及检索佐证，{morphological_rule}，可推导「{t}」为合法术语。"
        else:
            node.final_conclusion = f"构词拆解「{'+'.join(best)}」符合{morphological_rule}，但未找到外部佐证。"
    if use_llm_conclusion and (
        os.environ.get("GEMINI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("FAST_MODEL_API_KEY")
    ):
        node = _llm_fill_chain(t, domain, node)
    return node


def evidence_chain_to_chunk(term: str, node: EvidenceChainNode, source_prefix: str = "chain") -> Any:
    """
    将 EvidenceChainNode 转为 EvidenceChunk，便于写入 evidence_chunks 表。
    snippet 含构词拆解、佐证摘要、构词规则、结论，供后续模型微调使用。
    """
    try:
        from verification_cascade import EvidenceChunk
    except ImportError:
        from dataclasses import dataclass
        @dataclass
        class EvidenceChunk:
            term: str
            source: str
            url: str
            title: str
            snippet: str
            retrieved_at: int
            confidence: float
    parts = [f"构词拆解: {'+'.join(node.hypothesis_segmentation)}"]
    if node.associative_evidence:
        parts.append(f"佐证: {(node.associative_evidence or '')[:300]}…")
    parts.append(f"规则: {node.morphological_rule}")
    parts.append(f"结论: {node.final_conclusion}")
    snippet = " | ".join(parts)[:1000]
    return EvidenceChunk(
        term=term,
        source=f"{source_prefix}_derivation",
        url="",
        title=term,
        snippet=snippet,
        retrieved_at=int(time.time()),
        confidence=node.confidence_score,
    )
