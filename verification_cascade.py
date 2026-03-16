"""
Multi-Tier Cascade Verification: Tier 2 (academic APIs), Tier 3 (cross-lingual), Tier 4 (derivation).
Evidence sources: pubchem, europepmc, arxiv, wikipedia_en, derivation_chem/bio/phy.
"""

from __future__ import annotations

import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

# EvidenceChunk is defined in patent_agent_pipeline; we import for type hints
# but we'll return a simple dict-like structure that the pipeline can convert


@dataclass
class EvidenceChunk:
    term: str
    source: str
    url: str
    title: str
    snippet: str
    retrieved_at: int
    confidence: float


def _http_get_json(url: str, timeout_s: int = 15) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "patent-agent-pipeline/1.0 (edu project; contact: local)",
            "Accept": "application/json",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    import json
    return json.loads(data.decode("utf-8", errors="ignore"))


def _http_get_bytes(url: str, timeout_s: int = 15) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "patent-agent-pipeline/1.0 (edu project; contact: local)"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


# ------------------------- Tier 2: Academic APIs -------------------------


def retrieve_pubchem(term: str) -> EvidenceChunk | None:
    """
    PubChem PUG REST: compound by name.
    Returns EvidenceChunk with IUPAC name, formula, etc. or None.
    """
    t = (term or "").strip()
    if not t:
        return None
    enc = urllib.parse.quote(t, safe="")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{enc}/JSON"
    try:
        obj = _http_get_json(url)
    except Exception:
        return None
    compounds = obj.get("PC_Compounds") or []
    if not compounds:
        return None
    comp = compounds[0]
    props = comp.get("props") or []
    parts: list[str] = []
    iupac = ""
    formula = ""
    mw = ""
    for p in props:
        urn = p.get("urn") or {}
        label = (urn.get("label") or "").strip()
        name = (urn.get("name") or "").strip()
        val = p.get("value") or {}
        if label == "IUPAC Name" and (name == "Preferred" or name == "Traditional" or not name):
            s = val.get("sval") or ""
            if s and not iupac:
                iupac = s
        elif label == "Molecular Formula":
            formula = val.get("sval") or ""
        elif label == "Molecular Weight":
            mw = val.get("sval") or ""
    if iupac:
        parts.append(f"IUPAC: {iupac}")
    if formula:
        parts.append(f"分子式: {formula}")
    if mw:
        parts.append(f"分子量: {mw}")
    if not parts:
        parts.append(f"PubChem 收录该化合物: {t}")
    cid = ""
    if "id" in comp:
        id_obj = comp.get("id") or {}
        cid_obj = id_obj.get("id") or {}
        cid = str(cid_obj.get("cid", ""))
    page_url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" if cid else url
    return EvidenceChunk(
        term=t,
        source="pubchem",
        url=page_url,
        title=t,
        snippet="; ".join(parts)[:800],
        retrieved_at=int(time.time()),
        confidence=0.55,
    )


def retrieve_europepmc(term: str) -> EvidenceChunk | None:
    """
    Europe PMC REST API. No API key required.
    Returns first matching abstract as snippet.
    """
    t = (term or "").strip()
    if not t:
        return None
    enc = urllib.parse.quote(t, safe="")
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={enc}&format=json&pageSize=3&resultType=core"
    try:
        obj = _http_get_json(url)
    except Exception:
        return None
    results = (obj.get("resultList") or {}).get("result") or []
    for r in results:
        if not isinstance(r, dict):
            continue
        abstract = (r.get("abstractText") or "").strip()
        title = (r.get("title") or "").strip()
        if not abstract and not title:
            continue
        snippet = abstract or title
        pmid = r.get("pmid") or r.get("id") or ""
        page_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}" if pmid else ""
        if not page_url:
            doi = r.get("doi") or ""
            page_url = f"https://doi.org/{doi}" if doi else ""
        return EvidenceChunk(
            term=t,
            source="europepmc",
            url=page_url or "https://europepmc.org",
            title=title[:200] or t,
            snippet=snippet[:800],
            retrieved_at=int(time.time()),
            confidence=0.5,
        )
    return None


def retrieve_arxiv(term: str) -> EvidenceChunk | None:
    """
    ArXiv API. Returns first matching summary/title.
    """
    t = (term or "").strip()
    if not t:
        return None
    enc = urllib.parse.quote(f"all:{t}", safe="")
    url = f"http://export.arxiv.org/api/query?search_query={enc}&max_results=3"
    try:
        raw = _http_get_bytes(url)
        root = ET.fromstring(raw)
    except Exception:
        return None
    ns = "http://www.w3.org/2005/Atom"
    entries = root.findall(f".//{{{ns}}}entry")
    for entry in entries:
        title_el = entry.find(f"{{{ns}}}title")
        summary_el = entry.find(f"{{{ns}}}summary")
        link_el = next((l for l in entry.findall(f"{{{ns}}}link") if l.get("rel") == "alternate"), None)
        title = (title_el.text or "").strip() if title_el is not None else ""
        summary = (summary_el.text or "").strip() if summary_el is not None else ""
        snippet = summary or title
        if not snippet:
            continue
        page_url = ""
        if link_el is not None:
            page_url = link_el.get("href") or ""
        return EvidenceChunk(
            term=t,
            source="arxiv",
            url=page_url or "https://arxiv.org",
            title=title[:200] or t,
            snippet=snippet[:800],
            retrieved_at=int(time.time()),
            confidence=0.5,
        )
    return None


# ------------------------- Tier 3: Cross-lingual -------------------------


def _llm_translate_to_en(text: str) -> str | None:
    """Translate Chinese term to English using LLM. Returns None if no LLM available."""
    try:
        from llm_client import chat_completion
    except ImportError:
        return None
    content = chat_completion(
        messages=[
            {"role": "system", "content": "Translate the given Chinese scientific/technical term to English. Output only the English term, no explanation."},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=128,
        tier="performance",
    )
    return (content or "").strip()[:200] if content else None


def _llm_translate_to_zh(text: str) -> str | None:
    """Translate English snippet to Chinese using LLM."""
    try:
        from llm_client import chat_completion
    except ImportError:
        return None
    content = chat_completion(
        messages=[
            {"role": "system", "content": "将以下英文科学/技术内容归纳翻译为1-2句中文，保持专业术语准确。只输出中文，不要解释。"},
            {"role": "user", "content": text[:1500]},
        ],
        temperature=0,
        max_tokens=512,
    )
    return (content or "").strip()[:600] if content else None


def _wikipedia_en_summary(term: str) -> EvidenceChunk | None:
    """English Wikipedia summary."""
    t = (term or "").strip()
    if not t:
        return None
    enc = urllib.parse.quote(t, safe="")
    api = f"https://en.wikipedia.org/api/rest_v1/page/summary/{enc}"
    try:
        obj = _http_get_json(api)
    except Exception:
        return None
    extract = str(obj.get("extract") or "").strip()
    if not extract:
        return None
    page_title = str(obj.get("title") or t).strip()
    content_urls = obj.get("content_urls") or {}
    desktop = content_urls.get("desktop") or {}
    page_url = str(desktop.get("page") or "").strip()
    if not page_url:
        page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page_title)}"
    return EvidenceChunk(
        term=t,
        source="wikipedia_en",
        url=page_url,
        title=page_title,
        snippet=extract[:1200],
        retrieved_at=int(time.time()),
        confidence=0.45,
    )


def retrieve_cross_lingual(term: str, domain: str) -> EvidenceChunk | None:
    """
    Tier 3: Translate term to EN -> search EN sources -> translate snippet back to ZH.
    """
    t = (term or "").strip()
    if not t:
        return None
    en_term = _llm_translate_to_en(t)
    if not en_term:
        return None
    chunk = _wikipedia_en_summary(en_term)
    if chunk is None and domain == "bio":
        chunk = retrieve_europepmc(en_term)
        if chunk:
            chunk = EvidenceChunk(term=t, source="europepmc_en", url=chunk.url, title=chunk.title, snippet=chunk.snippet, retrieved_at=chunk.retrieved_at, confidence=0.45)
    if chunk is None and domain == "phy":
        chunk = retrieve_arxiv(en_term)
        if chunk:
            chunk = EvidenceChunk(term=t, source="arxiv_en", url=chunk.url, title=chunk.title, snippet=chunk.snippet, retrieved_at=chunk.retrieved_at, confidence=0.45)
    if chunk is None and domain == "chem":
        chunk = retrieve_pubchem(en_term)
        if chunk:
            chunk = EvidenceChunk(term=t, source="pubchem_en", url=chunk.url, title=chunk.title, snippet=chunk.snippet, retrieved_at=chunk.retrieved_at, confidence=0.45)
    if chunk is None:
        return None
    # Ensure term is original (Chinese)
    chunk = EvidenceChunk(term=t, source=chunk.source, url=chunk.url, title=chunk.title, snippet=chunk.snippet, retrieved_at=chunk.retrieved_at, confidence=0.45)
    zh_snippet = _llm_translate_to_zh(chunk.snippet)
    if zh_snippet:
        chunk = EvidenceChunk(term=t, source=chunk.source, url=chunk.url, title=chunk.title, snippet=zh_snippet, retrieved_at=chunk.retrieved_at, confidence=0.45)
    return chunk


# ------------------------- Tier 4: Derivation -------------------------


# Rule-based chemical decomposition (prefix + group + monomer patterns)
_CHEM_PREFIXES = ("聚", "共聚", "嵌段", "接枝", "超", "多")
_CHEM_SUFFIXES = ("烯", "烷", "醇", "酸", "酯", "胺", "酮", "醚", "苯", "基")


def _decompose_chem_rule(term: str) -> list[str] | None:
    """Simple rule-based decomposition for chemical terms."""
    t = (term or "").strip()
    if not t or len(t) < 2:
        return None
    parts: list[str] = []
    rest = t
    for p in _CHEM_PREFIXES:
        if rest.startswith(p):
            parts.append(p)
            rest = rest[len(p):]
            break
    if not rest:
        return parts if parts else None
    for s in _CHEM_SUFFIXES:
        if s in rest and len(rest) > 1:
            idx = rest.find(s)
            if idx > 0:
                before = rest[:idx]
                if before and len(before) <= 6:
                    parts.append(before + s)
                    rest = rest[idx + len(s):]
                    break
            elif rest.endswith(s):
                parts.append(rest)
                rest = ""
                break
    if rest and len(rest) <= 8:
        parts.append(rest)
    return parts if parts else None


def _decompose_bio_rule(term: str) -> list[str] | None:
    """Simple rule-based decomposition for bio terms."""
    t = (term or "").strip()
    if not t or len(t) < 2:
        return None
    parts: list[str] = []
    if "蛋白" in t:
        idx = t.find("蛋白")
        if idx > 0:
            parts.append(t[:idx])
        parts.append("蛋白")
        if idx + 2 < len(t):
            parts.append(t[idx + 2:])
        return parts if parts else None
    if "基因" in t:
        idx = t.find("基因")
        parts.append(t[:idx] if idx > 0 else "")
        parts.append("基因")
        if idx + 2 < len(t):
            parts.append(t[idx + 2:])
        return [p for p in parts if p]
    return None


def _decompose_phy_rule(term: str) -> list[str] | None:
    """Simple rule-based decomposition for physics terms."""
    t = (term or "").strip()
    if not t or len(t) < 2:
        return None
    return [t] if len(t) <= 10 else None


def _llm_derive_snippet(term: str, domain: str, components: list[str]) -> str | None:
    """Use LLM to generate a short derivation conclusion."""
    try:
        from llm_client import chat_completion
    except ImportError:
        return None
    domain_hint = {"chem": "化学", "bio": "生物", "phy": "物理"}.get(domain, "专业")
    comp_str = "、".join(components) if components else term
    content = chat_completion(
        messages=[
            {"role": "system", "content": f"你是一名{domain_hint}领域专家。根据术语构词拆解，生成一句简短结论（如符合命名规范、推测类型等）。不编造文献来源，仅做合法性自证。输出1句话，不要解释。"},
            {"role": "user", "content": f"术语：{term}\n构词拆解：{comp_str}\n请生成一句短结论。"},
        ],
        temperature=0,
        max_tokens=128,
        tier="quality",
    )
    return (content or "").strip()[:400] if content else None


def derive_by_rules(term: str, domain: str) -> EvidenceChunk | None:
    """
    Tier 4: Rule-based decomposition + optional LLM conclusion.
    source: derivation_chem / derivation_bio / derivation_phy
    confidence: 0.35
    """
    t = (term or "").strip()
    if not t:
        return None
    components: list[str] = []
    if domain == "chem":
        components = _decompose_chem_rule(t) or []
    elif domain == "bio":
        components = _decompose_bio_rule(t) or []
    elif domain == "phy":
        components = _decompose_phy_rule(t) or []
    if not components:
        components = [t]
    snippet = _llm_derive_snippet(t, domain, components)
    if not snippet:
        snippet = f"符合{domain}领域命名规范，构词拆解: {'+'.join(components)}。"
    source = f"derivation_{domain}" if domain in ("chem", "bio", "phy") else "derivation_unknown"
    return EvidenceChunk(
        term=t,
        source=source,
        url="",
        title=t,
        snippet=snippet[:600],
        retrieved_at=int(time.time()),
        confidence=0.35,
    )
