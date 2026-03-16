"""
End-to-end pipeline (MVP):
train_ready.jsonl -> domain routing -> Wikipedia (primary) + DuckDuckGo (fallback) evidence retrieval
-> basic verification -> SQLite store -> export enhanced jsonl + knowledge graph + HTML viewers.

Multi-Tier Cascade Verification:
- Tier 0/1: Wikipedia ZH, DuckDuckGo
- Tier 2: PubChem (chem), Europe PMC (bio), ArXiv (phy)
- Tier 3: Cross-lingual (translate -> EN search -> back-translate)
- Tier 4: Rule/LLM derivation (when all external sources fail)

Wikipedia: free, no API key, high academic rigor for chem/bio/phy terms.
DuckDuckGo: fallback when Wikipedia has no entry (e.g. newer patent compound terms).
Requires: ddgs (pip install ddgs) for DuckDuckGo fallback; runs without it (Wikipedia only).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Config switches for cascade tiers (env vars, default True for Tier 2, False for Tier 3/4 if no LLM)
def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default

ENABLE_TIER2_ACADEMIC = _env_bool("ENABLE_TIER2_ACADEMIC", True)
ENABLE_TIER3_CROSS_LINGUAL = _env_bool("ENABLE_TIER3_CROSS_LINGUAL", True)
ENABLE_TIER4_DERIVATION = _env_bool("ENABLE_TIER4_DERIVATION", True)
ENABLE_EVIDENCE_CHAIN = _env_bool("ENABLE_EVIDENCE_CHAIN", True)  # 证据知识链：假设->佐证->构词规则->定论

try:
    from verification_cascade import (
        retrieve_pubchem as _retrieve_pubchem,
        retrieve_europepmc as _retrieve_europepmc,
        retrieve_arxiv as _retrieve_arxiv,
        retrieve_cross_lingual as _retrieve_cross_lingual,
        derive_by_rules as _derive_by_rules,
    )
except ImportError:
    _retrieve_pubchem = _retrieve_europepmc = _retrieve_arxiv = lambda t: None
    _retrieve_cross_lingual = lambda t, d: None
    _derive_by_rules = lambda t, d: None

try:
    from evidence_chain import build_evidence_chain, evidence_chain_to_chunk
except ImportError:
    build_evidence_chain = evidence_chain_to_chunk = None

ROOT = Path(__file__).resolve().parent
TRAIN_READY = ROOT / "train_ready.jsonl"
OUT_ENHANCED = ROOT / "train_spans_enhanced.jsonl"
DB_PATH = ROOT / "knowledge.db"
VIEWER_PATH = ROOT / "viewer.html"
GRAPH_JSON_PATH = ROOT / "knowledge_graph.json"
EVIDENCE_CHAINS_PATH = ROOT / "evidence_chains.jsonl"  # 证据链完整输出，供模型微调


def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


# ------------------------- Domain router (rules baseline) -------------------------


DOMAIN_RULES: list[tuple[str, str]] = [
    ("chem", r"(?:\bmol\b|\bM\b|Markush|马库什|化合物|官能团|羟基|甲基|乙基|丙基|苯|环|取代|溶剂|催化|聚(?:二|三|四)|硅氧烷)"),
    ("bio", r"(?:细胞|蛋白|基因|RNA|DNA|抗体|免疫|受体|通路|炎症|肿瘤|小鼠|人源化|表达载体)"),
    ("phy", r"(?:电离辐射|射线|量子|半导体|光谱|波长|电磁|电阻|热导|力学|材料学)"),
]


def route_domain(text: str) -> tuple[str, float]:
    """
    Very simple rule-based router.
    Returns (domain, confidence).
    """
    if not text:
        return ("unknown", 0.0)
    hits: dict[str, int] = {}
    for domain, pat in DOMAIN_RULES:
        n = len(re.findall(pat, text, flags=re.IGNORECASE))
        if n:
            hits[domain] = n
    if not hits:
        return ("unknown", 0.1)
    best = max(hits.items(), key=lambda kv: kv[1])
    total = sum(hits.values())
    conf = best[1] / max(1, total)
    return (best[0], float(conf))


# ------------------------- Wikipedia retriever (evidence chunks) -------------------------


@dataclass
class EvidenceChunk:
    term: str
    source: str  # e.g. wikipedia_zh
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
    return json.loads(data.decode("utf-8", errors="ignore"))


def wikipedia_zh_summary(term: str) -> EvidenceChunk | None:
    """
    Retrieve zh.wikipedia summary as evidence chunk.
    1) Try direct page summary
    2) If missing, use opensearch to find a title then fetch summary
    """
    t = (term or "").strip()
    if not t:
        return None

    def fetch_by_title(title: str) -> EvidenceChunk | None:
        enc = urllib.parse.quote(title, safe="")
        api = f"https://zh.wikipedia.org/api/rest_v1/page/summary/{enc}"
        try:
            obj = _http_get_json(api)
        except Exception:
            return None
        extract = str(obj.get("extract") or "").strip()
        page_title = str(obj.get("title") or title).strip()
        if not extract:
            return None
        page_url = ""
        content_urls = obj.get("content_urls") or {}
        desktop = content_urls.get("desktop") or {}
        page_url = str(desktop.get("page") or "").strip()
        if not page_url:
            page_url = f"https://zh.wikipedia.org/wiki/{urllib.parse.quote(page_title)}"
        return EvidenceChunk(
            term=t,
            source="wikipedia_zh",
            url=page_url,
            title=page_title,
            snippet=extract[:1200],
            retrieved_at=int(time.time()),
            confidence=0.6,
        )

    # 1) direct
    direct = fetch_by_title(t)
    if direct:
        return direct

    # 2) opensearch
    q = urllib.parse.quote(t, safe="")
    search_url = f"https://zh.wikipedia.org/w/api.php?action=opensearch&search={q}&limit=1&namespace=0&format=json"
    try:
        arr = _http_get_json(search_url)
        if isinstance(arr, list) and len(arr) >= 2 and isinstance(arr[1], list) and arr[1]:
            title = str(arr[1][0])
            return fetch_by_title(title)
    except Exception:
        return None
    return None


# ------------------------- DuckDuckGo fallback (when Wikipedia has no entry) -------------------------


def duckduckgo_fallback(term: str) -> EvidenceChunk | None:
    """
    Fallback evidence retrieval via DuckDuckGo search.
    When Wikipedia has no entry for a term (e.g. newer patent compound terms),
    use search result snippets to verify the term appears as a whole.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return None
    t = (term or "").strip()
    if not t:
        return None
    try:
        results = DDGS().text(t, max_results=5)
    except Exception:
        return None
    for r in results or []:
        if not isinstance(r, dict):
            continue
        snippet = (r.get("body") or r.get("snippet") or "").strip()
        if not snippet or len(snippet) < 10:
            continue
        if t not in snippet:
            continue
        return EvidenceChunk(
            term=t,
            source="duckduckgo",
            url=str(r.get("href", "")),
            title=str(r.get("title", "")),
            snippet=snippet[:800],
            retrieved_at=int(time.time()),
            confidence=0.45,
        )
    return None


def retrieve_evidence_for_term(
    term: str,
    persist_chain_path: Path | None = None,
) -> EvidenceChunk | None:
    """
    Full cascade: Wikipedia -> DuckDuckGo -> Tier 2 academic -> Tier 3 cross-lingual -> Tier 4 derivation.
    Reusable for convert_with_evidence.py and other callers.
    When persist_chain_path is set and evidence chain is used, append chain to that file.
    """
    t = (term or "").strip()
    if not t:
        return None
    domain, _ = route_domain(t)
    chunk = wikipedia_zh_summary(t)
    if chunk is None:
        chunk = duckduckgo_fallback(t)
    if chunk is None and ENABLE_TIER2_ACADEMIC:
        time.sleep(0.2)
        if domain == "chem":
            chunk = _retrieve_pubchem(t)
        elif domain == "bio":
            chunk = _retrieve_europepmc(t)
        elif domain == "phy":
            chunk = _retrieve_arxiv(t)
    if chunk is None and ENABLE_TIER3_CROSS_LINGUAL:
        chunk = _retrieve_cross_lingual(t, domain)
    if chunk is None and ENABLE_TIER4_DERIVATION:
        if ENABLE_EVIDENCE_CHAIN and build_evidence_chain and evidence_chain_to_chunk:
            def _retrieve_for_chain(term: str):
                c = wikipedia_zh_summary(term)
                if c is None:
                    c = duckduckgo_fallback(term)
                if c is None and ENABLE_TIER2_ACADEMIC:
                    time.sleep(0.2)
                    if domain == "chem":
                        c = _retrieve_pubchem(term)
                    elif domain == "bio":
                        c = _retrieve_europepmc(term)
                    elif domain == "phy":
                        c = _retrieve_arxiv(term)
                if c is None and ENABLE_TIER3_CROSS_LINGUAL:
                    c = _retrieve_cross_lingual(term, domain)
                return c
            chain_node = build_evidence_chain(t, domain, _retrieve_for_chain, use_llm_conclusion=True)
            chunk = evidence_chain_to_chunk(t, chain_node, source_prefix="chain")
            if persist_chain_path:
                try:
                    with open(persist_chain_path, "a", encoding="utf-8") as ef:
                        ef.write(json.dumps({"term": t, "domain": domain, **chain_node.to_dict()}, ensure_ascii=False) + "\n")
                except Exception:
                    pass
        else:
            chunk = _derive_by_rules(t, domain)
    return chunk


# ------------------------- SQLite store -------------------------


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        create table if not exists runs (
          run_id text primary key,
          created_at integer not null,
          dataset_version text not null,
          input_file text not null
        )
        """
    )
    conn.execute(
        """
        create table if not exists terms (
          term_id integer primary key autoincrement,
          term_text text not null,
          domain text,
          domain_conf real,
          created_at integer not null,
          unique(term_text)
        )
        """
    )
    conn.execute(
        """
        create table if not exists evidence_chunks (
          chunk_id integer primary key autoincrement,
          term_text text not null,
          source text not null,
          url text not null,
          title text,
          snippet text,
          retrieved_at integer not null,
          confidence real,
          unique(term_text, source, url)
        )
        """
    )
    conn.commit()


def upsert_term(conn: sqlite3.Connection, term: str, domain: str, conf: float) -> None:
    now = int(time.time())
    conn.execute(
        """
        insert into terms(term_text, domain, domain_conf, created_at)
        values(?,?,?,?)
        on conflict(term_text) do update set
          domain=excluded.domain,
          domain_conf=excluded.domain_conf
        """,
        (term, domain, conf, now),
    )


def insert_chunk(conn: sqlite3.Connection, c: EvidenceChunk) -> None:
    conn.execute(
        """
        insert into evidence_chunks(term_text, source, url, title, snippet, retrieved_at, confidence)
        values(?,?,?,?,?,?,?)
        on conflict(term_text, source, url) do update set
          title=excluded.title,
          snippet=excluded.snippet,
          retrieved_at=excluded.retrieved_at,
          confidence=excluded.confidence
        """,
        (c.term, c.source, c.url, c.title, c.snippet, c.retrieved_at, c.confidence),
    )


# ------------------------- Knowledge graph export -------------------------


def _copy_to_frontend(graph_path: Path) -> None:
    """复制 knowledge_graph.json 到 3/public/data 供 Next.js 读取"""
    dest = ROOT / "3" / "public" / "data" / "knowledge_graph.json"
    if not graph_path.exists() or not (ROOT / "3").exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    try:
        shutil.copy2(graph_path, dest)
        print(f"[ok] frontend: {dest}")
    except Exception as e:
        print(f"[warn] copy to frontend failed: {e}")


def export_graph_json(conn: sqlite3.Connection, out_path: Path) -> None:
    """
    Export terms + evidence_chunks as a graph (nodes + links) for visualization.
    Nodes: term nodes (type=term) and chunk nodes (type=knowledge).
    Links: term -> chunk (evidence association).
    """
    nodes: list[dict] = []
    links: list[dict] = []
    node_ids: set[str] = set()

    def add_node(nid: str, label: str, ntype: str, **extra: Any) -> None:
        if nid in node_ids:
            return
        node_ids.add(nid)
        n = {"id": nid, "label": label, "type": ntype, **extra}
        nodes.append(n)

    rows = conn.execute(
        "select term_text, domain from terms order by term_text"
    ).fetchall()
    for term_text, domain in rows or []:
        tid = f"term:{term_text}"
        add_node(tid, term_text, "term", domain=domain or "unknown")

    chunk_rows = conn.execute(
        """
        select chunk_id, term_text, source, url, title, snippet, confidence
        from evidence_chunks
        order by chunk_id
        """
    ).fetchall()
    for chunk_id, term_text, source, url, title, snippet, confidence in chunk_rows or []:
        cid = f"chunk:{chunk_id}"
        label = f"{source}: {(title or '')[:40]}"
        add_node(cid, label, "knowledge", source=source, url=url or "", title=title or "", snippet=(snippet or "")[:200])
        tid = f"term:{term_text}"
        add_node(tid, term_text, "term", domain="unknown")
        links.append({"source": tid, "target": cid, "strength": float(confidence or 0.5)})

    out = {"nodes": nodes, "links": links}
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------- Export + Viewer -------------------------


def export_enhanced(records: list[dict], term_to_chunk: dict[str, EvidenceChunk], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            sent = rec.get("sentence") or ""
            ents = rec.get("entities") or []
            spans = []
            for e in ents:
                if not isinstance(e, dict):
                    continue
                spans.append(
                    {
                        "start": int(e.get("start", 0)),
                        "end": int(e.get("end", 0)),
                        "label": str(e.get("label", "term")),
                        "text": str(e.get("text", "")),
                        "evidence": None,
                    }
                )
            # attach evidence by exact term match
            for s in spans:
                t = (s.get("text") or "").strip()
                c = term_to_chunk.get(t)
                if c:
                    s["evidence"] = {
                        "source": c.source,
                        "url": c.url,
                        "title": c.title,
                        "snippet": c.snippet,
                        "confidence": c.confidence,
                        "retrieved_at": c.retrieved_at,
                    }
            sample = {"context": sent, "spans": spans}
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_viewer_html(out_path: Path) -> None:
    """
    Minimal offline viewer for:
    - train_spans_enhanced.jsonl
    """
    html = r"""<!doctype html>
<meta charset="utf-8">
<title>Patent Term Dataset Viewer</title>
<style>
  :root { --bg:#0b0f14; --fg:#e7eef7; --muted:#9bb0c5; --card:#121923; --accent:#5eead4; --warn:#fbbf24; --bad:#fb7185; }
  body { margin:0; font-family: ui-sans-serif, system-ui, Segoe UI, Arial; background:var(--bg); color:var(--fg); }
  header { padding:16px 20px; border-bottom:1px solid #1f2a37; position:sticky; top:0; background:rgba(11,15,20,.92); backdrop-filter: blur(8px); }
  main { display:grid; grid-template-columns: 360px 1fr; gap:16px; padding:16px 20px; }
  .card { background:var(--card); border:1px solid #1f2a37; border-radius:14px; padding:12px; }
  .muted { color:var(--muted); }
  textarea { width:100%; height:120px; background:#0f1622; color:var(--fg); border:1px solid #233042; border-radius:12px; padding:10px; }
  button { background:linear-gradient(135deg, #0ea5e9, #14b8a6); border:none; color:#001018; padding:10px 12px; border-radius:12px; font-weight:700; cursor:pointer; }
  button.secondary { background:#0f1622; color:var(--fg); border:1px solid #233042; }
  .list { max-height: 68vh; overflow:auto; padding-right:4px; }
  .item { padding:10px; border-radius:12px; border:1px solid transparent; cursor:pointer; }
  .item:hover { border-color:#233042; background:#0f1622; }
  .item.active { border-color: rgba(94,234,212,.5); background: rgba(94,234,212,.06); }
  .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; margin-left:6px; border:1px solid #2a3b52; color:var(--muted); }
  .hl { padding:0 3px; border-radius:6px; background: rgba(94,234,212,.18); border:1px solid rgba(94,234,212,.35); }
  .hl[data-has-ev="1"]{ background: rgba(14,165,233,.18); border-color: rgba(14,165,233,.35);}
  .ev { margin-top:10px; padding:10px; border-radius:12px; background:#0f1622; border:1px solid #233042; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
</style>
<header>
  <div style="display:flex; gap:12px; align-items:center; justify-content:space-between;">
    <div>
      <div style="font-weight:900; letter-spacing:.2px;">Dataset Viewer</div>
      <div class="muted" style="font-size:13px;">Load <code>train_spans_enhanced.jsonl</code> content below (paste) to inspect spans & evidence.</div>
    </div>
    <div style="display:flex; gap:10px;">
      <button id="loadBtn">Load</button>
      <button id="demoBtn" class="secondary">Demo</button>
    </div>
  </div>
</header>
<main>
  <section class="card">
    <div style="font-weight:800; margin-bottom:8px;">Input JSONL</div>
    <textarea id="inp" placeholder="Paste train_spans_enhanced.jsonl here..."></textarea>
    <div class="muted" style="font-size:12px; margin-top:8px;">Tip: open the jsonl file, copy all, paste here, click Load.</div>
    <hr style="border:none;border-top:1px solid #1f2a37; margin:12px 0;">
    <div style="font-weight:800; margin-bottom:8px;">Samples</div>
    <div id="list" class="list"></div>
  </section>
  <section class="card">
    <div id="detailTitle" style="font-weight:900; font-size:16px;">Select a sample</div>
    <div id="detailMeta" class="muted" style="font-size:12px; margin-top:6px;"></div>
    <div id="context" style="margin-top:12px; line-height:1.9;"></div>
    <div id="spans" style="margin-top:14px;"></div>
  </section>
</main>
<script>
let samples = [];
let active = -1;

function esc(s){ return (s||'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;'); }
function parseJSONL(text){
  const lines = (text||'').split(/\\r?\\n/).map(l=>l.trim()).filter(Boolean);
  const out=[];
  for(const ln of lines){
    try{ out.push(JSON.parse(ln)); }catch(e){}
  }
  return out;
}

function renderList(){
  const el = document.getElementById('list');
  el.innerHTML = '';
  samples.forEach((s,i)=>{
    const div = document.createElement('div');
    div.className = 'item' + (i===active ? ' active':'');
    const nsp = (s.spans||[]).length;
    div.innerHTML = `<div style="font-weight:800;">Sample ${i+1}<span class="badge">${nsp} spans</span></div>` +
      `<div class="muted" style="font-size:12px; margin-top:4px;">${esc((s.context||'').slice(0,90))}${(s.context||'').length>90?'…':''}</div>`;
    div.onclick = ()=>{ active=i; renderList(); renderDetail(); };
    el.appendChild(div);
  });
}

function highlightContext(ctx, spans){
  const arr = (spans||[]).slice().sort((a,b)=> (a.start-b.start) || (a.end-b.end));
  let out = '';
  let pos = 0;
  for(const sp of arr){
    const st = Math.max(0, sp.start|0), ed = Math.max(st, sp.end|0);
    if(st > pos) out += esc(ctx.slice(pos, st));
    const hasEv = sp.evidence ? 1 : 0;
    const seg = ctx.slice(st, ed);
    out += `<span class="hl" data-has-ev="${hasEv}" title="${esc(sp.label||'term')}">${esc(seg)}</span>`;
    pos = ed;
  }
  if(pos < ctx.length) out += esc(ctx.slice(pos));
  return out;
}

function renderDetail(){
  if(active < 0 || active >= samples.length) return;
  const s = samples[active];
  document.getElementById('detailTitle').textContent = `Sample ${active+1}`;
  document.getElementById('detailMeta').textContent = `spans=${(s.spans||[]).length}`;
  const ctx = s.context || '';
  document.getElementById('context').innerHTML = highlightContext(ctx, s.spans||[]);
  const spEl = document.getElementById('spans');
  spEl.innerHTML = '';
  (s.spans||[]).forEach((sp, idx)=>{
    const d = document.createElement('div');
    d.className = 'ev';
    const ev = sp.evidence;
    d.innerHTML = `<div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">` +
      `<div><div style="font-weight:900;">${esc(sp.text)} <span class="badge">${esc(sp.label||'term')}</span></div>` +
      `<div class="muted" style="font-size:12px;margin-top:3px;">[${sp.start}, ${sp.end})</div></div>` +
      `<div class="muted" style="font-size:12px;">#${idx+1}</div>` +
      `</div>`;
    if(ev){
      d.innerHTML += `<div style="margin-top:8px;font-weight:800;">Evidence <span class="badge">${esc(ev.source||'')}</span></div>` +
        `<div class="muted" style="font-size:12px;margin-top:4px;">${esc(ev.title||'')}</div>` +
        `<div style="margin-top:6px;">${esc((ev.snippet||'').slice(0,280))}${(ev.snippet||'').length>280?'…':''}</div>` +
        `<div style="margin-top:8px;"><a href="${esc(ev.url||'#')}" target="_blank" rel="noreferrer">Open source</a></div>`;
    }else{
      d.innerHTML += `<div class="muted" style="margin-top:8px;">No evidence attached.</div>`;
    }
    spEl.appendChild(d);
  });
}

document.getElementById('loadBtn').onclick = ()=>{
  samples = parseJSONL(document.getElementById('inp').value);
  active = samples.length ? 0 : -1;
  renderList(); renderDetail();
};

document.getElementById('demoBtn').onclick = ()=>{
  const demo = {
    context: "细胞是生命活动的基本结构和功能单位。",
    spans: [
      {start:0,end:2,label:"term",text:"细胞",evidence:{source:"wikipedia_zh",url:"https://zh.wikipedia.org/wiki/%E7%BB%86%E8%83%9E",title:"细胞",snippet:"细胞是所有已知生物的结构与功能的基本单位。",confidence:0.6,retrieved_at:0}}
    ]
  };
  document.getElementById('inp').value = JSON.stringify(demo);
};
</script>
"""
    out_path.write_text(html, encoding="utf-8")


# ------------------------- Pipeline runner -------------------------


def load_train_ready(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def collect_unique_terms(records: list[dict]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for rec in records:
        for e in rec.get("entities") or []:
            if not isinstance(e, dict):
                continue
            t = (e.get("text") or "").strip()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t)
    return out


def main() -> None:
    if not TRAIN_READY.exists():
        print(f"[fail] missing: {TRAIN_READY}")
        return

    records = load_train_ready(TRAIN_READY)
    if not records:
        print("[fail] no records")
        return

    # dataset version: based on content hash (stable)
    dataset_version = _sha1_text(TRAIN_READY.read_text(encoding="utf-8"))
    run_id = f"run_{int(time.time())}_{dataset_version[:8]}"
    print(f"[info] records={len(records)} dataset_version={dataset_version[:12]} run_id={run_id}")

    terms = collect_unique_terms(records)
    print(f"[info] unique terms={len(terms)}")

    if ENABLE_EVIDENCE_CHAIN and EVIDENCE_CHAINS_PATH:
        EVIDENCE_CHAINS_PATH.write_text("", encoding="utf-8")  # 每轮清空

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    conn.execute(
        "insert into runs(run_id, created_at, dataset_version, input_file) values(?,?,?,?)",
        (run_id, int(time.time()), dataset_version, str(TRAIN_READY)),
    )
    conn.commit()

    term_to_chunk: dict[str, EvidenceChunk] = {}
    retrieved = 0
    for idx, t in enumerate(terms, 1):
        domain, conf = route_domain(t)
        upsert_term(conn, t, domain, conf)

        # retrieval: full cascade via retrieve_evidence_for_term
        chunk = retrieve_evidence_for_term(t, persist_chain_path=EVIDENCE_CHAINS_PATH if ENABLE_EVIDENCE_CHAIN else None)
        if chunk:
            insert_chunk(conn, chunk)
            term_to_chunk[t] = chunk
            retrieved += 1

        if idx % 20 == 0:
            conn.commit()
            print(f"[progress] {idx}/{len(terms)} retrieved={retrieved}")

        # polite delay
        time.sleep(0.05)

    conn.commit()
    export_graph_json(conn, GRAPH_JSON_PATH)
    conn.close()
    print(f"[ok] db updated: {DB_PATH} evidence={retrieved}")
    print(f"[ok] graph: {GRAPH_JSON_PATH}")
    # 复制到 3/public/data 供 Next.js 前端读取
    _copy_to_frontend(GRAPH_JSON_PATH)

    export_enhanced(records, term_to_chunk, OUT_ENHANCED)
    write_viewer_html(VIEWER_PATH)
    print(f"[ok] export: {OUT_ENHANCED}")
    print(f"[ok] viewer: {VIEWER_PATH} (open in browser)")
    print(f"[ok] graph viewer: graph_viewer.html (load {GRAPH_JSON_PATH})")
    if ENABLE_EVIDENCE_CHAIN and EVIDENCE_CHAINS_PATH.exists() and EVIDENCE_CHAINS_PATH.stat().st_size > 0:
        n_lines = sum(1 for _ in open(EVIDENCE_CHAINS_PATH, encoding="utf-8"))
        print(f"[ok] evidence chains: {EVIDENCE_CHAINS_PATH} ({n_lines} chains for fine-tuning)")


if __name__ == "__main__":
    main()

