"""
Milestone 1: 规则认知 Agent (Rule Cognitive Engine)

大模型读取包含嵌套实体的专利术语 JSON，挖掘「词法 / 概念 / 组合」三维规则，
输出结构化规则树（Dict + 可选 NetworkX 图谱）。

与 PAT 工程落地路线图衔接：
- 输入：BIO 或 span 格式的嵌套实体 JSON
- 输出：lexical_rules, concept_rules, composition_rules → 图谱数据结构
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


RULE_EXTRACTION_SYSTEM = """# Role
你是一个顶尖的生物医药与化学领域的语言学专家，专门负责为知识图谱提取「专利术语嵌套构词规则」。

# Task
我将提供一个包含嵌套实体的专利术语 JSON。请你进行「组合泛化」分析，将其拆解为多维度、可复用的构词规则，并严格按照要求的 JSON 格式输出。

# Analysis Dimension (分析维度)
你必须从以下三个维度提取规则，允许规则并发：
1. Lexical Rules (词法规则): 识别出前缀（如「重组」、「分离的」）、词根（如「HCV」、「细胞」）、后缀（如「抑制剂」、「组合物」）。
2. Concept Rules (概念映射): 说明词根属于什么基础实体类型（如 Virus, Cell, Tissue）。
3. Composition Rules (组合规则): 这是核心！用公式表达嵌套逻辑，例如 "[Virus] + [Protein] = [Target]" 或 "[Method] + [Medical_Device] = [Method_of_Use]"。

# Output Format (严格遵循 JSON)
{
  "lexical_rules": [
    {"type": "prefix|suffix|root", "text": "...", "transforms_to": "EntityType"}
  ],
  "concept_rules": [
    {"concept": "EntityType", "examples": ["..."]}
  ],
  "composition_rules": [
    {"formula": "[A] + [B] = [C]", "description": "..."}
  ]
}

只输出一个合法 JSON 对象，不要其他说明。"""

RULE_EXTRACTION_USER = """请对以下专利术语 JSON 进行规则提取，输出 lexical_rules、concept_rules、composition_rules。

输入：
{input_json}
"""


@dataclass
class ExtractedRules:
    """规则提取结果。"""
    lexical_rules: list[dict] = field(default_factory=list)
    concept_rules: list[dict] = field(default_factory=list)
    composition_rules: list[dict] = field(default_factory=list)
    source_text: str = ""
    source_entities: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lexical_rules": self.lexical_rules,
            "concept_rules": self.concept_rules,
            "composition_rules": self.composition_rules,
            "source_text": self.source_text,
            "source_entities": self.source_entities,
        }

    def to_graph_dict(self) -> dict[str, Any]:
        """转为图谱友好的 Dict 结构。"""
        nodes: list[dict] = []
        edges: list[dict] = []
        for r in self.lexical_rules:
            t = r.get("type", "unknown")
            text = r.get("text", "")
            to_type = r.get("transforms_to", "")
            nodes.append({"id": f"lex:{t}:{text}", "type": "lexical", "label": text, "rule_type": t, "transforms_to": to_type})
        for r in self.concept_rules:
            concept = r.get("concept", "")
            examples = r.get("examples", [])
            nodes.append({"id": f"concept:{concept}", "type": "concept", "label": concept, "examples": examples})
        for r in self.composition_rules:
            formula = r.get("formula", "")
            desc = r.get("description", "")
            nodes.append({"id": f"comp:{formula[:50]}", "type": "composition", "label": formula, "description": desc})
            # 从 formula 解析 [A] + [B] = [C] 得到边
            parts = re.findall(r"\[([^\]]+)\]", formula)
            if len(parts) >= 2:
                for i in range(len(parts) - 1):
                    edges.append({"source": parts[i], "target": parts[-1], "formula": formula})
        return {"nodes": nodes, "edges": edges}




def _parse_json_from_content(content: str | None) -> dict | None:
    """从 LLM 返回中解析 JSON。"""
    if not content or not content.strip():
        return None
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
    if start_brace < 0:
        return None
    depth = 0
    for i in range(start_brace, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start_brace : i + 1])
                except json.JSONDecodeError:
                    pass
                break
    return None


def extract_rules_from_json(input_data: dict | str) -> ExtractedRules | None:
    """
    从嵌套实体 JSON 提取构词规则。

    输入格式示例：
    {"text": "HCV NS5B聚合酶抑制剂", "entities": [{"text": "HCV", "label": "Virus"}, ...]}
    或 span 格式：{"context": "...", "spans": [{"text": "...", "label": "...", "start", "end"}, ...]}
    """
    if isinstance(input_data, str):
        try:
            input_data = json.loads(input_data)
        except json.JSONDecodeError:
            return None
    if not isinstance(input_data, dict):
        return None

    # 统一为 {text, entities} 格式
    text = input_data.get("text") or input_data.get("context") or ""
    entities = input_data.get("entities") or input_data.get("spans") or []
    if not text and not entities:
        return None

    # 转为 entities 格式 [{text, label}]
    norm_entities = []
    for e in entities:
        if isinstance(e, dict):
            t = e.get("text") or e.get("label", "")
            lab = e.get("label") or e.get("type", "entity")
            if t or lab:
                norm_entities.append({"text": str(t), "label": str(lab)})
    payload = {"text": text, "entities": norm_entities}

    from llm_client import chat_completion
    user_msg = RULE_EXTRACTION_USER.format(input_json=json.dumps(payload, ensure_ascii=False, indent=2))
    messages = [
        {"role": "system", "content": RULE_EXTRACTION_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    try:
        content = chat_completion(messages, temperature=0, max_tokens=1024, tier="quality")
    except Exception:
        return None

    parsed = _parse_json_from_content(content)
    if not parsed:
        return None

    return ExtractedRules(
        lexical_rules=parsed.get("lexical_rules") or [],
        concept_rules=parsed.get("concept_rules") or [],
        composition_rules=parsed.get("composition_rules") or [],
        source_text=text,
        source_entities=norm_entities,
    )


def rules_to_networkx(rules: ExtractedRules) -> "nx.DiGraph | None":
    """将规则转为 NetworkX 有向图（需 pip install networkx）。"""
    if not HAS_NETWORKX:
        return None
    G = nx.DiGraph()
    gd = rules.to_graph_dict()
    for n in gd.get("nodes", []):
        G.add_node(n.get("id", ""), **{k: v for k, v in n.items() if k != "id"})
    for e in gd.get("edges", []):
        G.add_edge(e.get("source", ""), e.get("target", ""), formula=e.get("formula", ""))
    return G


def run_rule_mining_on_file(
    input_path: Path,
    output_path: Path | None = None,
    limit: int = 0,
) -> list[ExtractedRules]:
    """
    对 JSONL 文件逐行运行规则提取，每行一个 {context, spans} 或 {text, entities}。
    """
    input_path = Path(input_path)
    if not input_path.exists():
        return []
    results: list[ExtractedRules] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit > 0 and len(results) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            r = extract_rules_from_json(obj)
            if r:
                results.append(r)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return results


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="规则认知 Agent：从嵌套实体 JSON 提取构词规则")
    ap.add_argument("--input", type=str, default="train_ready.jsonl", help="输入 JSONL 路径")
    ap.add_argument("--output", type=str, default="artifacts/rule_trees.jsonl", help="输出规则 JSONL 路径")
    ap.add_argument("--limit", type=int, default=0, help="限制处理条数，0 表示全部")
    ap.add_argument("--demo", action="store_true", help="使用示例 JSON 运行单次测试")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parent

    if args.demo:
        demo = {
            "text": "HCV NS5B聚合酶抑制剂",
            "entities": [
                {"text": "HCV", "label": "Virus"},
                {"text": "NS5B聚合酶", "label": "Protein"},
                {"text": "HCV NS5B聚合酶", "label": "Target"},
                {"text": "HCV NS5B聚合酶抑制剂", "label": "Chemical"},
            ],
        }
        print("[demo] 输入:", json.dumps(demo, ensure_ascii=False, indent=2))
        r = extract_rules_from_json(demo)
        if r:
            print("[ok] 规则提取成功:")
            print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
            gd = r.to_graph_dict()
            print("[graph] nodes:", len(gd["nodes"]), "edges:", len(gd["edges"]))
            if HAS_NETWORKX:
                G = rules_to_networkx(r)
                print("[networkx] 节点数:", G.number_of_nodes(), "边数:", G.number_of_edges())
        else:
            print("[fail] 规则提取失败（需配置 GEMINI_API_KEY、DEEPSEEK_API_KEY 或 FAST_MODEL_API_KEY）")
        return

    inp = ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    out = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)

    if not inp.exists():
        print(f"[fail] 输入不存在: {inp}")
        return

    print(f"[info] 规则提取: {inp} -> {out}")
    results = run_rule_mining_on_file(inp, out, limit=args.limit)
    print(f"[ok] 提取 {len(results)} 条规则树 -> {out}")


if __name__ == "__main__":
    main()
