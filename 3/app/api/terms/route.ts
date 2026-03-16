import { NextResponse } from "next/server"
import { readFile } from "fs/promises"
import { join } from "path"

/**
 * 从 knowledge_graph.json 派生术语列表（含证据数量、领域、验证状态）
 */
export async function GET() {
  try {
    const base = process.cwd()
    const paths = [
      join(base, "..", "knowledge_graph.json"),
      join(base, "knowledge_graph.json"),
      join(base, "public", "data", "knowledge_graph.json"),
    ]
    let data: string | null = null
    for (const p of paths) {
      try {
        data = await readFile(p, "utf-8")
        break
      } catch {
        continue
      }
    }
    if (!data) {
      return NextResponse.json(
        { terms: [], total: 0 },
        { status: 200 }
      )
    }
    const graph = JSON.parse(data) as {
      nodes: Array<{ id: string; label: string; type: string; domain?: string; source?: string; url?: string; title?: string; snippet?: string }>
      links: Array<{ source: string; target: string; strength: number }>
    }
    const termNodes = graph.nodes.filter((n) => n.type === "term")
    const linkCount: Record<string, number> = {}
    for (const l of graph.links || []) {
      if (l.source.startsWith("term:")) {
        linkCount[l.source] = (linkCount[l.source] || 0) + 1
      }
    }
    const domainLabels: Record<string, string> = {
      bio: "生物",
      chem: "化学",
      phy: "物理",
      materials: "材料",
      unknown: "未知",
    }
    const terms = termNodes.map((n, i) => ({
      id: n.id,
      term: n.label,
      domain: n.domain || "unknown",
      domainLabel: domainLabels[n.domain || "unknown"] || n.domain || "未知",
      confidence: 0.9,
      verified: (linkCount[n.id] || 0) > 0,
      knowledgeBlocks: linkCount[n.id] || 0,
      paperTitle: "",
    }))
    return NextResponse.json({
      terms,
      total: terms.length,
      verified: terms.filter((t) => t.verified).length,
    })
  } catch (e) {
    console.error("[api/terms]", e)
    return NextResponse.json(
      { error: String(e), terms: [], total: 0 },
      { status: 500 }
    )
  }
}
