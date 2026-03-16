import { NextResponse } from "next/server"
import { readFile } from "fs/promises"
import { join } from "path"

/**
 * 从 knowledge_graph.json 聚合仪表盘统计
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
      return NextResponse.json({
        papers: 0,
        terms: 0,
        verified: 0,
        knowledgeBlocks: 0,
        domainCounts: {},
      })
    }
    const graph = JSON.parse(data) as {
      nodes: Array<{ type: string; domain?: string }>
      links: Array<{ source: string }>
    }
    const termNodes = graph.nodes.filter((n) => n.type === "term")
    const chunkNodes = graph.nodes.filter((n) => n.type === "knowledge")
    const verifiedTerms = new Set<string>()
    for (const l of graph.links || []) {
      if (l.source.startsWith("term:")) verifiedTerms.add(l.source)
    }
    const domainCounts: Record<string, number> = {}
    for (const n of termNodes) {
      const d = n.domain || "unknown"
      domainCounts[d] = (domainCounts[d] || 0) + 1
    }
    return NextResponse.json({
      papers: 1,
      terms: termNodes.length,
      verified: verifiedTerms.size,
      knowledgeBlocks: chunkNodes.length,
      domainCounts,
    })
  } catch (e) {
    console.error("[api/stats]", e)
    return NextResponse.json(
      { error: String(e) },
      { status: 500 }
    )
  }
}
