import { NextResponse } from "next/server"
import { readFile } from "fs/promises"
import { join } from "path"

/**
 * 读取项目根目录的 knowledge_graph.json（由 patent_agent_pipeline 生成）
 * 当从 3/ 目录运行 next dev 时，cwd 为 3/，故使用 ../knowledge_graph.json
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
        { error: "knowledge_graph.json not found. Run patent_agent_pipeline first." },
        { status: 404 }
      )
    }
    const graph = JSON.parse(data)
    return NextResponse.json(graph)
  } catch (e) {
    console.error("[api/knowledge-graph]", e)
    return NextResponse.json(
      { error: String(e) },
      { status: 500 }
    )
  }
}
