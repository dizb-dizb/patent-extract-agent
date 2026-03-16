"use client"

import { useState, useEffect } from "react"
import { DashboardLayout } from "@/components/layout"
import { KnowledgeGraphViewer } from "@/components/knowledge-graph/graph-viewer"
import { KnowledgeBlockPanel } from "@/components/knowledge-graph/block-panel"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { 
  Search, 
  ZoomIn, 
  ZoomOut, 
  Maximize2, 
  RefreshCw,
  Download,
  Filter
} from "lucide-react"

interface GraphData {
  nodes: Array<{ id: string; label: string; type: string; domain?: string; source?: string; url?: string; title?: string; snippet?: string }>
  links: Array<{ source: string; target: string; strength: number }>
}

export default function KnowledgeGraphPage() {
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [zoom, setZoom] = useState(1)
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch("/api/knowledge-graph")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.statusText))))
      .then(setGraphData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <DashboardLayout>
      <div className="grid h-[calc(100vh-8rem)] gap-6 lg:grid-cols-4">
        {/* 左侧 - 图谱可视化 */}
        <div className="lg:col-span-3">
          <Card className="gradient-border h-full flex flex-col">
            <CardHeader className="flex flex-row items-center justify-between shrink-0">
              <div>
                <CardTitle>知识图谱</CardTitle>
                <CardDescription>术语与知识块的关联网络</CardDescription>
              </div>
              <div className="flex items-center gap-2">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <Input placeholder="搜索节点..." className="w-48 pl-9" />
                </div>
                <Button variant="outline" size="icon">
                  <Filter className="h-4 w-4" />
                </Button>
                <div className="flex items-center rounded-lg border border-border">
                  <Button 
                    variant="ghost" 
                    size="icon" 
                    className="h-8 w-8"
                    onClick={() => setZoom(z => Math.max(0.5, z - 0.1))}
                  >
                    <ZoomOut className="h-4 w-4" />
                  </Button>
                  <span className="px-2 text-sm text-muted-foreground">
                    {Math.round(zoom * 100)}%
                  </span>
                  <Button 
                    variant="ghost" 
                    size="icon" 
                    className="h-8 w-8"
                    onClick={() => setZoom(z => Math.min(2, z + 0.1))}
                  >
                    <ZoomIn className="h-4 w-4" />
                  </Button>
                </div>
                <Button variant="outline" size="icon">
                  <Maximize2 className="h-4 w-4" />
                </Button>
                <Button variant="outline" size="icon">
                  <Download className="h-4 w-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="flex-1 min-h-0">
              {loading ? (
                <div className="flex h-full items-center justify-center text-muted-foreground">
                  加载图谱中...
                </div>
              ) : error ? (
                <div className="flex h-full flex-col items-center justify-center gap-2 text-muted-foreground">
                  <p>加载失败: {error}</p>
                  <p className="text-xs">请先运行 python patent_agent_pipeline.py 生成 knowledge_graph.json</p>
                </div>
              ) : (
                <KnowledgeGraphViewer 
                  zoom={zoom} 
                  onSelectNode={setSelectedNode}
                  selectedNode={selectedNode}
                  graphData={graphData}
                />
              )}
            </CardContent>
          </Card>
        </div>

        {/* 右侧 - 详情面板 */}
        <div className="lg:col-span-1">
          <KnowledgeBlockPanel selectedNode={selectedNode} graphData={graphData} />
        </div>
      </div>
    </DashboardLayout>
  )
}
