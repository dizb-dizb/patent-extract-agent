"use client"

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { 
  Network, 
  ExternalLink, 
  RefreshCw, 
  CheckCircle,
  Globe,
  BookOpen,
  FileText
} from "lucide-react"
import { cn } from "@/lib/utils"

interface GraphData {
  nodes: Array<{ id: string; label: string; type: string; domain?: string; source?: string; url?: string; title?: string; snippet?: string }>
  links: Array<{ source: string; target: string; strength: number }>
}

interface KnowledgeBlockPanelProps {
  selectedNode: string | null
  graphData?: GraphData | null
}

const sourceIcons: Record<string, typeof Globe> = {
  Wikipedia: Globe,
  wikipedia_zh: Globe,
  duckduckgo: Globe,
  "专利文献": FileText,
  "学术论文": BookOpen,
}

export function KnowledgeBlockPanel({ selectedNode, graphData }: KnowledgeBlockPanelProps) {
  if (!selectedNode) {
    return (
      <Card className="gradient-border h-full">
        <CardContent className="flex h-full items-center justify-center">
          <div className="text-center">
            <Network className="mx-auto h-12 w-12 text-muted-foreground" />
            <p className="mt-4 text-sm text-muted-foreground">
              点击图谱中的节点查看详情
            </p>
          </div>
        </CardContent>
      </Card>
    )
  }

  const nodeDetail = graphData?.nodes.find((n) => n.id === selectedNode)
  const linkedChunks =
    nodeDetail?.type === "term"
      ? (graphData?.links
          ?.filter((l) => l.source === selectedNode)
          .map((l) => graphData.nodes.find((n) => n.id === l.target))
          .filter(Boolean) ?? [])
      : []

  if (!nodeDetail) {
    return (
      <Card className="gradient-border h-full">
        <CardContent className="flex h-full items-center justify-center">
          <div className="text-center">
            <Network className="mx-auto h-12 w-12 text-muted-foreground" />
            <p className="mt-4 text-sm text-muted-foreground">
              未找到节点详情
            </p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="gradient-border h-full flex flex-col">
      <CardHeader className="shrink-0">
        <div className="flex items-center justify-between">
          <Badge 
            variant="outline" 
            className={cn(
              nodeDetail.type === "term" ? "bg-primary/20 text-primary" : "bg-accent/20 text-accent-foreground"
            )}
          >
            {nodeDetail.type === "term" ? "术语" : "知识块"}
          </Badge>
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
        <CardTitle className="text-lg">{nodeDetail.label}</CardTitle>
        {nodeDetail.domain && (
          <CardDescription className="text-sm">
            领域: {nodeDetail.domain}
          </CardDescription>
        )}
      </CardHeader>

      <CardContent className="flex-1 min-h-0">
        <ScrollArea className="h-full pr-4">
          {nodeDetail.type === "term" && linkedChunks.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium text-foreground">关联知识块</h4>
                <Badge variant="secondary">{linkedChunks.length} 个</Badge>
              </div>

              {linkedChunks.map((block) => {
                if (!block || block.type !== "knowledge") return null
                const SourceIcon = sourceIcons[block.source || ""] || Globe
                return (
                  <div 
                    key={block.id}
                    className="rounded-lg bg-muted/30 p-3 space-y-2 hover:bg-muted/50 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <SourceIcon className="h-4 w-4 text-muted-foreground" />
                        <span className="text-xs text-muted-foreground">{block.source || "未知"}</span>
                      </div>
                    </div>
                    <h5 className="text-sm font-medium text-foreground">{block.title || block.label}</h5>
                    <p className="text-xs text-muted-foreground line-clamp-3">
                      {block.snippet || ""}
                    </p>
                    {block.url && (
                      <Button variant="link" size="sm" className="h-auto p-0 text-xs" asChild>
                        <a href={block.url} target="_blank" rel="noopener noreferrer">
                          查看原文 <ExternalLink className="ml-1 h-3 w-3" />
                        </a>
                      </Button>
                    )}
                  </div>
                )
              })}
            </div>
          )}

          {nodeDetail.type === "term" && linkedChunks.length === 0 && (
            <div className="flex flex-col items-center justify-center rounded-lg bg-muted/30 p-6">
              <Network className="h-8 w-8 text-muted-foreground" />
              <p className="mt-2 text-sm text-muted-foreground">暂无关联知识块</p>
              <p className="text-xs text-muted-foreground">运行 patent_agent_pipeline 进行联网检索</p>
            </div>
          )}

          {nodeDetail.type === "knowledge" && (
            <div className="space-y-4">
              <div className="rounded-lg bg-muted/30 p-3">
                <div className="flex items-center gap-2 mb-2">
                  <Globe className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">{nodeDetail.source || "未知"}</span>
                </div>
                {nodeDetail.title && (
                  <p className="text-sm font-medium text-foreground mb-2">{nodeDetail.title}</p>
                )}
                {nodeDetail.snippet && (
                  <p className="text-xs text-muted-foreground line-clamp-4 mb-2">{nodeDetail.snippet}</p>
                )}
                {nodeDetail.url && (
                  <Button variant="link" size="sm" className="h-auto p-0 text-xs" asChild>
                    <a href={nodeDetail.url} target="_blank" rel="noopener noreferrer">
                      访问原始来源 <ExternalLink className="ml-1 h-3 w-3" />
                    </a>
                  </Button>
                )}
              </div>

              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-success" />
                <span className="text-sm text-muted-foreground">已验证来源</span>
              </div>
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
