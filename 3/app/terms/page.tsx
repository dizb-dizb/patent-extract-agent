"use client"

import { useState, useEffect } from "react"
import { DashboardLayout } from "@/components/layout"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { 
  Table,
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table"
import { 
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { 
  Search, 
  CheckCircle, 
  XCircle, 
  Eye, 
  Network,
  Filter,
  Download,
  RefreshCw,
  ChevronRight
} from "lucide-react"
import { cn } from "@/lib/utils"

interface TermItem {
  id: string
  term: string
  nestedStructure?: { root: string; children: string[] }
  domain: string
  domainLabel: string
  paperTitle: string
  confidence: number
  verified: boolean
  knowledgeBlocks: number
}

const domainColors: Record<string, string> = {
  chemistry: "bg-chart-1/20 text-chart-1",
  biology: "bg-chart-2/20 text-chart-2",
  physics: "bg-chart-3/20 text-chart-3",
  materials: "bg-chart-4/20 text-chart-4",
  chem: "bg-chart-1/20 text-chart-1",
  bio: "bg-chart-2/20 text-chart-2",
  phy: "bg-chart-3/20 text-chart-3",
  unknown: "bg-muted/50 text-muted-foreground",
}

export default function TermsPage() {
  const [terms, setTerms] = useState<TermItem[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedTerm, setSelectedTerm] = useState<TermItem | null>(null)
  const [filter, setFilter] = useState<"all" | "verified" | "unverified">("all")

  useEffect(() => {
    fetch("/api/terms")
      .then((r) => r.json())
      .then((data) => setTerms(data.terms ?? []))
      .catch(() => setTerms([]))
      .finally(() => setLoading(false))
  }, [])

  const filteredTerms = terms.filter(term => {
    if (filter === "verified") return term.verified
    if (filter === "unverified") return !term.verified
    if (searchQuery) return term.term.toLowerCase().includes(searchQuery.toLowerCase())
    return true
  })

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* 统计卡片 */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Card className="gradient-border">
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-foreground">{terms.length}</div>
              <p className="text-sm text-muted-foreground">总术语数</p>
            </CardContent>
          </Card>
          <Card className="gradient-border">
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-success">
                {terms.filter(t => t.verified).length}
              </div>
              <p className="text-sm text-muted-foreground">已验证</p>
            </CardContent>
          </Card>
          <Card className="gradient-border">
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-warning">
                {terms.filter(t => !t.verified).length}
              </div>
              <p className="text-sm text-muted-foreground">待验证</p>
            </CardContent>
          </Card>
          <Card className="gradient-border">
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-primary">
                {terms.reduce((acc, t) => acc + t.knowledgeBlocks, 0)}
              </div>
              <p className="text-sm text-muted-foreground">关联知识块</p>
            </CardContent>
          </Card>
        </div>

        {/* 术语列表 */}
        <Card className="gradient-border">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>术语列表</CardTitle>
              <CardDescription>管理和验证提取的专利术语</CardDescription>
            </div>
            <div className="flex items-center gap-3">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input 
                  placeholder="搜索术语..." 
                  className="w-64 pl-9" 
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
              <Tabs value={filter} onValueChange={(v) => setFilter(v as typeof filter)}>
                <TabsList>
                  <TabsTrigger value="all">全部</TabsTrigger>
                  <TabsTrigger value="verified">已验证</TabsTrigger>
                  <TabsTrigger value="unverified">待验证</TabsTrigger>
                </TabsList>
              </Tabs>
              <Button variant="outline" size="icon">
                <Download className="h-4 w-4" />
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="py-12 text-center text-muted-foreground">加载中...</div>
            ) : filteredTerms.length === 0 ? (
              <div className="py-12 text-center text-muted-foreground">
                暂无术语数据。请先运行 python patent_agent_pipeline.py 生成数据。
              </div>
            ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>术语</TableHead>
                  <TableHead>嵌套结构</TableHead>
                  <TableHead>领域</TableHead>
                  <TableHead>来源论文</TableHead>
                  <TableHead className="text-right">置信度</TableHead>
                  <TableHead>验证状态</TableHead>
                  <TableHead className="text-right">知识块</TableHead>
                  <TableHead className="text-right">操作</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredTerms.map((term) => (
                  <TableRow key={term.id} className="group">
                    <TableCell className="font-medium">{term.term}</TableCell>
                    <TableCell>
                      {term.nestedStructure ? (
                        <NestedStructurePreview structure={term.nestedStructure} />
                      ) : (
                        <span className="text-muted-foreground">-</span>
                      )}
                    </TableCell>
                    <TableCell>
                      <Badge 
                        variant="secondary" 
                        className={cn(domainColors[term.domain] || domainColors.unknown)}
                      >
                        {term.domainLabel}
                      </Badge>
                    </TableCell>
                    <TableCell className="max-w-[200px] truncate text-muted-foreground">
                      {term.paperTitle}
                    </TableCell>
                    <TableCell className="text-right">
                      <span className={cn(
                        "font-medium",
                        term.confidence >= 0.9 ? "text-success" : 
                        term.confidence >= 0.8 ? "text-warning" : "text-muted-foreground"
                      )}>
                        {Math.round(term.confidence * 100)}%
                      </span>
                    </TableCell>
                    <TableCell>
                      {term.verified ? (
                        <div className="flex items-center gap-1 text-success">
                          <CheckCircle className="h-4 w-4" />
                          <span className="text-sm">已验证</span>
                        </div>
                      ) : (
                        <div className="flex items-center gap-1 text-muted-foreground">
                          <XCircle className="h-4 w-4" />
                          <span className="text-sm">待验证</span>
                        </div>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {term.knowledgeBlocks > 0 ? (
                        <span className="text-primary">{term.knowledgeBlocks}</span>
                      ) : (
                        <span className="text-muted-foreground">-</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <Dialog>
                          <DialogTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8"
                              onClick={() => setSelectedTerm(term)}
                            >
                              <Eye className="h-4 w-4" />
                            </Button>
                          </DialogTrigger>
                          <DialogContent className="max-w-2xl">
                            <DialogHeader>
                              <DialogTitle>术语详情: {term.term}</DialogTitle>
                              <DialogDescription>
                                查看术语的嵌套结构和关联知识块
                              </DialogDescription>
                            </DialogHeader>
                            <TermDetail term={term} />
                          </DialogContent>
                        </Dialog>
                        <Button variant="ghost" size="icon" className="h-8 w-8">
                          <Network className="h-4 w-4" />
                        </Button>
                        {!term.verified && (
                          <Button variant="ghost" size="icon" className="h-8 w-8 text-success">
                            <CheckCircle className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            )}
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  )
}

// 嵌套结构预览组件
function NestedStructurePreview({ structure }: { structure?: { root: string; children: string[] } }) {
  if (!structure) return <span className="text-muted-foreground">-</span>
  return (
    <div className="flex items-center gap-1 text-sm font-mono">
      <span className="text-primary">{structure.root}</span>
      <ChevronRight className="h-3 w-3 text-muted-foreground" />
      <span className="text-muted-foreground">[{structure.children.join(", ")}]</span>
    </div>
  )
}

// 术语详情组件
function TermDetail({ term }: { term: TermItem }) {
  return (
    <div className="space-y-6 mt-4">
      {/* 嵌套结构可视化 */}
      {term.nestedStructure && (
      <div className="rounded-lg bg-muted/30 p-4">
        <h4 className="text-sm font-medium text-foreground mb-3">嵌套结构</h4>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 rounded-lg bg-primary/20 px-3 py-2 text-primary">
            <span className="font-medium">{term.nestedStructure.root}</span>
          </div>
          <ChevronRight className="h-4 w-4 text-muted-foreground" />
          {term.nestedStructure.children.map((child, idx) => (
            <div key={idx} className="flex items-center gap-1">
              <div className="rounded-lg bg-accent/20 px-3 py-2 text-accent-foreground">
                <span>{child}</span>
              </div>
              {idx < term.nestedStructure!.children.length - 1 && (
                <span className="text-muted-foreground">+</span>
              )}
            </div>
          ))}
        </div>
      </div>
      )}

      {/* 关联知识块 */}
      <div>
        <h4 className="text-sm font-medium text-foreground mb-3">关联知识块</h4>
        {term.knowledgeBlocks > 0 ? (
          <div className="space-y-2">
            <div className="rounded-lg bg-muted/30 p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-foreground">已关联 {term.knowledgeBlocks} 个知识块</span>
              </div>
              <p className="text-sm text-muted-foreground">
                在知识图谱页面可查看该术语的详细证据来源
              </p>
              <Button variant="outline" size="sm" className="mt-2" asChild>
                <a href="/knowledge-graph">查看知识图谱</a>
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center rounded-lg bg-muted/30 p-6">
            <Network className="h-8 w-8 text-muted-foreground" />
            <p className="mt-2 text-sm text-muted-foreground">暂无关联知识块</p>
            <Button variant="outline" size="sm" className="mt-3">
              <RefreshCw className="mr-2 h-4 w-4" />
              联网搜索
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}
