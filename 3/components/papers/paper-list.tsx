"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { 
  Table,
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table"
import { Search, Eye, Trash2, Play, CheckCircle, Clock, AlertCircle } from "lucide-react"
import { cn } from "@/lib/utils"

interface PaperListProps {
  onSelectPaper: (id: string) => void
}

// 模拟论文数据
const mockPapers = [
  {
    id: "1",
    title: "基于深度学习的药物分子设计",
    domain: "chemistry",
    domainLabel: "化学",
    status: "completed",
    wordCount: 5420,
    termsCount: 45,
    createdAt: "2024-01-15",
  },
  {
    id: "2", 
    title: "CRISPR-Cas9 基因编辑技术在癌症治疗中的应用",
    domain: "biology",
    domainLabel: "生物",
    status: "processing",
    wordCount: 7830,
    termsCount: 0,
    createdAt: "2024-01-14",
  },
  {
    id: "3",
    title: "量子计算在密码学中的威胁与机遇",
    domain: "physics",
    domainLabel: "物理",
    status: "completed",
    wordCount: 4200,
    termsCount: 32,
    createdAt: "2024-01-13",
  },
  {
    id: "4",
    title: "新型二维材料的电学性质研究",
    domain: "materials",
    domainLabel: "材料",
    status: "pending",
    wordCount: 6100,
    termsCount: 0,
    createdAt: "2024-01-12",
  },
  {
    id: "5",
    title: "纳米药物载体的靶向递送机制",
    domain: "biology",
    domainLabel: "生物",
    status: "error",
    wordCount: 5800,
    termsCount: 0,
    createdAt: "2024-01-11",
  },
]

const statusConfig = {
  completed: { 
    label: "已完成", 
    icon: CheckCircle, 
    color: "text-success",
    bgColor: "bg-success/10"
  },
  processing: { 
    label: "处理中", 
    icon: Clock, 
    color: "text-warning",
    bgColor: "bg-warning/10"
  },
  pending: { 
    label: "待处理", 
    icon: Clock, 
    color: "text-muted-foreground",
    bgColor: "bg-muted/50"
  },
  error: { 
    label: "错误", 
    icon: AlertCircle, 
    color: "text-destructive",
    bgColor: "bg-destructive/10"
  },
}

const domainColors: Record<string, string> = {
  chemistry: "bg-chart-1/20 text-chart-1",
  biology: "bg-chart-2/20 text-chart-2",
  physics: "bg-chart-3/20 text-chart-3",
  materials: "bg-chart-4/20 text-chart-4",
}

export function PaperList({ onSelectPaper }: PaperListProps) {
  return (
    <Card className="gradient-border">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>论文列表</CardTitle>
        <div className="relative w-64">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input placeholder="搜索论文..." className="pl-9" />
        </div>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>论文标题</TableHead>
              <TableHead>领域</TableHead>
              <TableHead>状态</TableHead>
              <TableHead className="text-right">字数</TableHead>
              <TableHead className="text-right">术语数</TableHead>
              <TableHead>创建时间</TableHead>
              <TableHead className="text-right">操作</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {mockPapers.map((paper) => {
              const status = statusConfig[paper.status as keyof typeof statusConfig]
              const StatusIcon = status.icon

              return (
                <TableRow key={paper.id} className="group">
                  <TableCell className="font-medium max-w-[300px] truncate">
                    {paper.title}
                  </TableCell>
                  <TableCell>
                    <Badge 
                      variant="secondary" 
                      className={cn(domainColors[paper.domain])}
                    >
                      {paper.domainLabel}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className={cn(
                      "flex items-center gap-1.5 rounded-full px-2 py-1 text-xs w-fit",
                      status.bgColor
                    )}>
                      <StatusIcon className={cn("h-3.5 w-3.5", status.color)} />
                      <span className={status.color}>{status.label}</span>
                    </div>
                  </TableCell>
                  <TableCell className="text-right text-muted-foreground">
                    {paper.wordCount.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right">
                    {paper.termsCount > 0 ? (
                      <span className="text-success">{paper.termsCount}</span>
                    ) : (
                      <span className="text-muted-foreground">-</span>
                    )}
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {paper.createdAt}
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={() => onSelectPaper(paper.id)}
                      >
                        <Eye className="h-4 w-4" />
                      </Button>
                      {paper.status === "pending" && (
                        <Button variant="ghost" size="icon" className="h-8 w-8">
                          <Play className="h-4 w-4" />
                        </Button>
                      )}
                      <Button variant="ghost" size="icon" className="h-8 w-8 text-destructive">
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
