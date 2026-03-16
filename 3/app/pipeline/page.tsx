"use client"

import { useState } from "react"
import { DashboardLayout } from "@/components/layout"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { 
  Play, 
  Pause, 
  RotateCcw,
  CheckCircle,
  Clock,
  AlertCircle,
  ArrowRight,
  FileText,
  Tags,
  Network,
  Search,
  Database,
  Sparkles,
  ChevronRight,
  Loader2
} from "lucide-react"
import { cn } from "@/lib/utils"

// Pipeline阶段配置
const pipelineStages = [
  {
    id: "input",
    name: "论文输入",
    description: "读取和预处理论文文本",
    icon: FileText,
    estimatedTime: "~30s"
  },
  {
    id: "classification",
    name: "领域分类",
    description: "使用大模型分析论文领域",
    icon: Search,
    estimatedTime: "~1min"
  },
  {
    id: "extraction",
    name: "术语提取",
    description: "提取专利术语和嵌套结构",
    icon: Tags,
    estimatedTime: "~2min"
  },
  {
    id: "knowledge",
    name: "知识关联",
    description: "联网搜索关联知识块",
    icon: Network,
    estimatedTime: "~3min"
  },
  {
    id: "verification",
    name: "验证校正",
    description: "验证术语分割正确性",
    icon: CheckCircle,
    estimatedTime: "~1min"
  },
  {
    id: "augmentation",
    name: "数据增强",
    description: "生成增强训练数据",
    icon: Sparkles,
    estimatedTime: "~2min"
  },
  {
    id: "storage",
    name: "数据存储",
    description: "保存到数据库",
    icon: Database,
    estimatedTime: "~30s"
  }
]

// 模拟运行历史
const mockHistory = [
  {
    id: "run1",
    paperId: "1",
    paperTitle: "基于深度学习的药物分子设计",
    status: "completed",
    startedAt: "2024-01-15 14:30",
    completedAt: "2024-01-15 14:42",
    termsExtracted: 45,
    knowledgeBlocks: 128
  },
  {
    id: "run2",
    paperId: "2",
    paperTitle: "CRISPR-Cas9 基因编辑技术",
    status: "running",
    startedAt: "2024-01-15 15:00",
    currentStage: "knowledge",
    progress: 60
  },
  {
    id: "run3",
    paperId: "3",
    paperTitle: "量子计算在密码学中的应用",
    status: "failed",
    startedAt: "2024-01-15 13:00",
    errorMessage: "知识块搜索超时",
    failedStage: "knowledge"
  }
]

export default function PipelinePage() {
  const [currentRun, setCurrentRun] = useState<typeof mockHistory[1] | null>(mockHistory[1])
  const [isRunning, setIsRunning] = useState(true)

  const getCurrentStageIndex = () => {
    if (!currentRun || currentRun.status !== "running") return -1
    return pipelineStages.findIndex(s => s.id === currentRun.currentStage)
  }

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* 当前运行状态 */}
        <Card className="gradient-border">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>工作流程管线</CardTitle>
              <CardDescription>从论文输入到训练数据生成的完整流程</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              {isRunning ? (
                <Button variant="outline" onClick={() => setIsRunning(false)}>
                  <Pause className="mr-2 h-4 w-4" />
                  暂停
                </Button>
              ) : (
                <Button onClick={() => setIsRunning(true)}>
                  <Play className="mr-2 h-4 w-4" />
                  继续
                </Button>
              )}
              <Button variant="outline">
                <RotateCcw className="mr-2 h-4 w-4" />
                重新开始
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {/* Pipeline 可视化 */}
            <div className="relative">
              {/* 连接线 */}
              <div className="absolute top-8 left-8 right-8 h-0.5 bg-border" />
              <div 
                className="absolute top-8 left-8 h-0.5 bg-primary transition-all duration-500"
                style={{ 
                  width: `${Math.max(0, (getCurrentStageIndex() / (pipelineStages.length - 1)) * 100)}%` 
                }}
              />

              {/* 阶段节点 */}
              <div className="relative flex justify-between">
                {pipelineStages.map((stage, index) => {
                  const Icon = stage.icon
                  const currentIndex = getCurrentStageIndex()
                  const isCompleted = index < currentIndex
                  const isCurrent = index === currentIndex
                  const isPending = index > currentIndex

                  return (
                    <div 
                      key={stage.id}
                      className="flex flex-col items-center"
                      style={{ width: `${100 / pipelineStages.length}%` }}
                    >
                      {/* 节点 */}
                      <div className={cn(
                        "relative z-10 flex h-16 w-16 items-center justify-center rounded-full border-2 transition-all",
                        isCompleted && "border-success bg-success/20",
                        isCurrent && "border-primary bg-primary/20 glow-primary",
                        isPending && "border-muted bg-muted/20"
                      )}>
                        {isCurrent && isRunning ? (
                          <Loader2 className={cn(
                            "h-6 w-6 animate-spin",
                            "text-primary"
                          )} />
                        ) : (
                          <Icon className={cn(
                            "h-6 w-6",
                            isCompleted && "text-success",
                            isCurrent && "text-primary",
                            isPending && "text-muted-foreground"
                          )} />
                        )}
                        
                        {/* 完成标记 */}
                        {isCompleted && (
                          <div className="absolute -right-1 -top-1 flex h-5 w-5 items-center justify-center rounded-full bg-success">
                            <CheckCircle className="h-3 w-3 text-success-foreground" />
                          </div>
                        )}
                      </div>

                      {/* 标签 */}
                      <div className="mt-3 text-center">
                        <p className={cn(
                          "text-sm font-medium",
                          isCompleted && "text-success",
                          isCurrent && "text-primary",
                          isPending && "text-muted-foreground"
                        )}>
                          {stage.name}
                        </p>
                        <p className="text-xs text-muted-foreground mt-0.5">
                          {stage.estimatedTime}
                        </p>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* 当前阶段详情 */}
            {currentRun && currentRun.status === "running" && (
              <div className="mt-8 rounded-lg bg-muted/30 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-foreground">
                      正在处理: {currentRun.paperTitle}
                    </span>
                    <Badge variant="secondary">
                      {pipelineStages[getCurrentStageIndex()]?.name}
                    </Badge>
                  </div>
                  <span className="text-sm text-muted-foreground">
                    {currentRun.progress}%
                  </span>
                </div>
                <Progress value={currentRun.progress} className="h-2" />
                <p className="mt-2 text-xs text-muted-foreground">
                  {pipelineStages[getCurrentStageIndex()]?.description}
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* 运行历史 */}
        <Card className="gradient-border">
          <CardHeader>
            <CardTitle>运行历史</CardTitle>
            <CardDescription>查看所有工作流程运行记录</CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[300px]">
              <div className="space-y-3">
                {mockHistory.map((run) => (
                  <div 
                    key={run.id}
                    className={cn(
                      "flex items-center justify-between rounded-lg bg-muted/30 p-4 hover:bg-muted/50 transition-colors cursor-pointer",
                      run.status === "running" && "border border-primary/50"
                    )}
                    onClick={() => run.status === "running" && setCurrentRun(run)}
                  >
                    <div className="flex items-center gap-4">
                      {/* 状态图标 */}
                      <div className={cn(
                        "flex h-10 w-10 items-center justify-center rounded-full",
                        run.status === "completed" && "bg-success/20",
                        run.status === "running" && "bg-primary/20",
                        run.status === "failed" && "bg-destructive/20"
                      )}>
                        {run.status === "completed" && <CheckCircle className="h-5 w-5 text-success" />}
                        {run.status === "running" && <Loader2 className="h-5 w-5 text-primary animate-spin" />}
                        {run.status === "failed" && <AlertCircle className="h-5 w-5 text-destructive" />}
                      </div>

                      {/* 信息 */}
                      <div>
                        <p className="text-sm font-medium text-foreground">{run.paperTitle}</p>
                        <p className="text-xs text-muted-foreground">
                          开始于 {run.startedAt}
                          {run.completedAt && ` • 完成于 ${run.completedAt}`}
                        </p>
                        {run.status === "failed" && run.errorMessage && (
                          <p className="text-xs text-destructive mt-1">{run.errorMessage}</p>
                        )}
                      </div>
                    </div>

                    {/* 统计/状态 */}
                    <div className="flex items-center gap-4">
                      {run.status === "completed" && (
                        <>
                          <div className="text-right">
                            <p className="text-sm font-medium text-foreground">{run.termsExtracted}</p>
                            <p className="text-xs text-muted-foreground">术语</p>
                          </div>
                          <div className="text-right">
                            <p className="text-sm font-medium text-foreground">{run.knowledgeBlocks}</p>
                            <p className="text-xs text-muted-foreground">知识块</p>
                          </div>
                        </>
                      )}
                      {run.status === "running" && (
                        <div className="flex items-center gap-2">
                          <Progress value={run.progress} className="w-24 h-2" />
                          <span className="text-sm text-muted-foreground">{run.progress}%</span>
                        </div>
                      )}
                      {run.status === "failed" && (
                        <Button variant="outline" size="sm">
                          <RotateCcw className="mr-2 h-3 w-3" />
                          重试
                        </Button>
                      )}
                      <ChevronRight className="h-5 w-5 text-muted-foreground" />
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  )
}
