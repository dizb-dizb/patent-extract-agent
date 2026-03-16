"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import { FileText, Tags, Network, CheckCircle, Clock, AlertCircle } from "lucide-react"

interface Activity {
  id: string
  type: "paper" | "term" | "knowledge" | "training"
  title: string
  description: string
  status: "success" | "pending" | "error"
  timestamp: string
}

const mockActivities: Activity[] = [
  {
    id: "1",
    type: "paper",
    title: "论文上传完成",
    description: "《基于深度学习的药物分子设计》已完成解析",
    status: "success",
    timestamp: "2分钟前"
  },
  {
    id: "2",
    type: "term",
    title: "术语提取进行中",
    description: "正在从论文中提取专利术语...",
    status: "pending",
    timestamp: "5分钟前"
  },
  {
    id: "3",
    type: "knowledge",
    title: "知识块关联完成",
    description: "已关联 12 个术语到 Wiki 知识块",
    status: "success",
    timestamp: "10分钟前"
  },
  {
    id: "4",
    type: "training",
    title: "训练任务失败",
    description: "化学领域模型训练出现异常",
    status: "error",
    timestamp: "15分钟前"
  },
]

const typeIcons = {
  paper: FileText,
  term: Tags,
  knowledge: Network,
  training: CheckCircle,
}

const statusConfig = {
  success: { icon: CheckCircle, color: "text-success", bg: "bg-success/10" },
  pending: { icon: Clock, color: "text-warning", bg: "bg-warning/10" },
  error: { icon: AlertCircle, color: "text-destructive", bg: "bg-destructive/10" },
}

export function RecentActivity() {
  return (
    <Card className="gradient-border">
      <CardHeader>
        <CardTitle className="text-base font-semibold">最近活动</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {mockActivities.map((activity) => {
          const TypeIcon = typeIcons[activity.type]
          const status = statusConfig[activity.status]
          const StatusIcon = status.icon

          return (
            <div
              key={activity.id}
              className="flex items-start gap-3 rounded-lg bg-muted/30 p-3 transition-colors hover:bg-muted/50"
            >
              <div className={cn("flex h-8 w-8 shrink-0 items-center justify-center rounded-lg", status.bg)}>
                <TypeIcon className={cn("h-4 w-4", status.color)} />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-foreground truncate">
                    {activity.title}
                  </span>
                  <StatusIcon className={cn("h-3.5 w-3.5 shrink-0", status.color)} />
                </div>
                <p className="text-xs text-muted-foreground mt-0.5 truncate">
                  {activity.description}
                </p>
              </div>
              <span className="text-xs text-muted-foreground shrink-0">
                {activity.timestamp}
              </span>
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}
