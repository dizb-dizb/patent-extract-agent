"use client"

import { usePathname } from "next/navigation"
import { Bell, Search, Activity, Zap, HardDrive } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"

const pageTitles: Record<string, { title: string; description: string }> = {
  "/": { title: "仪表盘", description: "系统总览与快速操作" },
  "/papers": { title: "论文输入", description: "上传和管理论文文本数据" },
  "/terms": { title: "术语提取", description: "查看和验证提取的专利术语" },
  "/knowledge-graph": { title: "知识图谱", description: "术语关联知识块网络可视化" },
  "/pipeline": { title: "工作流程", description: "端到端数据处理管线" },
  "/training": { title: "训练监控", description: "模型训练指标与性能评估" },
  "/routing": { title: "路由管理", description: "专项领域分词路由配置" },
}

export function TopBar() {
  const pathname = usePathname()
  const pageInfo = pageTitles[pathname] || { title: "PatentTerm Agent", description: "专利术语提取训练系统" }

  return (
    <header className="flex h-16 items-center justify-between border-b border-border bg-card/50 px-6 backdrop-blur-sm">
      {/* 页面标题 */}
      <div className="flex flex-col">
        <h1 className="text-lg font-semibold text-foreground">{pageInfo.title}</h1>
        <p className="text-sm text-muted-foreground">{pageInfo.description}</p>
      </div>

      {/* 右侧操作区 */}
      <div className="flex items-center gap-4">
        {/* 搜索框 */}
        <div className="relative hidden md:block">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="搜索术语、论文..."
            className="w-64 bg-muted/50 pl-9 focus:bg-muted"
          />
        </div>

        {/* 系统状态指示器 */}
        <div className="flex items-center gap-2 rounded-lg bg-muted/50 px-3 py-1.5">
          <div className="flex items-center gap-1.5">
            <Activity className="h-4 w-4 text-success" />
            <span className="text-xs text-muted-foreground">Agent</span>
          </div>
          <div className="h-4 w-px bg-border" />
          <div className="flex items-center gap-1.5">
            <Zap className="h-4 w-4 text-warning" />
            <span className="text-xs text-muted-foreground">GPU</span>
          </div>
          <div className="h-4 w-px bg-border" />
          <div className="flex items-center gap-1.5">
            <HardDrive className="h-4 w-4 text-info" />
            <span className="text-xs text-muted-foreground">存储</span>
          </div>
        </div>

        {/* 通知按钮 */}
        <Button variant="ghost" size="icon" className="relative">
          <Bell className="h-5 w-5" />
          <Badge 
            variant="destructive" 
            className="absolute -right-1 -top-1 h-5 w-5 rounded-full p-0 text-xs flex items-center justify-center"
          >
            3
          </Badge>
        </Button>
      </div>
    </header>
  )
}
