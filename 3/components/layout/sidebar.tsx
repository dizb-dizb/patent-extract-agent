"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import {
  FileText,
  Tags,
  Network,
  GitBranch,
  BarChart3,
  Settings,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Database
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

interface SidebarProps {
  collapsed: boolean
  onToggle: () => void
}

const navItems = [
  {
    title: "论文输入",
    href: "/papers",
    icon: FileText,
    description: "上传和管理论文文本"
  },
  {
    title: "术语提取",
    href: "/terms",
    icon: Tags,
    description: "查看提取的专利术语"
  },
  {
    title: "知识图谱",
    href: "/knowledge-graph",
    icon: Network,
    description: "术语关联知识网络"
  },
  {
    title: "工作流程",
    href: "/pipeline",
    icon: GitBranch,
    description: "端到端处理管线"
  },
  {
    title: "训练监控",
    href: "/training",
    icon: BarChart3,
    description: "模型训练指标"
  },
  {
    title: "路由管理",
    href: "/routing",
    icon: Settings,
    description: "专项分词路由配置"
  },
]

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const pathname = usePathname()

  return (
    <TooltipProvider delayDuration={0}>
      <aside
        className={cn(
          "flex flex-col border-r border-sidebar-border bg-sidebar transition-all duration-300",
          collapsed ? "w-16" : "w-64"
        )}
      >
        {/* Logo区域 */}
        <div className="flex h-16 items-center justify-between border-b border-sidebar-border px-4">
          {!collapsed && (
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 glow-primary">
                <Cpu className="h-5 w-5 text-primary" />
              </div>
              <div className="flex flex-col">
                <span className="text-sm font-semibold text-sidebar-foreground">PatentTerm</span>
                <span className="text-xs text-muted-foreground">Agent System</span>
              </div>
            </div>
          )}
          {collapsed && (
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 glow-primary mx-auto">
              <Cpu className="h-5 w-5 text-primary" />
            </div>
          )}
        </div>

        {/* 导航菜单 */}
        <nav className="flex-1 space-y-1 p-2">
          {navItems.map((item) => {
            const isActive = pathname === item.href
            const Icon = item.icon

            const linkContent = (
              <Link
                href={item.href}
                className={cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all",
                  "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                  isActive && "bg-sidebar-primary/10 text-sidebar-primary glow-primary",
                  !isActive && "text-sidebar-foreground",
                  collapsed && "justify-center px-2"
                )}
              >
                <Icon className={cn("h-5 w-5 shrink-0", isActive && "text-primary")} />
                {!collapsed && <span>{item.title}</span>}
              </Link>
            )

            if (collapsed) {
              return (
                <Tooltip key={item.href}>
                  <TooltipTrigger asChild>
                    {linkContent}
                  </TooltipTrigger>
                  <TooltipContent side="right" className="flex flex-col">
                    <span className="font-medium">{item.title}</span>
                    <span className="text-xs text-muted-foreground">{item.description}</span>
                  </TooltipContent>
                </Tooltip>
              )
            }

            return <div key={item.href}>{linkContent}</div>
          })}
        </nav>

        {/* 数据库状态 */}
        <div className="border-t border-sidebar-border p-3">
          <div className={cn(
            "flex items-center gap-2 rounded-lg bg-muted/50 px-3 py-2",
            collapsed && "justify-center px-2"
          )}>
            <Database className="h-4 w-4 text-success" />
            {!collapsed && (
              <div className="flex flex-col">
                <span className="text-xs font-medium text-sidebar-foreground">Supabase</span>
                <span className="text-xs text-muted-foreground">已连接</span>
              </div>
            )}
            <div className="status-dot success ml-auto" />
          </div>
        </div>

        {/* 折叠按钮 */}
        <div className="border-t border-sidebar-border p-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggle}
            className="w-full justify-center"
          >
            {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </Button>
        </div>
      </aside>
    </TooltipProvider>
  )
}
