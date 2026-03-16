"use client"

import Link from "next/link"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { FileUp, Play, Search, Settings } from "lucide-react"

const actions = [
  {
    title: "上传论文",
    description: "添加新的论文文本",
    icon: FileUp,
    href: "/papers",
    variant: "default" as const,
  },
  {
    title: "启动流程",
    description: "运行完整处理管线",
    icon: Play,
    href: "/pipeline",
    variant: "secondary" as const,
  },
  {
    title: "搜索术语",
    description: "查找已提取的术语",
    icon: Search,
    href: "/terms",
    variant: "secondary" as const,
  },
  {
    title: "配置路由",
    description: "管理专项分词规则",
    icon: Settings,
    href: "/routing",
    variant: "secondary" as const,
  },
]

export function QuickActions() {
  return (
    <Card className="gradient-border">
      <CardHeader>
        <CardTitle className="text-base font-semibold">快速操作</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-2 gap-3">
        {actions.map((action) => {
          const Icon = action.icon
          return (
            <Link key={action.title} href={action.href}>
              <Button
                variant={action.variant}
                className="h-auto w-full flex-col gap-2 py-4"
              >
                <Icon className="h-5 w-5" />
                <div className="flex flex-col items-center">
                  <span className="text-sm font-medium">{action.title}</span>
                  <span className="text-xs text-muted-foreground">{action.description}</span>
                </div>
              </Button>
            </Link>
          )
        })}
      </CardContent>
    </Card>
  )
}
