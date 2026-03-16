"use client"

import { useState, useEffect } from "react"
import { DashboardLayout } from "@/components/layout"
import { StatsCard, RecentActivity, QuickActions } from "@/components/dashboard"
import { FileText, Tags, Network, BarChart3, TrendingUp, Clock } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"

const domainLabels: Record<string, string> = {
  chem: "化学",
  bio: "生物",
  phy: "物理",
  materials: "材料",
  unknown: "未知",
}

const domainColors: Record<string, string> = {
  chem: "#00d4ff",
  bio: "#7c3aed",
  phy: "#22c55e",
  materials: "#f59e0b",
  unknown: "#9ca3af",
}

export default function HomePage() {
  const [stats, setStats] = useState({
    papers: 0,
    terms: 0,
    verified: 0,
    knowledgeBlocks: 0,
    domainCounts: {} as Record<string, number>,
  })

  useEffect(() => {
    fetch("/api/stats")
      .then((r) => r.json())
      .then(setStats)
      .catch(() => {})
  }, [])

  const domainData = Object.entries(stats.domainCounts).map(([key, count]) => ({
    name: domainLabels[key] || key,
    count,
    color: domainColors[key] || domainColors.unknown,
  }))
  const totalDomainCount = domainData.reduce((acc, d) => acc + d.count, 0) || 1

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* 欢迎区域 */}
        <div className="gradient-border overflow-hidden rounded-xl p-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-foreground">
                欢迎使用 PatentTerm Agent
              </h2>
              <p className="mt-1 text-muted-foreground">
                专利术语提取与模型训练数据生成系统
              </p>
            </div>
            <div className="hidden md:flex items-center gap-4">
              <div className="flex items-center gap-2 rounded-lg bg-primary/10 px-4 py-2">
                <TrendingUp className="h-5 w-5 text-primary" />
                <div>
                  <p className="text-sm font-medium text-foreground">系统运行正常</p>
                  <p className="text-xs text-muted-foreground">已运行 72 小时</p>
                </div>
              </div>
              <div className="flex items-center gap-2 rounded-lg bg-success/10 px-4 py-2">
                <Clock className="h-5 w-5 text-success" />
                <div>
                  <p className="text-sm font-medium text-foreground">3 个任务进行中</p>
                  <p className="text-xs text-muted-foreground">预计 15 分钟完成</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 统计卡片 */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatsCard
            title="论文总数"
            value={String(stats.papers || 0)}
            description="输入论文数量"
            icon={FileText}
            trend={{ value: 0, isPositive: true }}
          />
          <StatsCard
            title="提取术语"
            value={String(stats.terms)}
            description={`已验证 ${stats.verified} 个`}
            icon={Tags}
            trend={{ value: 0, isPositive: true }}
          />
          <StatsCard
            title="知识块"
            value={String(stats.knowledgeBlocks)}
            description="关联 Wiki/DuckDuckGo 来源"
            icon={Network}
            trend={{ value: 0, isPositive: true }}
          />
          <StatsCard
            title="验证率"
            value={stats.terms ? `${Math.round((stats.verified / stats.terms) * 100)}%` : "-"}
            description="有证据的术语占比"
            icon={BarChart3}
            trend={{ value: 0, isPositive: true }}
          />
        </div>

        {/* 主内容区 */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* 左侧 - 最近活动 */}
          <div className="lg:col-span-2">
            <RecentActivity />
          </div>

          {/* 右侧 */}
          <div className="space-y-6">
            {/* 快速操作 */}
            <QuickActions />

            {/* 领域分布 */}
            <Card className="gradient-border">
              <CardHeader>
                <CardTitle className="text-base font-semibold">领域分布</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {domainData.map((domain) => (
                  <div key={domain.name} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-foreground">{domain.name}</span>
                      <span className="text-muted-foreground">
                        {domain.count} 篇 ({Math.round(domain.count / totalDomainCount * 100)}%)
                      </span>
                    </div>
                    <Progress 
                      value={(domain.count / totalDomainCount) * 100} 
                      className="h-2"
                      style={{ 
                        // @ts-expect-error CSS custom property
                        "--progress-background": domain.color 
                      }}
                    />
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </DashboardLayout>
  )
}
