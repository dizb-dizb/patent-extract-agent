"use client"

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { FileText, Tags, Info } from "lucide-react"
import { cn } from "@/lib/utils"

interface PaperPreviewProps {
  paperId: string | null
}

// 模拟论文详情数据
const mockPaperDetail = {
  id: "1",
  title: "基于深度学习的药物分子设计",
  content: `本文提出了一种基于深度学习的药物分子设计方法。在药物研发过程中，分子设计是关键的一环。传统方法依赖于人工经验和计算化学方法，效率较低。

我们采用了变分自编码器（VAE）和生成对抗网络（GAN）相结合的架构。首先，使用SMILES表示法将分子结构编码为字符串序列。然后，通过卷积神经网络（CNN）提取分子特征。

实验结果表明，该方法在药效预测、ADMET性质预测等方面均优于传统方法。特别是在靶点结合亲和力预测中，准确率达到92.3%。

关键技术包括：
1. 分子指纹编码
2. 图神经网络（GNN）
3. 注意力机制
4. 迁移学习

未来工作将聚焦于多靶点药物设计和药物-药物相互作用预测。`,
  terms: [
    { text: "深度学习", start: 7, end: 11, type: "method", confidence: 0.95 },
    { text: "药物分子设计", start: 13, end: 19, type: "concept", confidence: 0.92 },
    { text: "变分自编码器", start: 89, end: 95, type: "method", confidence: 0.88 },
    { text: "VAE", start: 96, end: 99, type: "abbreviation", confidence: 0.99 },
    { text: "生成对抗网络", start: 101, end: 107, type: "method", confidence: 0.91 },
    { text: "GAN", start: 108, end: 111, type: "abbreviation", confidence: 0.99 },
    { text: "SMILES", start: 127, end: 133, type: "notation", confidence: 0.97 },
    { text: "卷积神经网络", start: 156, end: 162, type: "method", confidence: 0.93 },
    { text: "CNN", start: 163, end: 166, type: "abbreviation", confidence: 0.99 },
    { text: "ADMET", start: 199, end: 204, type: "concept", confidence: 0.89 },
    { text: "靶点结合亲和力", start: 225, end: 232, type: "concept", confidence: 0.86 },
    { text: "分子指纹", start: 259, end: 263, type: "concept", confidence: 0.90 },
    { text: "图神经网络", start: 269, end: 274, type: "method", confidence: 0.94 },
    { text: "GNN", start: 275, end: 278, type: "abbreviation", confidence: 0.99 },
    { text: "注意力机制", start: 283, end: 288, type: "method", confidence: 0.92 },
    { text: "迁移学习", start: 292, end: 296, type: "method", confidence: 0.91 },
  ]
}

const termTypeConfig = {
  method: { label: "方法", color: "bg-chart-1/20 text-chart-1 border-chart-1/30" },
  concept: { label: "概念", color: "bg-chart-2/20 text-chart-2 border-chart-2/30" },
  abbreviation: { label: "缩写", color: "bg-chart-3/20 text-chart-3 border-chart-3/30" },
  notation: { label: "表示法", color: "bg-chart-4/20 text-chart-4 border-chart-4/30" },
}

export function PaperPreview({ paperId }: PaperPreviewProps) {
  if (!paperId) {
    return (
      <Card className="gradient-border">
        <CardContent className="flex min-h-[400px] items-center justify-center">
          <div className="text-center">
            <FileText className="mx-auto h-12 w-12 text-muted-foreground" />
            <p className="mt-4 text-muted-foreground">请从论文列表中选择一篇论文进行预览</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  const paper = mockPaperDetail

  return (
    <TooltipProvider>
      <div className="grid gap-6 lg:grid-cols-3">
        {/* 左侧 - 原文预览 */}
        <Card className="gradient-border lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-primary" />
              {paper.title}
            </CardTitle>
            <CardDescription>
              原文内容（已标注提取的术语）
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[500px] rounded-lg bg-muted/30 p-4">
              <HighlightedText content={paper.content} terms={paper.terms} />
            </ScrollArea>
          </CardContent>
        </Card>

        {/* 右侧 - 术语列表 */}
        <Card className="gradient-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Tags className="h-5 w-5 text-primary" />
              提取的术语
            </CardTitle>
            <CardDescription>
              共 {paper.terms.length} 个术语
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[500px]">
              <div className="space-y-2">
                {paper.terms.map((term, index) => {
                  const typeConfig = termTypeConfig[term.type as keyof typeof termTypeConfig]
                  return (
                    <div
                      key={index}
                      className="flex items-center justify-between rounded-lg bg-muted/30 p-3 hover:bg-muted/50 transition-colors"
                    >
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-foreground">{term.text}</span>
                        <Badge variant="outline" className={cn("text-xs", typeConfig.color)}>
                          {typeConfig.label}
                        </Badge>
                      </div>
                      <Tooltip>
                        <TooltipTrigger>
                          <div className="flex items-center gap-1 text-xs text-muted-foreground">
                            <Info className="h-3 w-3" />
                            {Math.round(term.confidence * 100)}%
                          </div>
                        </TooltipTrigger>
                        <TooltipContent>
                          置信度: {Math.round(term.confidence * 100)}%
                        </TooltipContent>
                      </Tooltip>
                    </div>
                  )
                })}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </TooltipProvider>
  )
}

// 高亮文本组件
function HighlightedText({ 
  content, 
  terms 
}: { 
  content: string
  terms: Array<{ text: string; type: string; confidence: number }>
}) {
  // 简化处理：直接在文本中标注术语
  let highlightedContent = content

  // 按长度降序排列，避免短词覆盖长词
  const sortedTerms = [...terms].sort((a, b) => b.text.length - a.text.length)

  sortedTerms.forEach((term) => {
    const typeConfig = termTypeConfig[term.type as keyof typeof termTypeConfig]
    const regex = new RegExp(`(${term.text})`, "g")
    highlightedContent = highlightedContent.replace(
      regex,
      `<mark class="rounded px-1 py-0.5 ${typeConfig.color} cursor-pointer" data-term="${term.text}">$1</mark>`
    )
  })

  return (
    <div 
      className="prose prose-invert max-w-none text-sm leading-relaxed whitespace-pre-wrap"
      dangerouslySetInnerHTML={{ __html: highlightedContent }}
    />
  )
}
