"use client"

import { useState, useCallback } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Upload, FileText, Sparkles, Check, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"

interface UploadState {
  status: "idle" | "uploading" | "processing" | "success" | "error"
  progress: number
  message?: string
}

export function PaperUpload() {
  const [title, setTitle] = useState("")
  const [content, setContent] = useState("")
  const [domain, setDomain] = useState<string>("")
  const [uploadState, setUploadState] = useState<UploadState>({ status: "idle", progress: 0 })
  const [isDragging, setIsDragging] = useState(false)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFileUpload(files[0])
    }
  }, [])

  const handleFileUpload = async (file: File) => {
    if (file.type !== "text/plain" && !file.name.endsWith(".txt")) {
      setUploadState({ status: "error", progress: 0, message: "请上传 .txt 格式的文件" })
      return
    }

    setUploadState({ status: "uploading", progress: 30 })
    
    try {
      const text = await file.text()
      setContent(text)
      setTitle(file.name.replace(".txt", ""))
      setUploadState({ status: "success", progress: 100, message: "文件读取成功" })
    } catch {
      setUploadState({ status: "error", progress: 0, message: "文件读取失败" })
    }
  }

  const handleSubmit = async () => {
    if (!title || !content) return

    setUploadState({ status: "processing", progress: 50, message: "正在分析论文内容..." })

    // 模拟处理过程
    await new Promise(resolve => setTimeout(resolve, 2000))
    setUploadState({ status: "success", progress: 100, message: "论文已提交，正在进行术语提取" })
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* 左侧 - 上传区域 */}
      <Card className="gradient-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5 text-primary" />
            上传论文
          </CardTitle>
          <CardDescription>
            支持 .txt 格式的论文文本文件，或直接粘贴文本内容
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* 拖拽上传区域 */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={cn(
              "relative flex min-h-[200px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed transition-all",
              isDragging ? "border-primary bg-primary/5" : "border-muted hover:border-muted-foreground/50",
              uploadState.status === "success" && "border-success bg-success/5"
            )}
          >
            <input
              type="file"
              accept=".txt"
              className="absolute inset-0 cursor-pointer opacity-0"
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) handleFileUpload(file)
              }}
            />
            
            {uploadState.status === "idle" && (
              <>
                <FileText className="h-12 w-12 text-muted-foreground" />
                <p className="mt-4 text-sm font-medium text-foreground">
                  拖放文件到此处或点击上传
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  支持 .txt 格式
                </p>
              </>
            )}

            {uploadState.status === "uploading" && (
              <>
                <Loader2 className="h-12 w-12 animate-spin text-primary" />
                <p className="mt-4 text-sm font-medium text-foreground">正在读取文件...</p>
              </>
            )}

            {uploadState.status === "success" && (
              <>
                <Check className="h-12 w-12 text-success" />
                <p className="mt-4 text-sm font-medium text-foreground">{uploadState.message}</p>
              </>
            )}

            {uploadState.status === "error" && (
              <>
                <FileText className="h-12 w-12 text-destructive" />
                <p className="mt-4 text-sm font-medium text-destructive">{uploadState.message}</p>
              </>
            )}
          </div>

          {/* 标题输入 */}
          <div className="space-y-2">
            <Label htmlFor="title">论文标题</Label>
            <Input
              id="title"
              placeholder="输入论文标题"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
            />
          </div>

          {/* 领域选择 */}
          <div className="space-y-2">
            <Label htmlFor="domain">所属领域</Label>
            <Select value={domain} onValueChange={setDomain}>
              <SelectTrigger>
                <SelectValue placeholder="选择论文所属领域" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="chemistry">化学</SelectItem>
                <SelectItem value="biology">生物</SelectItem>
                <SelectItem value="physics">物理</SelectItem>
                <SelectItem value="materials">材料科学</SelectItem>
                <SelectItem value="unknown">未知/自动检测</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* 右侧 - 文本输入 */}
      <Card className="gradient-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            论文内容
          </CardTitle>
          <CardDescription>
            直接粘贴论文文本内容，系统将自动进行分析
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder="粘贴论文文本内容..."
            className="min-h-[300px] resize-none font-mono text-sm"
            value={content}
            onChange={(e) => setContent(e.target.value)}
          />
          
          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">
              {content.length > 0 && (
                <span>字符数: {content.length} | 预估词数: {Math.round(content.length / 2)}</span>
              )}
            </div>
            <Button 
              onClick={handleSubmit}
              disabled={!title || !content || uploadState.status === "processing"}
            >
              {uploadState.status === "processing" ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  处理中...
                </>
              ) : (
                <>
                  <Sparkles className="mr-2 h-4 w-4" />
                  开始提取
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
