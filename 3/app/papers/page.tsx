"use client"

import { useState } from "react"
import { DashboardLayout } from "@/components/layout"
import { PaperUpload } from "@/components/papers/paper-upload"
import { PaperList } from "@/components/papers/paper-list"
import { PaperPreview } from "@/components/papers/paper-preview"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function PapersPage() {
  const [selectedPaperId, setSelectedPaperId] = useState<string | null>(null)

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <Tabs defaultValue="upload" className="w-full">
          <TabsList className="grid w-full max-w-md grid-cols-3">
            <TabsTrigger value="upload">上传论文</TabsTrigger>
            <TabsTrigger value="list">论文列表</TabsTrigger>
            <TabsTrigger value="preview">预览解析</TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="mt-6">
            <PaperUpload />
          </TabsContent>

          <TabsContent value="list" className="mt-6">
            <PaperList onSelectPaper={setSelectedPaperId} />
          </TabsContent>

          <TabsContent value="preview" className="mt-6">
            <PaperPreview paperId={selectedPaperId} />
          </TabsContent>
        </Tabs>
      </div>
    </DashboardLayout>
  )
}
