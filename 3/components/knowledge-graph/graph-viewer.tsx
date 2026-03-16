"use client"

import { useEffect, useRef, useState } from "react"
import { cn } from "@/lib/utils"

interface GraphNode {
  id: string
  label: string
  type: "term" | "knowledge"
  domain?: string
  x: number
  y: number
  vx: number
  vy: number
}

interface GraphLink {
  source: string
  target: string
  strength: number
}

interface KnowledgeGraphViewerProps {
  zoom: number
  onSelectNode: (nodeId: string | null) => void
  selectedNode: string | null
  graphData?: { nodes: Array<{ id: string; label: string; type: string; domain?: string; source?: string; url?: string; title?: string; snippet?: string }>; links: Array<{ source: string; target: string; strength: number }> } | null
}

function graphDataToNodes(
  graphData: KnowledgeGraphViewerProps["graphData"],
  centerX: number,
  centerY: number
): GraphNode[] {
  if (!graphData?.nodes?.length) return []
  const n = graphData.nodes.length
  const r = Math.min(centerX, centerY) * 0.6
  return graphData.nodes.map((node, i) => ({
    ...node,
    x: centerX + r * Math.cos((2 * Math.PI * i) / Math.max(1, n)),
    y: centerY + r * Math.sin((2 * Math.PI * i) / Math.max(1, n)),
    vx: 0,
    vy: 0,
  }))
}

const domainColors: Record<string, string> = {
  chemistry: "#00d4ff",
  biology: "#7c3aed", 
  physics: "#22c55e",
  materials: "#f59e0b",
  chem: "#00d4ff",
  bio: "#7c3aed",
  phy: "#22c55e",
  unknown: "#9ca3af",
}

export function KnowledgeGraphViewer({ zoom, onSelectNode, selectedNode, graphData }: KnowledgeGraphViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const links = graphData?.links ?? []
  const [nodes, setNodes] = useState<GraphNode[]>(() =>
    graphDataToNodes(graphData, 400, 300)
  )
  const animationRef = useRef<number>()
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)

  useEffect(() => {
    if (graphData?.nodes?.length) {
      setNodes(graphDataToNodes(graphData, 400, 300))
    }
  }, [graphData])

  // 简单的力导向布局模拟
  useEffect(() => {
    if (nodes.length === 0) return
    const simulate = () => {
      setNodes(prevNodes => {
        const newNodes = prevNodes.map(node => ({ ...node }))
        
        // 斥力
        for (let i = 0; i < newNodes.length; i++) {
          for (let j = i + 1; j < newNodes.length; j++) {
            const dx = newNodes[j].x - newNodes[i].x
            const dy = newNodes[j].y - newNodes[i].y
            const dist = Math.sqrt(dx * dx + dy * dy) || 1
            const force = 2000 / (dist * dist)
            
            newNodes[i].vx -= (dx / dist) * force * 0.01
            newNodes[i].vy -= (dy / dist) * force * 0.01
            newNodes[j].vx += (dx / dist) * force * 0.01
            newNodes[j].vy += (dy / dist) * force * 0.01
          }
        }
        
        // 引力（连接的节点）
        links.forEach(link => {
          const source = newNodes.find(n => n.id === link.source)
          const target = newNodes.find(n => n.id === link.target)
          if (source && target) {
            const dx = target.x - source.x
            const dy = target.y - source.y
            const dist = Math.sqrt(dx * dx + dy * dy) || 1
            const force = (dist - 100) * 0.005 * link.strength
            
            source.vx += (dx / dist) * force
            source.vy += (dy / dist) * force
            target.vx -= (dx / dist) * force
            target.vy -= (dy / dist) * force
          }
        })
        
        // 中心引力
        newNodes.forEach(node => {
          node.vx += (400 - node.x) * 0.001
          node.vy += (300 - node.y) * 0.001
        })
        
        // 更新位置
        newNodes.forEach(node => {
          node.vx *= 0.9
          node.vy *= 0.9
          node.x += node.vx
          node.y += node.vy
          // 边界限制
          node.x = Math.max(50, Math.min(750, node.x))
          node.y = Math.max(50, Math.min(550, node.y))
        })
        
        return newNodes
      })
      
      animationRef.current = requestAnimationFrame(simulate)
    }
    
    simulate()
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [links])

  // 绘制图谱
  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext("2d")
    if (!canvas || !ctx) return

    const container = containerRef.current
    if (container) {
      canvas.width = container.clientWidth
      canvas.height = container.clientHeight
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.save()
    
    // 应用缩放
    const centerX = canvas.width / 2
    const centerY = canvas.height / 2
    ctx.translate(centerX, centerY)
    ctx.scale(zoom, zoom)
    ctx.translate(-centerX, -centerY)

    // 绘制连接线
    links.forEach(link => {
      const source = nodes.find(n => n.id === link.source)
      const target = nodes.find(n => n.id === link.target)
      if (source && target) {
        ctx.beginPath()
        ctx.moveTo(source.x, source.y)
        ctx.lineTo(target.x, target.y)
        ctx.strokeStyle = selectedNode === source.id || selectedNode === target.id 
          ? "rgba(0, 212, 255, 0.6)" 
          : "rgba(100, 100, 120, 0.3)"
        ctx.lineWidth = link.strength * 2
        ctx.stroke()
      }
    })

    // 绘制节点
    nodes.forEach(node => {
      const isSelected = selectedNode === node.id
      const isHovered = hoveredNode === node.id
      const radius = node.type === "term" ? 20 : 15
      
      // 节点光晕
      if (isSelected || isHovered) {
        ctx.beginPath()
        ctx.arc(node.x, node.y, radius + 8, 0, Math.PI * 2)
        ctx.fillStyle = node.type === "term" 
          ? `${domainColors[node.domain || "chemistry"]}33`
          : "rgba(124, 58, 237, 0.2)"
        ctx.fill()
      }

      // 节点
      ctx.beginPath()
      ctx.arc(node.x, node.y, radius, 0, Math.PI * 2)
      
      if (node.type === "term") {
        ctx.fillStyle = domainColors[node.domain || "chemistry"]
      } else {
        ctx.fillStyle = "#7c3aed"
      }
      ctx.fill()
      
      // 边框
      ctx.strokeStyle = isSelected ? "#fff" : "rgba(255,255,255,0.3)"
      ctx.lineWidth = isSelected ? 2 : 1
      ctx.stroke()

      // 标签
      ctx.font = "12px Geist, sans-serif"
      ctx.fillStyle = "#e5e5e5"
      ctx.textAlign = "center"
      ctx.fillText(node.label, node.x, node.y + radius + 15)
    })

    ctx.restore()
  }, [nodes, zoom, selectedNode, hoveredNode])

  // 处理鼠标事件
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left) / zoom
    const y = (e.clientY - rect.top) / zoom

    // 检测点击的节点
    const clickedNode = nodes.find(node => {
      const radius = node.type === "term" ? 20 : 15
      const dx = node.x - x
      const dy = node.y - y
      return Math.sqrt(dx * dx + dy * dy) < radius
    })

    onSelectNode(clickedNode?.id || null)
  }

  const handleCanvasMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left) / zoom
    const y = (e.clientY - rect.top) / zoom

    const hovered = nodes.find(node => {
      const radius = node.type === "term" ? 20 : 15
      const dx = node.x - x
      const dy = node.y - y
      return Math.sqrt(dx * dx + dy * dy) < radius
    })

    setHoveredNode(hovered?.id || null)
  }

  return (
    <div ref={containerRef} className="relative h-full w-full rounded-lg bg-muted/20 overflow-hidden">
      <canvas
        ref={canvasRef}
        className="absolute inset-0 cursor-crosshair"
        onClick={handleCanvasClick}
        onMouseMove={handleCanvasMove}
      />
      
      {/* 图例 */}
      <div className="absolute bottom-4 left-4 flex items-center gap-4 rounded-lg bg-card/80 backdrop-blur-sm px-4 py-2">
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full" style={{ background: domainColors.chemistry }} />
          <span className="text-xs text-muted-foreground">化学</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full" style={{ background: domainColors.biology }} />
          <span className="text-xs text-muted-foreground">生物</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full" style={{ background: domainColors.physics }} />
          <span className="text-xs text-muted-foreground">物理</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full" style={{ background: domainColors.materials }} />
          <span className="text-xs text-muted-foreground">材料</span>
        </div>
        <div className="h-4 w-px bg-border" />
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full bg-accent" />
          <span className="text-xs text-muted-foreground">知识块</span>
        </div>
      </div>

      {/* 统计信息 */}
      <div className="absolute top-4 right-4 rounded-lg bg-card/80 backdrop-blur-sm px-4 py-2">
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <span>节点: {nodes.length}</span>
          <span>连接: {links.length}</span>
        </div>
      </div>
    </div>
  )
}
