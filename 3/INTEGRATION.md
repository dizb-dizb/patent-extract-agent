# 前端与 Python Pipeline 集成说明

## 数据流

```
patent_agent_pipeline.py
    │
    ├── knowledge.db (SQLite)
    ├── knowledge_graph.json
    └── 3/public/data/knowledge_graph.json (自动复制)
    │
    ▼
Next.js API Routes (3/app/api/)
    ├── /api/knowledge-graph  → 读取 knowledge_graph.json
    ├── /api/terms            → 从图谱派生术语列表
    └── /api/stats            → 聚合仪表盘统计
    │
    ▼
前端页面
    ├── 仪表盘 (/)           → 论文数、术语数、知识块、验证率
    ├── 术语提取 (/terms)    → 术语列表、搜索、筛选、详情
    └── 知识图谱 (/knowledge-graph) → 术语-证据网络可视化
```

## 使用步骤

1. **生成数据**（在项目根目录）：
   ```bash
   python patent_agent_pipeline.py
   ```
   或单独导出图谱：
   ```bash
   python export_graph.py
   ```

2. **启动前端**（在 3/ 目录）：
   ```bash
   cd 3
   pnpm dev
   ```

3. **访问**：http://localhost:3000

## 数据读取优先级

API 按以下顺序查找 `knowledge_graph.json`：
1. `../knowledge_graph.json`（项目根目录，相对 3/）
2. `knowledge_graph.json`（3/ 目录）
3. `public/data/knowledge_graph.json`（静态资源）

运行 `patent_agent_pipeline` 或 `export_graph.py` 后会自动复制到 `3/public/data/`。

## 已集成页面

| 页面 | 数据来源 | 说明 |
|------|----------|------|
| 仪表盘 | /api/stats | 术语数、验证数、知识块、领域分布 |
| 术语提取 | /api/terms | 术语列表、领域、验证状态、知识块数 |
| 知识图谱 | /api/knowledge-graph | 术语与证据节点、边、详情面板 |

## 待完善

- **论文输入** (`/papers`)：当前为 UI 占位，需对接后端或本地 input/ 目录
- **工作流程** (`/pipeline`)：当前为模拟状态，可对接 cloud_ssh_runner 等
- **训练监控** (`/training`)、**路由管理** (`/routing`)：侧边栏有入口，页面待实现
