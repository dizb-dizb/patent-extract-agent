// 专利术语提取Agent训练数据生成系统 - 类型定义
// ============================================

// 领域类型
export type Domain = 'chemistry' | 'biology' | 'physics' | 'materials' | 'unknown'

// 状态类型
export type PaperStatus = 'pending' | 'processing' | 'completed' | 'error'
export type PipelineStatus = 'pending' | 'running' | 'completed' | 'failed'

// 论文
export interface Paper {
  id: string
  title: string
  content: string
  domain: Domain
  status: PaperStatus
  word_count: number
  created_at: string
  updated_at: string
}

// 嵌套结构组件
export interface NestedComponent {
  text: string
  type: string
  meaning?: string
}

// 术语嵌套结构
export interface NestedStructure {
  root: string
  components: NestedComponent[]
  validation?: {
    correct_split: boolean
    incorrect_alternatives?: string[]
  }
}

// 术语
export interface Term {
  id: string
  paper_id: string
  term: string
  nested_structure: NestedStructure
  position_start: number
  position_end: number
  confidence: number
  verified: boolean
  verification_note?: string
  domain?: string
  created_at: string
}

// 知识块
export interface KnowledgeBlock {
  id: string
  term_id: string
  source: string
  source_url?: string
  content: string
  relevance_score: number
  language: string
  created_at: string
}

// 实体标注
export interface EntityLabel {
  text: string
  start: number
  end: number
  label: string
}

// 训练数据输出标签
export interface OutputLabels {
  entities: EntityLabel[]
}

// 训练数据
export interface TrainingData {
  id: string
  paper_id: string
  input_text: string
  output_labels: OutputLabels
  augmented: boolean
  augmentation_type?: string
  quality_score?: number
  created_at: string
}

// 训练历史
export interface TrainingHistory {
  id: string
  model_name: string
  model_version?: string
  domain?: string
  accuracy?: number
  precision_score?: number
  recall_score?: number
  f1_score?: number
  loss?: number
  epoch?: number
  benchmark_score?: number
  training_samples?: number
  training_duration?: number
  notes?: string
  created_at: string
}

// 分词规则
export interface TokenizationRules {
  preserve_elements?: boolean
  handle_numbers?: boolean
  greek_letters?: boolean
  preserve_latin?: boolean
  handle_genes?: boolean
  preserve_units?: boolean
  handle_equations?: boolean
  preserve_compositions?: boolean
}

// 构词规则
export interface WordFormationRules {
  suffixes: string[]
  prefixes: string[]
}

// 路由配置
export interface RoutingConfig {
  id: string
  domain: string
  display_name: string
  description?: string
  model_endpoint?: string
  tokenization_rules: TokenizationRules
  word_formation_rules: WordFormationRules
  example_terms: string[]
  active: boolean
  priority: number
  created_at: string
  updated_at: string
}

// Pipeline阶段
export type PipelineStage = 
  | 'upload'
  | 'domain_classification'
  | 'term_extraction'
  | 'knowledge_retrieval'
  | 'data_generation'
  | 'training'
  | 'validation'

// Pipeline运行记录
export interface PipelineRun {
  id: string
  paper_id: string
  status: PipelineStatus
  current_stage?: PipelineStage
  stages_completed: PipelineStage[]
  error_message?: string
  started_at: string
  completed_at?: string
}

// Dashboard统计数据
export interface DashboardStats {
  totalPapers: number
  totalTerms: number
  verifiedTerms: number
  totalKnowledgeBlocks: number
  totalTrainingData: number
  augmentedData: number
  averageAccuracy: number
  activeRoutes: number
}

// 知识图谱节点
export interface GraphNode {
  id: string
  label: string
  type: 'term' | 'knowledge' | 'paper'
  data?: Term | KnowledgeBlock | Paper
  x?: number
  y?: number
}

// 知识图谱边
export interface GraphEdge {
  source: string
  target: string
  type: 'has_knowledge' | 'from_paper' | 'related_to'
  weight?: number
}

// 知识图谱数据
export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

// 训练指标
export interface TrainingMetrics {
  epoch: number
  accuracy: number
  loss: number
  precision: number
  recall: number
  f1: number
}

// API响应
export interface ApiResponse<T> {
  data?: T
  error?: string
  success: boolean
}

// 分页参数
export interface PaginationParams {
  page: number
  limit: number
}

// 分页响应
export interface PaginatedResponse<T> {
  data: T[]
  total: number
  page: number
  limit: number
  totalPages: number
}
