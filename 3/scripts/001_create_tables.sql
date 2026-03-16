-- 专利术语提取Agent训练数据生成系统 - 数据库表结构
-- ============================================

-- 1. 论文表 - 存储输入的论文文本
CREATE TABLE IF NOT EXISTS papers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  domain TEXT CHECK (domain IN ('chemistry', 'biology', 'physics', 'materials', 'unknown')),
  status TEXT CHECK (status IN ('pending', 'processing', 'completed', 'error')) DEFAULT 'pending',
  word_count INTEGER DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. 术语表 - 存储提取的专利术语
CREATE TABLE IF NOT EXISTS terms (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  paper_id UUID REFERENCES papers(id) ON DELETE CASCADE,
  term TEXT NOT NULL,
  nested_structure JSONB DEFAULT '{}',
  position_start INTEGER,
  position_end INTEGER,
  confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1) DEFAULT 0.5,
  verified BOOLEAN DEFAULT FALSE,
  verification_note TEXT,
  domain TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. 知识块表 - 存储从外部来源获取的知识
CREATE TABLE IF NOT EXISTS knowledge_blocks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  term_id UUID REFERENCES terms(id) ON DELETE CASCADE,
  source TEXT NOT NULL,
  source_url TEXT,
  content TEXT NOT NULL,
  relevance_score FLOAT CHECK (relevance_score >= 0 AND relevance_score <= 1) DEFAULT 0.5,
  language TEXT DEFAULT 'zh',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. 训练数据表 - 存储生成的训练数据
CREATE TABLE IF NOT EXISTS training_data (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  paper_id UUID REFERENCES papers(id) ON DELETE CASCADE,
  input_text TEXT NOT NULL,
  output_labels JSONB NOT NULL,
  augmented BOOLEAN DEFAULT FALSE,
  augmentation_type TEXT,
  quality_score FLOAT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 5. 训练历史表 - 记录模型训练历史
CREATE TABLE IF NOT EXISTS training_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_name TEXT NOT NULL,
  model_version TEXT,
  domain TEXT,
  accuracy FLOAT,
  precision_score FLOAT,
  recall_score FLOAT,
  f1_score FLOAT,
  loss FLOAT,
  epoch INTEGER,
  benchmark_score FLOAT,
  training_samples INTEGER,
  training_duration INTEGER,
  notes TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 6. 路由配置表 - 存储专项路由配置
CREATE TABLE IF NOT EXISTS routing_configs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  domain TEXT UNIQUE NOT NULL,
  display_name TEXT NOT NULL,
  description TEXT,
  model_endpoint TEXT,
  tokenization_rules JSONB DEFAULT '{}',
  word_formation_rules JSONB DEFAULT '{}',
  example_terms JSONB DEFAULT '[]',
  active BOOLEAN DEFAULT TRUE,
  priority INTEGER DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 7. 工作流运行记录表
CREATE TABLE IF NOT EXISTS pipeline_runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  paper_id UUID REFERENCES papers(id) ON DELETE CASCADE,
  status TEXT CHECK (status IN ('pending', 'running', 'completed', 'failed')) DEFAULT 'pending',
  current_stage TEXT,
  stages_completed JSONB DEFAULT '[]',
  error_message TEXT,
  started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  completed_at TIMESTAMP WITH TIME ZONE
);

-- 创建索引以优化查询性能
CREATE INDEX IF NOT EXISTS idx_papers_domain ON papers(domain);
CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status);
CREATE INDEX IF NOT EXISTS idx_terms_paper_id ON terms(paper_id);
CREATE INDEX IF NOT EXISTS idx_terms_verified ON terms(verified);
CREATE INDEX IF NOT EXISTS idx_knowledge_blocks_term_id ON knowledge_blocks(term_id);
CREATE INDEX IF NOT EXISTS idx_training_data_paper_id ON training_data(paper_id);
CREATE INDEX IF NOT EXISTS idx_training_history_domain ON training_history(domain);
CREATE INDEX IF NOT EXISTS idx_routing_configs_active ON routing_configs(active);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_paper_id ON pipeline_runs(paper_id);
