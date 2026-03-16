-- 种子数据 - 用于演示和测试
-- ============================================

-- 插入路由配置
INSERT INTO routing_configs (domain, display_name, description, tokenization_rules, word_formation_rules, example_terms, active, priority)
VALUES 
  ('chemistry', '化学领域', '化学相关专利术语的分词规则，包含化合物命名、反应类型等', 
   '{"preserve_elements": true, "handle_numbers": true, "greek_letters": true}',
   '{"suffixes": ["-ase", "-ol", "-ide", "-ate", "-ine"], "prefixes": ["poly-", "mono-", "di-", "tri-"]}',
   '["聚乙烯", "氧化还原反应", "催化剂", "纳米颗粒", "电解质"]',
   true, 1),
  ('biology', '生物领域', '生物学相关专利术语的分词规则，包含基因、蛋白质命名等',
   '{"preserve_latin": true, "handle_genes": true}',
   '{"suffixes": ["-ase", "-in", "-gen", "-cyte"], "prefixes": ["anti-", "neo-", "proto-"]}',
   '["细胞膜", "蛋白质折叠", "基因表达", "抗体", "核酸序列"]',
   true, 2),
  ('physics', '物理领域', '物理学相关专利术语的分词规则，包含量子、光学等',
   '{"preserve_units": true, "handle_equations": true}',
   '{"suffixes": ["-tion", "-ity"], "prefixes": ["super-", "semi-", "quasi-"]}',
   '["量子纠缠", "超导体", "半导体", "光子晶体", "电磁波"]',
   true, 3),
  ('materials', '材料科学', '材料科学相关专利术语的分词规则',
   '{"preserve_compositions": true}',
   '{"suffixes": ["-ite", "-ene"], "prefixes": ["nano-", "micro-"]}',
   '["石墨烯", "碳纳米管", "复合材料", "合金", "陶瓷"]',
   true, 4)
ON CONFLICT (domain) DO NOTHING;

-- 插入示例论文
INSERT INTO papers (id, title, content, domain, status, word_count)
VALUES 
  ('a1b2c3d4-e5f6-7890-abcd-ef1234567890', 
   '基于纳米材料的高效催化剂研究',
   '本发明涉及一种基于纳米材料的高效催化剂及其制备方法。该催化剂由纳米颗粒组成，具有较高的比表面积和优异的催化活性。纳米颗粒的粒径范围为1-100纳米，优选为5-50纳米。催化剂载体可选用氧化铝、二氧化硅或活性炭。本发明的催化剂可广泛应用于石油化工、精细化工和环保领域的催化反应中，特别适用于氧化还原反应和加氢反应。实验结果表明，该催化剂的转化率可达95%以上，选择性可达90%以上。',
   'chemistry', 'completed', 203),
  ('b2c3d4e5-f6a7-8901-bcde-f23456789012',
   '一种新型基因编辑载体的构建方法',
   '本发明公开了一种新型基因编辑载体的构建方法。该载体基于CRISPR-Cas9系统，包含向导RNA表达盒和Cas9蛋白表达盒。通过优化启动子序列和核定位信号，显著提高了基因编辑效率。该载体可用于真核细胞和原核细胞的基因编辑，特别适用于哺乳动物细胞的基因敲除和基因敲入实验。细胞膜穿透肽的引入进一步提高了载体的转染效率。蛋白质折叠辅助因子的共表达有效降低了脱靶效应。',
   'biology', 'completed', 178),
  ('c3d4e5f6-a7b8-9012-cdef-345678901234',
   '量子点发光器件及其制造方法',
   '本发明涉及一种量子点发光器件及其制造方法。该器件采用核壳结构的量子点作为发光层，具有高亮度、宽色域和长寿命的特点。量子点材料选用硒化镉或磷化铟，通过热注入法合成。器件结构包括阳极、空穴传输层、量子点发光层、电子传输层和阴极。通过优化各功能层的厚度和界面处理，器件的外量子效率可达20%以上。该发明可应用于显示器、照明和光通信等领域。',
   'physics', 'completed', 165)
ON CONFLICT DO NOTHING;

-- 插入示例术语
INSERT INTO terms (paper_id, term, nested_structure, position_start, position_end, confidence, verified, domain)
VALUES
  ('a1b2c3d4-e5f6-7890-abcd-ef1234567890', '纳米材料', 
   '{"root": "纳米材料", "components": [{"text": "纳米", "type": "prefix", "meaning": "10^-9米级别"}, {"text": "材料", "type": "base", "meaning": "物质"}]}',
   5, 9, 0.95, true, 'chemistry'),
  ('a1b2c3d4-e5f6-7890-abcd-ef1234567890', '催化剂',
   '{"root": "催化剂", "components": [{"text": "催化", "type": "verb", "meaning": "加速反应"}, {"text": "剂", "type": "suffix", "meaning": "物质"}]}',
   13, 16, 0.92, true, 'chemistry'),
  ('a1b2c3d4-e5f6-7890-abcd-ef1234567890', '纳米颗粒',
   '{"root": "纳米颗粒", "components": [{"text": "纳米", "type": "prefix"}, {"text": "颗粒", "type": "base"}]}',
   28, 32, 0.88, true, 'chemistry'),
  ('a1b2c3d4-e5f6-7890-abcd-ef1234567890', '比表面积',
   '{"root": "比表面积", "components": [{"text": "比", "type": "prefix"}, {"text": "表面积", "type": "base"}]}',
   40, 44, 0.85, false, 'chemistry'),
  ('a1b2c3d4-e5f6-7890-abcd-ef1234567890', '氧化还原反应',
   '{"root": "氧化还原反应", "components": [{"text": "氧化", "type": "process"}, {"text": "还原", "type": "process"}, {"text": "反应", "type": "base"}]}',
   120, 126, 0.91, true, 'chemistry'),
  ('b2c3d4e5-f6a7-8901-bcde-f23456789012', '基因编辑',
   '{"root": "基因编辑", "components": [{"text": "基因", "type": "object"}, {"text": "编辑", "type": "action"}]}',
   8, 12, 0.94, true, 'biology'),
  ('b2c3d4e5-f6a7-8901-bcde-f23456789012', 'CRISPR-Cas9',
   '{"root": "CRISPR-Cas9", "components": [{"text": "CRISPR", "type": "system"}, {"text": "Cas9", "type": "protein"}]}',
   25, 36, 0.97, true, 'biology'),
  ('b2c3d4e5-f6a7-8901-bcde-f23456789012', '细胞膜',
   '{"root": "细胞膜", "components": [{"text": "细胞", "type": "base"}, {"text": "膜", "type": "structure"}], "validation": {"correct_split": true, "incorrect_alternatives": ["细/胞/膜", "细/胞膜"]}}',
   80, 83, 0.93, true, 'biology'),
  ('b2c3d4e5-f6a7-8901-bcde-f23456789012', '蛋白质折叠',
   '{"root": "蛋白质折叠", "components": [{"text": "蛋白质", "type": "molecule"}, {"text": "折叠", "type": "process"}]}',
   95, 100, 0.89, true, 'biology'),
  ('c3d4e5f6-a7b8-9012-cdef-345678901234', '量子点',
   '{"root": "量子点", "components": [{"text": "量子", "type": "physics_concept"}, {"text": "点", "type": "structure"}]}',
   5, 8, 0.96, true, 'physics'),
  ('c3d4e5f6-a7b8-9012-cdef-345678901234', '发光器件',
   '{"root": "发光器件", "components": [{"text": "发光", "type": "function"}, {"text": "器件", "type": "device"}]}',
   8, 12, 0.90, true, 'physics'),
  ('c3d4e5f6-a7b8-9012-cdef-345678901234', '核壳结构',
   '{"root": "核壳结构", "components": [{"text": "核", "type": "core"}, {"text": "壳", "type": "shell"}, {"text": "结构", "type": "base"}]}',
   25, 29, 0.87, true, 'physics')
ON CONFLICT DO NOTHING;

-- 插入示例知识块
INSERT INTO knowledge_blocks (term_id, source, source_url, content, relevance_score)
SELECT t.id, 'Wikipedia', 'https://zh.wikipedia.org/wiki/纳米材料',
  '纳米材料是指在三维空间中至少有一维处于纳米尺度范围（1-100nm）或由它们作为基本单元构成的材料。纳米材料具有表面效应、小尺寸效应、量子尺寸效应和宏观量子隧道效应等特殊性质。',
  0.95
FROM terms t WHERE t.term = '纳米材料' LIMIT 1;

INSERT INTO knowledge_blocks (term_id, source, source_url, content, relevance_score)
SELECT t.id, 'Wikipedia', 'https://zh.wikipedia.org/wiki/催化剂',
  '催化剂是一种能够改变化学反应速率而本身在反应前后保持不变的物质。催化剂通过降低反应的活化能来加速反应。根据相态可分为均相催化剂和多相催化剂。',
  0.93
FROM terms t WHERE t.term = '催化剂' LIMIT 1;

INSERT INTO knowledge_blocks (term_id, source, source_url, content, relevance_score)
SELECT t.id, 'Wikipedia', 'https://zh.wikipedia.org/wiki/细胞膜',
  '细胞膜（Cell membrane）又称质膜，是包围在细胞质外面的一层薄膜。主要由磷脂双分子层和蛋白质组成，具有选择透过性。细胞膜的主要功能包括物质运输、信号传导和细胞识别。注意：细胞是一个完整的术语单位，不应分割为"细"和"胞"。',
  0.96
FROM terms t WHERE t.term = '细胞膜' LIMIT 1;

INSERT INTO knowledge_blocks (term_id, source, source_url, content, relevance_score)
SELECT t.id, 'Wikipedia', 'https://zh.wikipedia.org/wiki/CRISPR',
  'CRISPR-Cas9是一种基因编辑技术，利用细菌的免疫防御机制来编辑基因。CRISPR代表"成簇规律间隔短回文重复序列"，Cas9是一种核酸酶蛋白。该系统可以精确定位并切割DNA序列。',
  0.98
FROM terms t WHERE t.term = 'CRISPR-Cas9' LIMIT 1;

INSERT INTO knowledge_blocks (term_id, source, source_url, content, relevance_score)
SELECT t.id, 'Wikipedia', 'https://zh.wikipedia.org/wiki/量子点',
  '量子点（Quantum dot）是一种半导体纳米晶体，其电子被限制在三维空间的极小区域内。由于量子限域效应，量子点表现出独特的光电性质，可通过控制尺寸来调节发光颜色。',
  0.94
FROM terms t WHERE t.term = '量子点' LIMIT 1;

-- 插入示例训练数据
INSERT INTO training_data (paper_id, input_text, output_labels, augmented, augmentation_type, quality_score)
VALUES
  ('a1b2c3d4-e5f6-7890-abcd-ef1234567890',
   '该催化剂由纳米颗粒组成',
   '{"entities": [{"text": "催化剂", "start": 1, "end": 4, "label": "MATERIAL"}, {"text": "纳米颗粒", "start": 5, "end": 9, "label": "MATERIAL"}]}',
   false, null, 0.92),
  ('a1b2c3d4-e5f6-7890-abcd-ef1234567890',
   '该触媒由微米粒子组成',
   '{"entities": [{"text": "触媒", "start": 1, "end": 3, "label": "MATERIAL"}, {"text": "微米粒子", "start": 4, "end": 8, "label": "MATERIAL"}]}',
   true, 'synonym_replacement', 0.85),
  ('b2c3d4e5-f6a7-8901-bcde-f23456789012',
   '通过优化启动子序列和核定位信号',
   '{"entities": [{"text": "启动子", "start": 4, "end": 7, "label": "BIO_ELEMENT"}, {"text": "核定位信号", "start": 10, "end": 15, "label": "BIO_ELEMENT"}]}',
   false, null, 0.90),
  ('c3d4e5f6-a7b8-9012-cdef-345678901234',
   '量子点材料选用硒化镉或磷化铟',
   '{"entities": [{"text": "量子点", "start": 0, "end": 3, "label": "MATERIAL"}, {"text": "硒化镉", "start": 8, "end": 11, "label": "COMPOUND"}, {"text": "磷化铟", "start": 12, "end": 15, "label": "COMPOUND"}]}',
   false, null, 0.94)
ON CONFLICT DO NOTHING;

-- 插入示例训练历史
INSERT INTO training_history (model_name, model_version, domain, accuracy, precision_score, recall_score, f1_score, loss, epoch, benchmark_score, training_samples, training_duration, notes)
VALUES
  ('TermExtractor-Base', 'v1.0', 'chemistry', 0.78, 0.75, 0.80, 0.77, 0.45, 10, 0.72, 1000, 3600, '基线模型'),
  ('TermExtractor-Base', 'v1.1', 'chemistry', 0.82, 0.80, 0.83, 0.81, 0.38, 15, 0.78, 1500, 5400, '增加训练数据'),
  ('TermExtractor-Base', 'v1.2', 'chemistry', 0.85, 0.84, 0.86, 0.85, 0.32, 20, 0.82, 2000, 7200, '引入知识增强'),
  ('TermExtractor-Base', 'v1.3', 'chemistry', 0.88, 0.87, 0.89, 0.88, 0.28, 25, 0.85, 2500, 9000, '优化分词规则'),
  ('TermExtractor-Bio', 'v1.0', 'biology', 0.76, 0.74, 0.78, 0.76, 0.48, 10, 0.70, 800, 2800, '生物领域基线'),
  ('TermExtractor-Bio', 'v1.1', 'biology', 0.81, 0.79, 0.82, 0.80, 0.40, 15, 0.76, 1200, 4200, '增加训练数据'),
  ('TermExtractor-Bio', 'v1.2', 'biology', 0.84, 0.83, 0.85, 0.84, 0.35, 20, 0.80, 1600, 5600, '引入知识增强'),
  ('TermExtractor-Phys', 'v1.0', 'physics', 0.74, 0.72, 0.76, 0.74, 0.50, 10, 0.68, 600, 2100, '物理领域基线'),
  ('TermExtractor-Phys', 'v1.1', 'physics', 0.79, 0.77, 0.80, 0.78, 0.42, 15, 0.74, 900, 3200, '增加训练数据')
ON CONFLICT DO NOTHING;

-- 插入示例Pipeline运行记录
INSERT INTO pipeline_runs (paper_id, status, current_stage, stages_completed)
VALUES
  ('a1b2c3d4-e5f6-7890-abcd-ef1234567890', 'completed', 'validation',
   '["upload", "domain_classification", "term_extraction", "knowledge_retrieval", "data_generation", "validation"]'),
  ('b2c3d4e5-f6a7-8901-bcde-f23456789012', 'completed', 'validation',
   '["upload", "domain_classification", "term_extraction", "knowledge_retrieval", "data_generation", "validation"]'),
  ('c3d4e5f6-a7b8-9012-cdef-345678901234', 'running', 'knowledge_retrieval',
   '["upload", "domain_classification", "term_extraction"]')
ON CONFLICT DO NOTHING;
