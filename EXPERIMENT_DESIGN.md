# 面向专利术语提取的 Agent 强化小样本学习实验设计方案

## 一、实验总体目标与评测基准设定

### 1. 实验数据集配置

为保证实验的权威性与领域交叉验证，采用三大公开数据集配合严格隔离策略：

| 数据集 | 定位 | 验证目标 |
|--------|------|----------|
| **Few-NERD** | 基础通用集 | 通用小样本抽取能力 |
| **GENIA** | 生物医药集 | 英文复杂嵌套与生物医学规则抽取 |
| **CHEMDNER** | 化学专利集 | 长链化学式的规则路由与提取 |

**数据降维处理**：为保证传统模型的公平对比，构建一套「降维打平（Flattened）」的纯 BIO 格式副本，仅保留最长跨度（Outermost Span）。

### 2. 评测指标 (Metrics)

| 指标 | 说明 |
|------|------|
| **Flat F1** | 最长跨度 F1，用于传统 BIO 模型与 Span 模型的单层公平对决 |
| **Nested Micro-F1** | 嵌套微平均 F1，严格要求 [start, end, label] 完美匹配才算正确 |
| **BWT** | Backward Transfer（后向转移率），评估温习机制防止灾难性遗忘的能力 |

---

## 二、核心主实验：架构演进与消融对比

**实验目的**：证明「大模型底座 + Span 跨度解码 + Agent 图谱规则增强」这一组合在小样本场景下的必要性。

| 代号 | 模型底座 | 解码架构 | 数据策略 | 验证意图与预期表现 |
|------|----------|----------|----------|--------------------|
| **B1** | BiLSTM | CRF (BIO 序列) | 仅原始极小样本 | 最弱基线。证明传统模型在脱离海量标注且面对嵌套结构时的彻底失效 |
| **B2** | RoBERTa-WWM | CRF (BIO 序列) | 仅原始极小样本 | 基础基线。证明引入预训练大模型后特征提取增强，但 BIO 机制仍是识别嵌套的理论瓶颈 |
| **B3** | BiLSTM | Span (原型网络) | 仅原始极小样本 | 证明更换跨度解码后具备嵌套识别能力，但 BiLSTM 的表征能力无法支撑精准的空间距离度量 |
| **B4** | RoBERTa-WWM | Span (原型网络) | 仅原始极小样本 | 强大对比基线。验证现代预训练模型 + 度量学习的基准实力 |
| **B5** | BiLSTM | Span (原型网络) | + Agent 规则增强 | 反向论证。证明传统模型由于缺乏全局注意力，强行喂入 Agent 规则块会导致噪声干扰，性能不升反降 |
| **Ours** | RoBERTa-WWM | Span (原型网络) | + Agent 规则增强 | 核心方法。大模型底座 + Span 解码 + Agent 规则增强，预期显著优于 B4 |

---

## 三、数据流与隔离策略

- **数据集隔离**：fewnerd、genia、chemdner 分别训练与评测，不做混合。
- **数据准备流程**：build_dataset → convert_with_evidence → split_evidence_by_dataset → augment_from_evidence
- **Agent 增强**：基于证据链检索得到的 snippet，在 snippet 中定位术语生成新训练样本，扩充 few-shot 支持集

---

## 四、实现映射

| 实验 | 训练脚本 | 数据路径 |
|------|----------|----------|
| B1 | train_bilstm_crf.py | data/benchmarks/{ds}/train.jsonl |
| B2 | train_seq_ner.py | data/benchmarks/{ds}/train.jsonl |
| B3 | train_fewshot_proto_span.py (encoder=bilstm) | data/benchmarks/{ds}/train.jsonl |
| B4 | train_fewshot_proto_span.py | data/benchmarks/{ds}/train.jsonl |
| B5 | train_fewshot_proto_span.py (encoder=bilstm) | data/dataset/split/{ds}_train_augmented.jsonl |
| Ours | train_fewshot_proto_span.py | data/dataset/split/{ds}_train_augmented.jsonl |

## 五、扩展能力（已实现）

| 能力 | 实现 | 用法 |
|------|------|------|
| 监督对比学习 (SCL) | train_fewshot_proto_span --scl_weight 0.1 | 难负样本推远，决策边界更清晰 |
| Meta-Train/Meta-Test 类别隔离 | --train_labels A,B,C --test_labels D,E,F | 训练集与测试集类别无交集 |
| Flat F1 / Nested F1 | metrics.json 含 flat_f1, nested_micro_f1 | 降维打击与嵌套极限双指标 |
| BWT 持续学习 | scripts/run_continual.py | 温习机制防遗忘 |
| Zero/One-shot OOD | scripts/run_ood_oneshot.py | 未见类别 1-shot 泛化评测 |
