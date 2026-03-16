# PAT 项目工程落地路线图

## 流程概览

```
BIO/原始数据 → [LLM 标注 + 规则提取] → spans + 规则树 → 数据增强 → Few-shot 采样 → Span RoBERTa 训练
```

## Milestone 1：认知引擎跑通 (Agent & 规则图谱层)

**目标**：大模型读取基础训练集 JSON，挖掘「多条嵌套构词规则」，输出结构化规则树（NetworkX / Dict）。

**产出**：`rule_mining_agent.py`

**用法**：
```bash
# 单次示例测试（需 GEMINI_API_KEY、DEEPSEEK_API_KEY 或 FAST_MODEL_API_KEY）
python rule_mining_agent.py --demo

# 对 train_ready.jsonl 批量提取规则
python rule_mining_agent.py --input train_ready.jsonl --output artifacts/rule_trees.jsonl

# 限制条数
python rule_mining_agent.py --input train_spans.jsonl --output artifacts/rule_trees.jsonl --limit 10
```

**输入格式**：`{text, entities}` 或 `{context, spans}`，支持嵌套实体。

**输出**：`lexical_rules`、`concept_rules`、`composition_rules` → 可转为 NetworkX 图谱。

---

## Milestone 2：数据增强与组装 (RAG & 数据流水线)

**目标**：Agent 根据图谱规则进行排列组合，结合联网搜索，生成伪数据。

**产出**：`data_synthesizer.py`（待实现）

---

## Milestone 3：小样本数据采样器 (Few-Shot Sampler)

**目标**：N-way K-shot 采样，同时处理真实 Support Set 与 Agent 增强数据。

**产出**：`few_shot_dataset.py`（已有 `fewshot/episode_dataset.py` 可扩展）

---

## Milestone 4：底层模型训练与验证 (Span-based RoBERTa)

**目标**：Span 度量学习网络，连接 DataLoader 训练，输出 F1。

**产出**：`train_model.py`（已有 `train_fewshot_proto_span.py`、`train_span_ner.py`）

---

## 与现有流程衔接

| 阶段       | 现有脚本                    | 新增/扩展                    |
|------------|-----------------------------|------------------------------|
| 转换       | convert_*.py, build_dataset  | rule_mining_agent 可对输出运行 |
| 增强       | augment_from_evidence       | data_synthesizer（基于规则）  |
| 采样       | fewshot/episode_dataset     | 扩展支持增强数据混合          |
| 训练       | train_fewshot_proto_span    | 保持                         |
