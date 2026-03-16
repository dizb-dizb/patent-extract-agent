# 基础数据集构建说明

本文档说明如何从原始专利/论文文本构建用于 baseline 评测的基础数据集。

## 一、数据流概览

```
input/*.txt
    │
    ▼
[patent_annotator.main]  LLM 标注 + 词典 + 清洗
    │
    ▼
train_ready.jsonl  (sentence, entities)
    │
    ▼
[expand_train_ready / build_baseline_dataset step2]  嵌套子实体展开
    │
    ├── train_ready.jsonl (写回，含嵌套)
    └── train_spans.jsonl (context, spans)
    │
    ▼
[patent_agent_pipeline]  领域路由 + Wikipedia 检索 + 入库
    │
    ▼
train_spans_enhanced.jsonl  (带 evidence 块)
    │
    ▼
[augment_from_evidence]  基于 snippet 数据增强
    │
    ▼
train_spans_augmented.jsonl
    │
    ▼
[build_baseline_dataset step5]  train/val/test 划分
    │
    ▼
data_baseline/
  ├── train.jsonl
  ├── val.jsonl
  └── test.jsonl
```

## 二、一键构建（推荐）

```bash
# 从已有 train_ready.jsonl 开始（不重新标注）
python build_baseline_dataset.py

# 从 input/*.txt 重新标注（需配置 DEEPSEEK_API_KEY 等）
python build_baseline_dataset.py --from-annotate

# 跳过联网证据检索与增强（仅用 train_spans.jsonl）
python build_baseline_dataset.py --skip-evidence

# 自定义划分比例（默认 0.8,0.1,0.1）
python build_baseline_dataset.py --split 0.7,0.15,0.15 --seed 42
```

## 三、分步构建

### 1. 准备原始文本

将专利或论文全文放入 `input/` 目录，支持 `.txt` 文件：

```
input/
  ├── patent_001.txt
  └── paper_002.txt
```

### 2. 标注生成 train_ready.jsonl

```bash
python -m patent_annotator.main
```

- 需要配置 `DEEPSEEK_API_KEY` 或相应环境变量
- 输出：`train_ready.jsonl`，每行 `{"sentence": "...", "entities": [{"start", "end", "label", "text"}, ...]}`

### 3. 嵌套子实体展开

```bash
python expand_train_ready.py
```

或由 `build_baseline_dataset.py` 的 step2 自动执行：

- 在复合实体内部补充子术语 span（父实体 + 子实体）
- 输出：`train_ready.jsonl`（写回）、`train_spans.jsonl`

### 4. 联网证据检索（可选）

```bash
python patent_agent_pipeline.py
```

- 领域路由（chem/bio/phy）+ Wikipedia（主）+ DuckDuckGo（兜底）检索
- 输出：`train_spans_enhanced.jsonl`、`knowledge.db`、`knowledge_graph.json`、`viewer.html`、`graph_viewer.html`

### 5. 证据增强（可选）

```bash
python augment_from_evidence.py
```

- 基于 evidence.snippet 生成额外训练样本
- 输出：`train_spans_augmented.jsonl`

### 6. 划分 train/val/test

由 `build_baseline_dataset.py` 的 step5 完成，输出到 `data_baseline/`：

- `train.jsonl`：训练集
- `val.jsonl`：验证集（早停、超参）
- `test.jsonl`：测试集（最终评测）

## 四、数据格式

### train_ready.jsonl

```json
{"sentence": "放射性口腔黏膜炎致病机制...", "entities": [{"start": 0, "end": 7, "label": "Condition", "text": "放射性口腔黏膜炎"}, ...]}
```

### train_spans.jsonl / data_baseline/*.jsonl

```json
{"context": "放射性口腔黏膜炎致病机制...", "spans": [{"start": 0, "end": 7, "label": "Condition", "text": "放射性口腔黏膜炎"}, ...]}
```

### train_spans_enhanced.jsonl

在 spans 中增加 `evidence` 字段：

```json
{"context": "...", "spans": [{"start": 0, "end": 7, "label": "Condition", "text": "放射性口腔黏膜炎", "evidence": {"source": "wikipedia_zh", "snippet": "...", ...}}]}
```

## 五、知识图谱查看

从 `knowledge.db` 导出图谱 JSON（若已运行过 pipeline 可单独导出）：

```bash
python export_graph.py
```

在浏览器中打开 `graph_viewer.html`，点击「Load JSON」选择 `knowledge_graph.json`，即可查看术语-证据网络。

## 六、Baseline 使用方式

训练脚本优先从 `data_baseline/` 读取：

- `train_seq_ner.py`：RoBERTa BIO 序列标注
- `train_span_ner.py`：Span 分类（支持嵌套）
- `train_fewshot_proto_span.py`：N-way K-shot 原型网络

若未划分，可直接使用 `train_spans.jsonl` 或 `train_spans_augmented.jsonl`，在训练时内部划分。
