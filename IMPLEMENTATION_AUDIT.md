# 项目实现完整性审计

本文档汇总各计划实现状态与缺口，便于追踪与补充。

---

## 一、核查转换逻辑与证据链增强

| 计划项 | 状态 | 实现位置 |
|--------|------|----------|
| 转换阶段仅做直接格式转化 | ✅ | `convert_fewnerd.py`、`convert_genia.py`、`convert_chemdner.py` 输出 `{context, spans}` |
| 构建转化后的 span 训练数据集 | ✅ | `scripts/build_dataset.py` → `data/dataset/unified/` |
| run_baseline 支持 `--dataset unified` | ✅ | `scripts/run_baseline.py` |
| convert_with_evidence（证据链检索） | ✅ | `scripts/convert_with_evidence.py` |
| 无证据 span 写入 no_evidence_for_review.jsonl | ✅ | `convert_with_evidence.py` 输出 |
| 删除 verify_term_segmentation 等启发式 | ✅ | 已从 `patent_agent_pipeline.py` 移除 |
| 删除 term_verifications 表 | ✅ | 已从 `patent_agent_pipeline.py` 移除 |
| 人工查验入口 | ✅ | `scripts/review_no_evidence.py`（CLI + HTML） |

---

## 二、多级级联验证引擎

| 计划项 | 状态 | 实现位置 |
|--------|------|----------|
| Tier 0 Wikipedia ZH | ✅ | `patent_agent_pipeline.py` |
| Tier 1 DuckDuckGo | ✅ | `patent_agent_pipeline.py` |
| Tier 2 PubChem (chem) | ✅ | `verification_cascade.py` → `retrieve_pubchem` |
| Tier 2 Europe PMC (bio) | ✅ | `verification_cascade.py` → `retrieve_europepmc` |
| Tier 2 ArXiv (phy) | ✅ | `verification_cascade.py` → `retrieve_arxiv` |
| Tier 3 跨语言检索 | ✅ | `verification_cascade.py` → `retrieve_cross_lingual` |
| Tier 4 构词推导 | ✅ | `verification_cascade.py` → `derive_by_rules`；证据链 `evidence_chain.py` |
| 环境变量开关 | ✅ | `ENABLE_TIER2_ACADEMIC`、`ENABLE_TIER3_CROSS_LINGUAL`、`ENABLE_TIER4_DERIVATION` |
| graph_viewer 新 source 图例 | ✅ | `graph_viewer.html` |

---

## 三、四基准 Baseline 与消融实验

| 计划项 | 状态 | 实现位置 |
|--------|------|----------|
| Few-NERD 下载与转换 | ✅ | `scripts/download_benchmarks.py`、`convert_fewnerd.py` |
| GENIA 下载与转换 | ✅ | `scripts/download_benchmarks.py`、`convert_genia.py` |
| CHEmdNER 下载与转换 | ✅ | `scripts/download_benchmarks.py`、`convert_chemdner.py` |
| run_baseline 统一入口 | ✅ | `scripts/run_baseline.py` |
| run_ablations 消融调度 | ✅ | `scripts/run_ablations.py` |
| unified 数据集支持 | ✅ | `--dataset unified` |

---

## 四、PAT 工程落地路线图 (MILESTONE_ROADMAP.md)

| Milestone | 状态 | 实现位置 |
|-----------|------|----------|
| M1 规则认知 Agent | ✅ | `rule_mining_agent.py` |
| M2 数据增强 data_synthesizer | ⏳ 待实现 | - |
| M3 Few-shot 采样器扩展 | ⏳ 可扩展 | `fewshot/episode_dataset.py` |
| M4 底层模型训练 | ✅ | `train_fewshot_proto_span.py`、`train_span_ner.py` |

---

## 五、按数据集拆分与 Agent 增强流程

| 步骤 | 脚本 | 输入 | 输出 |
|------|------|------|------|
| 1 | `scripts/split_evidence_by_dataset.py` | unified/*_with_evidence.jsonl | split/{ds}_{split}_with_evidence.jsonl |
| 2 | `augment_from_evidence.py --input-dir data/dataset/split --output-dir data/dataset/split` | split/*_with_evidence | split/*_augmented.jsonl |
| 3 | `run_baseline.py --data-strategy augmented` | split/{ds}_train_augmented.jsonl | artifacts/run_* |

**衔接说明**：Ours 实验需先执行步骤 1、2，再运行 `run_baseline --dataset fewnerd --mode fewshot --data-strategy augmented`。

---

## 六、关键脚本与用法

| 脚本 | 用途 |
|------|------|
| `scripts/download_benchmarks.py` | 下载 Few-NERD、GENIA、CHEmdNER |
| `scripts/build_dataset.py` | 合并三基准为 unified 数据集 |
| `scripts/convert_with_evidence.py` | 对 span 做证据链检索，无证据写入待查验 |
| `scripts/split_evidence_by_dataset.py` | 按数据集拆分 *_with_evidence.jsonl |
| `augment_from_evidence.py --input-dir ... --output-dir ...` | 批量增强 split 下的 *_with_evidence |
| `scripts/review_no_evidence.py` | 人工查验无证据 span（HTML/CLI） |
| `scripts/run_baseline.py --dataset unified` | 在 unified 上训练 |
| `scripts/run_baseline.py --data-strategy augmented` | Ours：使用 Agent 增强数据 |
| `scripts/run_baseline.py --mode seq` | B2：BIO token 分类基线 |
| `scripts/run_baseline.py --scl-weight 0.1` | 监督对比学习（度量层） |
| `scripts/run_baseline.py --train-labels A,B --test-labels C,D` | Meta-Train/Meta-Test 类别隔离 |
| `scripts/run_ablations.py --data-strategy augmented` | 消融含 Ours |
| `scripts/run_continual.py` | 持续学习 + BWT 评测 |
| `scripts/run_ood_oneshot.py --test_data ... --k_shot 1` | Zero/One-shot OOD 评测 |
| `patent_agent_pipeline.py` | 主流程：train_ready → 级联检索 → 图谱导出 |
| `rule_mining_agent.py --demo` | Milestone 1 规则提取演示 |

---

## 七、双模型策略（规则核心 vs 普通分割）

| 任务类型 | tier | 模型 | 说明 |
|----------|------|------|------|
| 规则提取 | quality | GEMINI_QUALITY_MODEL / DEEPSEEK | rule_mining_agent |
| 证据链结论 | quality | 同上 | evidence_chain._llm_fill_chain |
| 构词推导 | quality | 同上 | verification_cascade._llm_derive_snippet |
| 翻译（中→英、英→中） | performance | GEMINI_PERFORMANCE_MODEL / FAST_MODEL | verification_cascade |
| 术语标注 / 标签生成 | performance | 同上 | patent_annotator/chains |

环境变量：`GEMINI_QUALITY_MODEL`（默认 gemini-3.1-pro-preview）、`GEMINI_PERFORMANCE_MODEL`（默认 gemini-3-flash-preview）

---

## 八、待补充项（可选）

1. **M2 data_synthesizer**：基于规则图谱的伪数据生成（路线图规划）。
2. **verdicts 持久化**：`review_no_evidence.py` 当前将查验结果存 localStorage，可扩展为导出 `verdicts.jsonl` 供后续训练过滤。
3. **convert_with_evidence 与训练衔接**：已通过 split_evidence_by_dataset + augment_from_evidence + run_baseline --data-strategy augmented 实现。

---

*审计日期：2025-02*
