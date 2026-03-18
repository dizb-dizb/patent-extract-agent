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

**增强数据：按比例分层、类型全覆盖**（`augment_from_evidence.py --stratified`）：  
- 按训练集中**各实体类型的 span 频次**分配增强条数（总量 ≈ 原句数 × `--aug-ratio`，且每种类型至少 `--min-per-label` 条），避免少数类型被 evidence 增强「饿死」。  
- 优先用 evidence snippet 生成增强；某类型**没有可用 snippet** 时，用含该类型的**原训练句镜像**兜底（`_aug_meta.source=original_mirror`），保证论文/实验中可声明「每种类型均在增强管线中有体现」。  
- 批量重生成三数据集：`python augment_from_evidence.py --input-dir data/dataset/split --output-dir data/dataset/split --stratified`

### 2. 评测指标 (Metrics)

| 指标 | 说明 |
|------|------|
| **Flat F1** | 最长跨度 F1，用于传统 BIO 模型与 Span 模型的单层公平对决 |
| **Nested Micro-F1** | 嵌套微平均 F1，严格要求 [start, end, label] 完美匹配才算正确 |
| **BWT** | Backward Transfer（后向转移率），评估温习机制防止灾难性遗忘的能力 |

---

## 二、核心主实验：架构演进与消融对比

**实验目的**：证明「大模型底座 + Span 跨度解码 + Agent 图谱规则增强」这一组合在小样本场景下的必要性。

### 主实验（全数据集，无原型网络/无 SCL）

| 代号 | 模型底座 | 解码架构 | 数据策略 | 验证意图 |
|------|----------|----------|----------|----------|
| **B1** | BiLSTM | CRF (BIO 序列) | 仅原始 | 最弱基线 |
| **B2** | BERT-base | CRF (BIO 序列) | 仅原始 | BIO 基线 |
| **B2r** | RoBERTa-base | CRF (BIO 序列) | 仅原始 | B2 的 RoBERTa 版本 |
| **B-Span** | BERT-base | Span 分类器（无原型） | 仅原始 | 有监督 Span 对照 |
| **B-Span+Aug** | BERT-base | Span 分类器（无原型） | + Agent 增强 | 增强数据 + 无原型 |

### 梯度+隔离实验（n=10/100/1000 样本 + meta-train/meta-test 类别隔离）

原型网络相关实验全部在梯度+隔离统一脚本 `run_gradient_isolate_unified.py` 中执行：

| 代号 | 模型底座 | 原型 | 增强 | 冻结 | 说明 |
|------|----------|------|------|------|------|
| **B4** | BERT-base | 有 | 无 | 无 | 强基线 |
| **B4r** | RoBERTa-base | 有 | 无 | 无 | B4 对比 |
| **B4f** | BERT-base | 有 | 无 | 有 | 未微调对照 |
| **B4rf** | RoBERTa-base | 有 | 无 | 有 | 未微调对照 |
| **B5** | BiLSTM | 有 | 有 | 无 | 反向论证 |
| **B-Span** | BERT-base | 无 | 无 | — | 仅梯度（无 isolate） |
| **B-Span+Aug** | BERT-base | 无 | 有 | — | 仅梯度（无 isolate） |
| **Ours** | BERT-base | 有 | 有 | 无 | 核心方法 |
| **Ours-r** | RoBERTa-base | 有 | 有 | 无 | 核心方法 |

**「Ours」指什么**：**原型网络（Prototypical Span NER）+ BERT-base + Agent 分层增强训练数据**（`train_fewshot_proto_span.py`，数据为 `{ds}_train_augmented.jsonl`）。与 **B-Span+Aug** 的区别：Ours 用 **episodic N-way K-shot + 原型头**；B-Span+Aug 用 **有监督 Span 多类分类（无原型）**。**Ours-r** 将编码器换为 RoBERTa-base，其余与 Ours 一致。

**增强数据更新后仅重跑 B-Span+Aug 的 n=10、n=100**：`python scripts/run_bspan_aug_n10_n100.py --multi-gpu`；AutoDL：`python upload_and_run_autodl.py --mode bspan_aug_n10_100`（需先在远端用 `--stratified` 重生成 `*_train_augmented.jsonl`）。

> **编码器说明**：三个基准数据集（Few-NERD、GENIA、CHEMDNER）均为英文数据集，使用英文预训练模型（`bert-base-cased` / `roberta-base`）。
>
> **SCL 已移除**：监督对比学习（Supervised Contrastive Loss）已从代码中完全移除，所有实验均不使用 SCL。

---

## 三、数据流与隔离策略

- **数据集隔离**：fewnerd、genia、chemdner 分别训练与评测，不做混合。
- **数据准备流程**：build_dataset → convert_with_evidence → split_evidence_by_dataset → augment_from_evidence
- **Agent 增强**：基于证据链检索得到的 snippet，在 snippet 中定位术语生成新训练样本，扩充 few-shot 支持集

---

## 四、实现映射

| 实验 | 训练脚本 | 编码器 | 数据路径 | artifacts 目录 |
|------|----------|--------|----------|----------------|
| B1 | train_bilstm_crf.py | — | data/benchmarks/{ds}/train.jsonl | run_bilstm_crf/{ds} |
| B2 | train_seq_ner.py | bert-base-cased | data/benchmarks/{ds}/train.jsonl | run_seq_ner/{ds} |
| B2r | train_seq_ner.py | roberta-base | data/benchmarks/{ds}/train.jsonl | run_seq_ner_roberta/{ds} |
| B3 | train_fewshot_proto_span.py (encoder=bilstm) | — | data/benchmarks/{ds}/train.jsonl | run_proto_span_bilstm/{ds} |
| B4 | train_fewshot_proto_span.py | bert-base-cased | data/benchmarks/{ds}/train.jsonl | run_proto_span/{ds} |
| B4r | train_fewshot_proto_span.py | roberta-base | data/benchmarks/{ds}/train.jsonl | run_proto_span_roberta/{ds} |
| B5 | train_fewshot_proto_span.py (encoder=bilstm) | — | data/dataset/split/{ds}_train_augmented.jsonl | run_proto_span_bilstm_aug/{ds} |
| B-Span | train_span_ner.py | bert-base-cased | data/benchmarks/{ds}/train.jsonl | run_span_ner/{ds} |
| B-Span+Aug | train_span_ner.py | bert-base-cased | data/dataset/split/{ds}_train_augmented.jsonl | run_span_ner_aug/{ds} |
| Ours | train_fewshot_proto_span.py | bert-base-cased | data/dataset/split/{ds}_train_augmented.jsonl | run_proto_span_aug/{ds} |
| Ours-r | train_fewshot_proto_span.py | roberta-base | data/dataset/split/{ds}_train_augmented.jsonl | run_proto_span_roberta_aug/{ds} |

## 五、数据增强 + 有无原型网络对比（核心消融）

**实验重点**：在梯度+隔离统一实验中对比「有原型网络 vs 无原型网络」。

| 对比 | 模型 | 训练范式 | 说明 |
|------|------|----------|------|
| 无原型 | B-Span / B-Span+Aug | 有监督全量（train_span_ner） | Span 表示 + 线性分类头，标准交叉熵 |
| 有原型 | B4 / Ours / Ours-r | episodic N-way K-shot（train_fewshot_proto_span） | Span 表示 + 原型头（支持集均值），距离度量分类 |

**数据梯度 + 类别隔离（与原型流程对齐）**：

| 环节 | 含义 |
|------|------|
| **n=10/100/1000** | 仅约束 **meta-train**：从**全量** `train.jsonl`（或增强集）中抽取 **至多 n 条「含 meta-train 类型标注」的句子** 作为训练句池；若存在足够句子，优先选**不含 meta-test 类型标注**的句，减少标注泄漏。 |
| **评测** | **meta-test（新类型）**：从全量数据中另建 **评测句池**（含 `test_labels` 的句子，与训练句上下文去重，默认最多 8000 条），在此池上组 **N-way K-shot episode**，`use_test_labels=True`。 |
| **原型距离** | 与实现一致：support span 嵌入 → 按类 **均值** 为原型（`compute_prototypes`）→ query 与原型 **余弦相似度 / 负欧氏距离**（`compute_logits`）→ argmax 分类 + NONE 槽。训练仅在 meta-train 类 episode 上更新编码器；验证仅在 meta-test 类 episode 上计 F1。 |

脚本：`run_gradient_isolate_unified.py`；实现：`train_fewshot_proto_span.py`（传 `--train-labels` / `--test-labels` 且 `--max_train_samples`>0 时自动采用上表协议）。输出：`artifacts/run_*_n{10,100,1000}_isolate/`。

**无隔离小样本对照（仅 n=10、n=100）**：对 fewnerd、genia 在相同 n 条训练样本下，**额外**跑一轮原型网络实验且**不传** `--train-labels`/`--test-labels`（评测与训练共享实体类型池，非跨类泛化）。结果目录为 `artifacts/run_*_n10/`、`run_*_n100/` 下对应数据集子目录，可与 `_n10_isolate`、`_n100_isolate` 对照。chemdner 仅 1 类，本身无隔离，仍仅用 `_n{n}`。

> **训练方式**：同上，且 **n 仅限制 meta-train 句数**；评测始终在 **新类型（test_labels）大句池** 上进行，与原型网络「train 类 / test 类 episodic 分离」一致。少样本 n 下 F1 仍可能低于全量，但不再因「验证子集无新类型 span」而系统性为 0。

> 全数据集场景下不再单独运行原型网络主实验，原型网络的价值通过少样本梯度实验验证。

---

## 五.1 实验中断原因与预防（AutoDL）

梯度+隔离统一实验在远程 GPU 上长时间运行，可能**突然中断**的常见原因与应对：

| 原因 | 说明 | 预防/应对 |
|------|------|-----------|
| **实例关机/欠费** | AutoDL 余额不足、按量计费到期或手动关机 | 保持余额、用「无卡模式」或开机后尽快续费；用 `python run_gradient_isolate_on_autodl.py` 重新启动 |
| **OOM（显存不足）** | 某次实验 batch 或模型过大导致进程被系统 kill | 减小 `--batch_size`、`--max_episodes`，或单卡只跑部分数据集 |
| **脚本异常退出** | `run_gradient_isolate_unified.py` 或子进程报错未捕获 | 查看 `logs/run_gradient_isolate.log` 末尾；修复后重新运行，脚本会从未完成的矩阵项继续（需手动从断点重跑或重跑全量） |
| **SSH 断开导致误杀** | 若在 SSH 前台直接跑（未用 nohup），关终端会杀进程 | 一律用 `nohup ... &` 或 `run_gradient_isolate_on_autodl.py` 在后台启动 |
| **网络/连接超时** | 本地用 paramiko 启动时，若等待后台命令输出过久会 TimeoutError | 已改为后台启动后立即返回；若仍超时可在 AutoDL 终端内手动执行 `nohup bash scripts/autodl_run_gradient_isolate.sh >> logs/run_gradient_isolate.log 2>&1 &` |

**续跑**：中断后在本机执行 `python run_gradient_isolate_on_autodl.py` 会先 pkill 旧进程再重新启动整轮；如需从断点续跑需自行改脚本或只跑未完成的数据集/n。

**数据与模型参数放入数据盘**：  
在 AutoDL 上若存在数据盘 `/root/autodl-tmp`，通过 `upload_and_run_autodl.py` 或 `run_gradient_isolate_on_autodl.py` 启动时，会自动将 `data/`、`artifacts/`、`logs/` 以及预训练模型目录 `/root/models` 迁移到 `/root/autodl-tmp/patent-extract-agent/`（或对应子目录），并在工作目录/根目录建立软链接，读写仍走原路径，避免系统盘占满。也可在远程执行 `bash scripts/autodl_run_gradient_isolate.sh` 或 `bash scripts/autodl_run_all_until_done.sh`，脚本开头会做同样挂载。

**持续运行直到实验完成**：  
使用 `python upload_and_run_autodl.py --mode all`（默认）会执行 `scripts/autodl_run_all_until_done.sh`：先挂载数据盘（含 models）、释放已有权重、再依次运行「剩余主实验」→「梯度+隔离统一实验」，单次 nohup 内跑完所有实验，保持一直运行直到全部完成。

**参数是否释放**：  
是，有两层：  
1. **得到实验数据后自动释放**：各训练脚本在写入 `metrics.json` 后会自动删除当次运行的 `model.pt` / `pytorch_model.bin` / `tokenizer.json`，只保留指标与配置，形成「跑完即释放」的固定流程。  
2. **启动前批量释放**：通过 `upload_and_run_autodl.py` 或 `run_gradient_isolate_on_autodl.py` 启动前会执行 `scripts/release_artifact_models.py`，清理历史实验的大文件。手动执行：`python scripts/release_artifact_models.py --artifacts-dir <workdir>/artifacts`。

---

## 六、扩展能力（已实现）

| 能力 | 实现 | 用法 |
|------|------|------|
| Meta-Train/Meta-Test 类别隔离 | --train_labels A,B,C --test_labels D,E,F | 训练集与测试集类别无交集 |
| Flat F1 / Nested F1 | metrics.json 含 flat_f1, nested_micro_f1 | 降维打击与嵌套极限双指标 |
| BWT 持续学习 | scripts/run_continual.py | 温习机制防遗忘 |
| Zero/One-shot OOD | scripts/run_ood_oneshot.py | 未见类别 1-shot 泛化评测 |
