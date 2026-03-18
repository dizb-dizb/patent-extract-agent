# 完整实验 - 从零开始运行

## 收敛配置（默认）

为保证模型充分收敛、获取准确实验结果，默认使用以下超参：

| Baseline | fewnerd | genia | chemdner |
|----------|---------|-------|----------|
| B1 epochs | 8 | 12 | 15 |
| B2 epochs | 5 | 5 | 5 |
| B3/B4/B5/Ours epochs | 8 | 8 | 8 |
| Proto max_episodes/epoch | 1000 | 800 | 800 |
| BWT epochs_per_task | 3 | max_episodes 300 |

使用 `--epochs N` 可统一覆盖所有 epochs（快速测试用）。

---

## 一、AutoDL 云端运行（推荐）

### 1. SSH 登录

```bash
ssh -p <端口> root@<AutoDL地址>
# 示例: ssh -p 38815 root@connect.cqa1.seetacloud.com
```

### 2. 克隆项目（若未克隆）

```bash
cd /root
git clone <你的仓库URL> patent-extract-agent
cd patent-extract-agent
```

### 3. 环境安装

```bash
bash setup_autodl.sh
conda activate patent-agent
```

### 4. 运行完整实验

**方式 A：收敛导向脚本（推荐）**

```bash
# 完整实验：收敛配置，B1-B5 + Ours + BWT + OOD
bash scripts/autodl_run_convergence.sh

# 清空旧结果后重跑
RESET=1 bash scripts/autodl_run_convergence.sh

# 快速模式：B1-B4，跳过 evidence
bash scripts/autodl_run_convergence.sh --fast

# 或使用 run_from_scratch.sh（同上）
bash scripts/run_from_scratch.sh
```

**方式 B：Python 直接调用**

```bash
# 完整实验 + 清空旧结果
python scripts/run_full_experiment.py --reset --epochs 3

# 快速模式
python scripts/run_full_experiment.py --fast --epochs 3

# 使用本地预下载模型（AutoDL 上）
python scripts/run_full_experiment.py --reset --encoder /root/models/bert-base-cased --roberta-encoder /root/models/roberta-base
```

### 5. 后台运行 + 监控

```bash
# 后台运行
nohup bash scripts/run_from_scratch.sh --fast > logs/run_full.log 2>&1 &

# 实时监控
bash _monitor.sh
```

---

## 二、本地运行（需 GPU）

```bash
cd "E:\algorithm\Graduation Project"

# 激活环境
conda activate paper   # 或 patent-agent

# 完整实验
python scripts/run_full_experiment.py --reset --epochs 3

# 快速模式
python scripts/run_full_experiment.py --fast --skip-data --epochs 3
```

---

## 三、实验矩阵

| Baseline | 模型 | 数据 | 输出目录 |
|----------|------|------|----------|
| B1 | BiLSTM-CRF | original | artifacts/run_bilstm_crf/ |
| B2 | RoBERTa-CRF | original | artifacts/run_seq_ner/ |
| B3 | BiLSTM-Proto | original | artifacts/run_proto_span_bilstm/ |
| B4 | RoBERTa-Proto | original | artifacts/run_proto_span/ |
| B5 | BiLSTM-Proto | augmented | artifacts/run_proto_span_bilstm_aug/ |
| Ours | BERT-Proto+Aug | augmented | artifacts/run_proto_span_aug/ |

**能力测试**：战役三 BWT → `artifacts/continual/`；战役四 OOD → `artifacts/ood_oneshot/`

---

## 四、检查状态

```bash
python _progress.py
```
