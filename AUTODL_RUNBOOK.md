## AutoDL 上从零跑通（SSH + 训练 + 指标回传）

### 0. 准备

- **开一台 AutoDL GPU 实例**（建议带 conda 的镜像）
- **确保可 SSH 登录**（拿到 host/user/port/密钥）

### 1. 在 AutoDL 上安装环境

在 AutoDL 的终端执行（一次即可）：

```bash
cd /root
git clone <你的项目仓库或上传代码到某目录>
cd <项目目录>

# 创建 conda 环境并安装依赖（不含 torch）
bash setup_autodl.sh
conda activate patent-agent

# 安装 torch（按 nvidia-smi 显示的 CUDA 版本选择）
# CUDA 12.1/12.4: cu121 或 cu124
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

### 2. 在本地生成训练数据（可选：增强）

在本地 Windows：

```powershell
cd "e:\algorithm\Graduation Project"

# 生成带 evidence 的增强数据集（会写 knowledge.db + train_spans_enhanced.jsonl + viewer.html）
python patent_agent_pipeline.py

# 基于 evidence.snippet 生成额外样本
python augment_from_evidence.py
```

输出：
- `train_spans_enhanced.jsonl`
- `train_spans_augmented.jsonl`（若运行增强）

### 3. 通过 SSH 一键提交到 AutoDL 跑训练，并拉回 metrics

在本地设置环境变量（建议写入 `.env` 或系统环境变量）：

```powershell
$env:CLOUD_SSH_HOST="你的autodl主机"
$env:CLOUD_SSH_PORT="22"
$env:CLOUD_SSH_USER="root"
$env:CLOUD_SSH_KEY_PATH="C:\Users\你\.ssh\id_rsa"
$env:CLOUD_REMOTE_WORKDIR="/root/patent-agent-jobs"
$env:CLOUD_REMOTE_PYTHON="python3"
# 如果你需要激活 conda
$env:CLOUD_REMOTE_SETUP_CMD="source /opt/conda/bin/activate patent-agent"
```

然后运行：

```powershell
cd "e:\algorithm\Graduation Project"
python cloud_ssh_runner.py
```

成功后你会得到：
- 本地下载目录：`artifacts/cloud/<job_id>/...`
- 回传指标：`artifacts/metrics/cloud_<job_id>.json`
- 远端运行日志：`artifacts/cloud/<job_id>/<job_id>/run.log`（若训练失败优先看这个）

并可更新报告汇总表：

```powershell
python report_generator.py
```

### 4. 直接在 AutoDL 上本地训练（不走 SSH runner）

如果你把数据文件也传到 AutoDL（例如 `scp` 上传），可直接运行：

```bash
conda activate patent-agent

# 原型网络 Few-shot 训练（N-way K-shot）
python train_fewshot_proto_span.py --data train_spans_augmented.jsonl --encoder hfl/chinese-roberta-wwm-ext --epochs 2 --output_dir run_out --n_way 5 --k_shot 5

# 或 Span 分类（非 episodic）
python train_span_ner.py --data train_spans_augmented.jsonl --encoder hfl/chinese-roberta-wwm-ext --epochs 2 --output_dir run_out

cat run_out/metrics.json
```

