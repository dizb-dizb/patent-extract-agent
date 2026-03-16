# AutoDL 云端训练部署指南

面向专利术语提取的 Agent 强化小样本学习实验，在 AutoDL 上完成 GPU 训练与测试。

---

## 快速开始（已有实例）

1. 在 `.env` 中配置 AutoDL SSH 信息（从 AutoDL 控制台复制）：
   ```
   CLOUD_SSH_HOST=connect.xx.autodl.com
   CLOUD_SSH_PORT=12345
   CLOUD_SSH_USER=root
   ```
2. 本地 PowerShell 一键连接：
   ```powershell
   .\scripts\connect_autodl.ps1
   ```
   或手动：
   ```powershell
   ssh -p <端口> root@<SSH地址>
   ```
3. 或使用 **一键部署**（推荐）：上传 + 自动安装环境 + 运行训练：
   ```powershell
   .\scripts\deploy_and_run.ps1 -Dataset fewnerd -Mode fewshot -MultiGpu
   ```

---

## 一键部署（deploy_and_run.ps1）

本地配置好 `.env` 后，可一键上传项目、自动安装环境（含 PyTorch CUDA）、并运行训练：

```powershell
# 仅部署并安装环境
.\scripts\deploy_and_run.ps1 -SetupOnly

# 部署 + 安装 + 运行 fewnerd fewshot（多 GPU）
.\scripts\deploy_and_run.ps1 -Dataset fewnerd -Mode fewshot -MultiGpu

# 部署 + 运行 Ours（augmented）
.\scripts\deploy_and_run.ps1 -Dataset fewnerd -Mode fewshot -DataStrategy augmented -MultiGpu

# 指定 CUDA 版本（默认 cu124）
.\scripts\deploy_and_run.ps1 -Dataset fewnerd -Mode fewshot -CudaVer cu121
```

脚本会排除 `3/`、`__pycache__`、`.git`、`node_modules` 后上传，远程执行 `setup_autodl.sh` 安装 PyTorch 与依赖。

---

## 一、连接 AutoDL

### 1. 创建实例

1. 登录 [AutoDL 控制台](https://www.autodl.com/)
2. 创建实例：选择 GPU（如 RTX 3090 / A100）、镜像（推荐 **PyTorch 2.x + CUDA 12.x**）
3. 记录实例信息：
   - **SSH 地址**：`region-xx.autodl.com` 或 `connect.xx.autodl.com`
   - **SSH 端口**：如 `12345`
   - **用户名**：通常为 `root`
   - **密码**：实例详情页获取

### 2. SSH 连接

```bash
# 格式
ssh -p <端口> root@<SSH地址>

# 示例
ssh -p 12345 root@connect.westb.seetacloud.com
```

首次连接输入密码，后续可使用 SSH 密钥免密登录。

---

## 二、上传项目

### 方式 A：Git 克隆（推荐）

若项目已推送到 Git 仓库：

```bash
cd /root
git clone <你的仓库地址> Graduation-Project
cd Graduation-Project
```

### 方式 B：本地 rsync / scp 同步

在**本地 Windows**（PowerShell）执行：

```powershell
# 使用 scp 上传整个项目（排除大文件）
scp -P <端口> -r "e:\algorithm\Graduation Project\*" root@<SSH地址>:/root/Graduation-Project/

# 或使用 rsync（若已安装）
# rsync -avz -e "ssh -p <端口>" --exclude "3/" --exclude "__pycache__" --exclude ".git" "e:\algorithm\Graduation Project/" root@<SSH地址>:/root/Graduation-Project/
```

### 方式 C：AutoDL 文件管理器

在 AutoDL 控制台使用「文件」→「上传」上传压缩包，再在终端解压：

```bash
cd /root
unzip Graduation-Project.zip
```

---

## 三、环境配置

### 1. 进入项目目录

```bash
cd /root/Graduation-Project
```

### 2. 一键安装（含 PyTorch）

`setup_autodl.sh` 会自动安装 PyTorch（按 `CUDA_VER`）和项目依赖：

```bash
# 默认 CUDA 12.4 (cu124)
bash setup_autodl.sh

# 或指定 CUDA 版本
CUDA_VER=cu121 bash setup_autodl.sh
```

**CUDA 13.0 说明**：PyTorch 暂无 cu130 wheel。若服务器为 CUDA 13.0，先尝试 `CUDA_VER=cu124`，多数情况下可兼容运行；若失败可试 `cu121`。

### 3. 手动安装（可选）

```bash
# 查看 CUDA 版本
nvidia-smi

# 手动安装 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

conda activate patent-agent
pip install -r requirements-autodl.txt
```

### 4. 配置环境变量（可选）

若需 LLM API（证据链、增强等）：

```bash
# 创建 .env 或 export
export GEMINI_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"
```

---

## 四、数据准备

确保以下数据已就绪（本地准备好后上传，或在云端重新生成）：

| 路径 | 说明 |
|------|------|
| `data/benchmarks/fewnerd/` | Few-NERD train/val/test.jsonl |
| `data/benchmarks/genia/` | GENIA train/val/test.jsonl |
| `data/benchmarks/chemdner/` | CHEMDNER train/val/test.jsonl |
| `data/dataset/unified/` | build_dataset 合并后的 unified |
| `data/dataset/split/` | split_evidence_by_dataset 拆分后的按数据集文件 |

若在云端从零构建：

```bash
# 1. 下载并转换基准数据（若脚本支持）
python scripts/download_benchmarks.py   # 若有

# 2. 合并 unified
python scripts/build_dataset.py

# 3. 证据增强与拆分（若已本地完成，直接上传 data/dataset/split/ 即可）
python scripts/split_evidence_by_dataset.py
```

---

## 五、运行训练

### B4 基线（RoBERTa + Span，原始数据）

```bash
python scripts/run_baseline.py --dataset fewnerd --mode fewshot --n_way 5 --k_shot 5
python scripts/run_baseline.py --dataset genia --mode fewshot --n_way 5 --k_shot 5
python scripts/run_baseline.py --dataset chemdner --mode fewshot --n_way 5 --k_shot 5
```

### Ours（RoBERTa + Span + Agent 增强）

**前置步骤**：需先执行 `split_evidence_by_dataset` 与 `augment_from_evidence` 生成 `*_train_augmented.jsonl`：

```bash
# 1. 拆分（若 unified 已有 *_with_evidence.jsonl）
python scripts/split_evidence_by_dataset.py

# 2. 增强（基于 evidence.snippet 生成额外样本）
python augment_from_evidence.py --input-dir data/dataset/split --output-dir data/dataset/split

# 3. 训练
python scripts/run_baseline.py --dataset fewnerd --mode fewshot --data-strategy augmented
python scripts/run_baseline.py --dataset genia --mode fewshot --data-strategy augmented
python scripts/run_baseline.py --dataset chemdner --mode fewshot --data-strategy augmented
```

### 监督学习模式

```bash
python scripts/run_baseline.py --dataset fewnerd --mode supervised --epochs 5
```

### 多 GPU（RTX 4090 多卡）

```bash
python scripts/run_baseline.py --dataset fewnerd --mode fewshot --multi-gpu
```

### 后台运行（防止断连）

```bash
nohup python scripts/run_baseline.py --dataset fewnerd --mode fewshot > logs/fewnerd.log 2>&1 &
```

---

## 六、结果与回传

- 训练输出：`artifacts/run_proto_span/{dataset}/` 或 `artifacts/run_span_ner/{dataset}/`
- 指标文件：`metrics.json`

从 AutoDL 下载结果到本地：

```powershell
# 在本地 PowerShell 执行
scp -P <端口> -r root@<SSH地址>:/root/Graduation-Project/artifacts "e:\algorithm\Graduation Project\"
```

---

## 七、SSH 一键提交（cloud_ssh_runner）

在本地配置好 `.env` 后，可一键上传数据、在云端跑训练并拉回结果：

```powershell
cd "e:\algorithm\Graduation Project"
python cloud_ssh_runner.py --mode proto_span
```

注意：`cloud_ssh_runner` 默认使用 `train_spans_augmented.jsonl`。若使用 split 后的 benchmark 数据（fewnerd/genia/chemdner），需先上传 `data/` 到云端，再在云端直接执行 `run_baseline.py`。

---

## 八、常见问题

| 问题 | 处理 |
|------|------|
| `CUDA out of memory` | 减小 `--n_way` / `--k_shot`，或换更大 GPU |
| CUDA 13.0 环境 | 使用 `CUDA_VER=cu124`，通常可兼容 |
| `No module named 'transformers'` | `pip install -r requirements-autodl.txt` |
| 数据路径不存在 | 检查 `data/benchmarks/`、`data/dataset/split/` 是否已上传 |
| 训练中断 | 使用 `nohup` 或 `screen`/`tmux` 后台运行 |

---

## 九、快速检查清单

- [ ] SSH 连接成功
- [ ] 项目已上传到 `/root/Graduation-Project`
- [ ] PyTorch + CUDA 安装正确（`python -c "import torch; print(torch.cuda.is_available())"`）
- [ ] `requirements-autodl.txt` 已安装
- [ ] `data/benchmarks/` 或 `data/dataset/split/` 存在
- [ ] `run_baseline.py` 可正常启动
