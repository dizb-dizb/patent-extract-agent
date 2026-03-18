"""
在 AutoDL 上运行梯度+隔离实验并核查结果。
用法: python run_gradient_isolate_on_autodl.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"')

def main():
    load_env()
    host = os.environ.get("CLOUD_SSH_HOST", "connect.cqa1.seetacloud.com")
    port = int(os.environ.get("CLOUD_SSH_PORT", "38815"))
    user = os.environ.get("CLOUD_SSH_USER", "root")
    password = os.environ.get("CLOUD_SSH_PASSWORD", "CQtlwjJT2xIF")
    workdir = os.environ.get("CLOUD_REMOTE_WORKDIR", "/root/patent-extract-agent")

    try:
        import paramiko
    except ImportError:
        print("[fail] pip install paramiko")
        sys.exit(1)

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        c.connect(host, port, user, password, timeout=15)
    except Exception as e:
        print(f"[fail] 连接失败: {e}")
        sys.exit(1)

    def run(cmd, timeout=60):
        _, out, _ = c.exec_command(cmd, timeout=timeout)
        return out.read().decode(errors="replace").strip()

    print("=" * 60)
    print("  在 AutoDL 上运行梯度+隔离实验")
    print("=" * 60)

    # 1. 终止旧进程
    print("\n[1] 终止旧进程...")
    run("pkill -f 'train_|run_full|run_remaining|run_data_gradient|run_gradient_isolate' 2>/dev/null || true")
    run("sleep 3")
    print("  已清理")

    # 1.5 释放已有实验的模型参数，防止磁盘占满
    print("\n[1.5] 释放已有模型权重（保留 metrics.json）...")
    out = run(f"cd {workdir} && python scripts/release_artifact_models.py --artifacts-dir {workdir}/artifacts", timeout=120)
    if out:
        for line in out.split("\n"):
            if line.strip():
                print(f"  {line.strip()}")
    print("  已执行释放")

    # 2. 若有数据盘则挂载 data/artifacts/logs/models 到数据盘（含模型参数）
    bind_cmd = (
        f"if [ -d /root/autodl-tmp ] && [ ! -L {workdir}/data ]; then "
        f"mkdir -p /root/autodl-tmp/patent-extract-agent/data /root/autodl-tmp/patent-extract-agent/artifacts /root/autodl-tmp/patent-extract-agent/logs /root/autodl-tmp/patent-extract-agent/models; "
        f"[ -d {workdir}/data ] && [ ! -L {workdir}/data ] && mv {workdir}/data /root/autodl-tmp/patent-extract-agent/; "
        f"[ -d {workdir}/artifacts ] && [ ! -L {workdir}/artifacts ] && mv {workdir}/artifacts /root/autodl-tmp/patent-extract-agent/; "
        f"[ -d {workdir}/logs ] && [ ! -L {workdir}/logs ] && mv {workdir}/logs /root/autodl-tmp/patent-extract-agent/; "
        f"[ -d /root/models ] && [ ! -L /root/models ] && mv /root/models /root/autodl-tmp/patent-extract-agent/ && ln -sfn /root/autodl-tmp/patent-extract-agent/models /root/models; "
        f"ln -sfn /root/autodl-tmp/patent-extract-agent/data {workdir}/data; "
        f"ln -sfn /root/autodl-tmp/patent-extract-agent/artifacts {workdir}/artifacts; "
        f"ln -sfn /root/autodl-tmp/patent-extract-agent/logs {workdir}/logs; "
        "echo ok_data_disk; fi"
    )
    run(f"cd {workdir} && {bind_cmd}", timeout=30)

    # 3. 启动梯度+隔离统一实验（后台启动，立即返回；拉满 GPU）
    print("\n[2] 启动 gradient_isolate_unified...")
    setup = "source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate base 2>/dev/null || true"
    cmd = f"cd {workdir} && {setup} && (nohup bash scripts/autodl_run_gradient_isolate.sh >> logs/run_gradient_isolate.log 2>&1 &); sleep 1; echo started"
    run(cmd, timeout=60)
    print("  已启动 (后台)")

    # 3. 等待几秒后核查
    print("\n[3] 等待 5 秒后核查...")
    run("sleep 5")

    # 4. 核查：进程 + 日志
    print("\n[核查] 进程与日志:")
    procs = run("ps aux | grep -E 'train_|run_gradient_isolate_unified' | grep -v grep | head -5")
    print(procs if procs else "  (暂无训练进程，可能正在启动)")
    log_tail = run(f"tail -20 {workdir}/logs/run_gradient_isolate.log 2>/dev/null")
    if log_tail:
        print("\n[日志 tail]:")
        for line in log_tail.split("\n"):
            print(f"  {line[:100]}")
    else:
        print("  (日志暂无)")

    print("\n[ok] 实验已启动")
    print(f"  日志: tail -f {workdir}/logs/run_gradient_isolate.log")
    print("  核查: python _progress.py")
    print("=" * 60)
    c.close()


if __name__ == "__main__":
    main()
