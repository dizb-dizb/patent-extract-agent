"""
上传更新代码到 AutoDL，终止旧进程，启动补充实验。

用法:
  python upload_and_run_autodl.py                    # 使用默认/硬编码连接
  python upload_and_run_autodl.py --env .env        # 从 .env 读取 CLOUD_SSH_*

连接配置（.env 或环境变量）:
  CLOUD_SSH_HOST=connect.xx.seetacloud.com
  CLOUD_SSH_PORT=38815
  CLOUD_SSH_USER=root
  CLOUD_SSH_PASSWORD=xxx
  CLOUD_REMOTE_WORKDIR=/root/patent-extract-agent
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 需上传的文件（保持 AutoDL 与本地同步）
UPLOADS = [
    "scripts/run_remaining_experiments.py",
    "scripts/run_gradient_isolate_unified.py",
    "scripts/run_baseline.py",
    "scripts/release_artifact_models.py",
    "scripts/kill_training_processes.py",
    "scripts/run_supplementary.sh",
    "scripts/autodl_run_supplementary.sh",
    "scripts/autodl_run_gradient_isolate.sh",
    "scripts/autodl_run_all_until_done.sh",
    "scripts/run_fix_main_experiments.py",
    "scripts/autodl_run_fix.sh",
    "train_fewshot_proto_span.py",
    "train_span_ner.py",
    "train_seq_ner.py",
    "train_bilstm_crf.py",
    "fewshot/model.py",
    "fewshot/episode_dataset.py",
]


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"')


def main() -> None:
    ap = argparse.ArgumentParser(description="上传到 AutoDL 并运行补充实验")
    ap.add_argument("--env", type=str, default=".env", help=".env 文件路径")
    ap.add_argument("--host", type=str, default="", help="覆盖 SSH 主机")
    ap.add_argument("--port", type=int, default=0, help="覆盖 SSH 端口")
    ap.add_argument("--user", type=str, default="root")
    ap.add_argument("--password", type=str, default="")
    ap.add_argument("--workdir", type=str, default="/root/patent-extract-agent")
    ap.add_argument("--no-run", action="store_true", help="仅上传，不运行")
    ap.add_argument("--foreground", action="store_true", help="前台运行（不 nohup）")
    ap.add_argument("--mode", type=str, default="all",
                    choices=["remaining", "gradient", "isolate", "gradient_isolate", "all"],
                    help="remaining=剩余实验; gradient_isolate=梯度+隔离; all=剩余→梯度+隔离 持续运行直到完成")
    args = ap.parse_args()

    load_env(ROOT / args.env)

    host = args.host or os.environ.get("CLOUD_SSH_HOST", "connect.cqa1.seetacloud.com")
    port = args.port or int(os.environ.get("CLOUD_SSH_PORT", "38815"))
    user = args.user or os.environ.get("CLOUD_SSH_USER", "root")
    password = args.password or os.environ.get("CLOUD_SSH_PASSWORD", "CQtlwjJT2xIF")
    workdir = args.workdir or os.environ.get("CLOUD_REMOTE_WORKDIR", "/root/patent-extract-agent")

    try:
        import paramiko
    except ImportError:
        print("[fail] 需要 paramiko: pip install paramiko  或  pip install -r requirements-upload.txt")
        sys.exit(1)

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        c.connect(host, port, user, password, timeout=15)
    except Exception as e:
        print(f"[fail] 连接失败 {host}:{port} - {e}")
        sys.exit(1)

    def run(cmd: str, timeout: int = 60) -> str:
        _, out, _ = c.exec_command(cmd, timeout=timeout)
        return out.read().decode(errors="replace").strip()

    print("=" * 60)
    print("  上传到 AutoDL 并运行补充实验")
    print(f"  {user}@{host}:{port}  workdir={workdir}")
    print("=" * 60)

    # 1. 上传
    sftp = c.open_sftp()
    run(f"mkdir -p {workdir}/scripts {workdir}/logs {workdir}/fewshot")
    for rel in UPLOADS:
        local = ROOT / rel
        if not local.exists():
            print(f"  [skip] {rel} 不存在")
            continue
        remote = f"{workdir}/{rel}"
        sftp.put(str(local), remote)
        print(f"  已上传: {rel}")
    sftp.close()

    if args.no_run:
        print("\n[ok] 仅上传完成 (--no-run)")
        c.close()
        return

    # 2. 终止旧进程
    print("\n[step] 终止旧训练进程...")
    run("pkill -f 'train_bilstm_crf|train_seq_ner|train_fewshot_proto_span|train_span_ner|run_full|run_remaining|run_gradient_isolate' 2>/dev/null || true")
    run("sleep 3")
    print("  已清理")

    # 2.5 先释放已有模型参数，防止磁盘占满
    print("\n[step] 释放已有模型权重（保留 metrics.json）...")
    out = run(f"cd {workdir} && python scripts/release_artifact_models.py --artifacts-dir {workdir}/artifacts", timeout=120)
    if out:
        for line in out.strip().split("\n"):
            if line.strip():
                print(f"  {line.strip()}")
    print("  已执行释放")

    # 2.6 若有数据盘，将 data/artifacts/logs/models 挂到数据盘（含模型参数）
    print("\n[step] 检查数据盘并挂载 data/artifacts/logs/models...")
    bind_cmd = (
        f"if [ -d /root/autodl-tmp ] && [ ! -L {workdir}/data ]; then "
        f"mkdir -p /root/autodl-tmp/patent-extract-agent/data /root/autodl-tmp/patent-extract-agent/artifacts /root/autodl-tmp/patent-extract-agent/logs /root/autodl-tmp/patent-extract-agent/models; "
        f"[ -d {workdir}/data ] && [ ! -L {workdir}/data ] && mv {workdir}/data /root/autodl-tmp/patent-extract-agent/; "
        f"[ -d {workdir}/artifacts ] && [ ! -L {workdir}/artifacts ] && mv {workdir}/artifacts /root/autodl-tmp/patent-extract-agent/; "
        f"[ -d {workdir}/logs ] && [ ! -L {workdir}/logs ] && mv {workdir}/logs /root/autodl-tmp/patent-extract-agent/; "
        f"[ -d /root/models ] && [ ! -L /root/models ] && mv /root/models /root/autodl-tmp/patent-extract-agent/; ln -sfn /root/autodl-tmp/patent-extract-agent/models /root/models; "
        f"ln -sfn /root/autodl-tmp/patent-extract-agent/data {workdir}/data; "
        f"ln -sfn /root/autodl-tmp/patent-extract-agent/artifacts {workdir}/artifacts; "
        f"ln -sfn /root/autodl-tmp/patent-extract-agent/logs {workdir}/logs; "
        "echo ok_data_disk; fi"
    )
    out_bind = run(f"cd {workdir} && {bind_cmd}", timeout=30)
    if "ok_data_disk" in (out_bind or ""):
        print("  已挂载到数据盘 /root/autodl-tmp（参数会按流程释放）")
    else:
        print("  未检测到数据盘或已挂载，使用当前目录")

    # 3. 运行实验
    setup = "source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh; conda activate base 2>/dev/null || conda activate patent-agent 2>/dev/null || true"
    if args.mode == "all":
        script_cmd = "bash scripts/autodl_run_all_until_done.sh"
        log_file = "logs/run_all_until_done.log"
    elif args.mode == "remaining":
        script_cmd = "python scripts/run_remaining_experiments.py --multi-gpu"
        log_file = "logs/run_remaining.log"
    elif args.mode in ("gradient", "isolate"):
        script_cmd = "bash scripts/autodl_run_gradient_isolate.sh"
        log_file = "logs/run_gradient_isolate.log"
    else:  # gradient_isolate
        script_cmd = "bash scripts/autodl_run_gradient_isolate.sh"
        log_file = "logs/run_gradient_isolate.log"

    print(f"\n[step] 启动实验 (mode={args.mode})...")
    if args.foreground:
        cmd = f"cd {workdir} && {setup} && {script_cmd}"
        print(run(cmd, timeout=3600 * 24))
    else:
        cmd = f"cd {workdir} && {setup} && (nohup {script_cmd} >> {log_file} 2>&1 &); sleep 1; echo started"
        run(cmd, timeout=60)
        print(f"\n[已启动] 实验在后台运行")
        print(f"  日志: tail -f {workdir}/{log_file}")
        print("  监控: ssh 登录后执行 python _progress.py")
    print("=" * 60)
    c.close()


if __name__ == "__main__":
    main()
