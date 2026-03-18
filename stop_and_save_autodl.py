"""
核查结果后终止 AutoDL 训练并保存 artifacts 到本地。

用法: python stop_and_save_autodl.py
       python stop_and_save_autodl.py --no-download   # 仅终止，不下载
"""
from __future__ import annotations

import os
import stat
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


def sftp_get_recursive(sftp, remote: str, local: Path) -> int:
    """递归下载 remote 目录到 local，返回下载文件数"""
    count = 0
    try:
        for item in sftp.listdir_attr(remote):
            rpath = f"{remote.rstrip('/')}/{item.filename}"
            lpath = local / item.filename
            if stat.S_ISDIR(item.st_mode):
                lpath.mkdir(parents=True, exist_ok=True)
                count += sftp_get_recursive(sftp, rpath, lpath)
            else:
                lpath.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(rpath, str(lpath))
                count += 1
    except FileNotFoundError:
        pass
    return count


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
    print("  核查结果 → 终止运行 → 保存")
    print("=" * 60)

    # 1. 终止训练进程
    print("\n[1] 终止训练进程...")
    run("pkill -f 'train_|run_full|run_remaining|run_gradient_isolate|autodl_run' 2>/dev/null || true")
    run("sleep 2")
    procs = run("ps aux | grep -E 'train_|run_gradient_isolate|run_remaining' | grep -v grep")
    if procs:
        print("  (部分进程可能仍在退出)")
    else:
        print("  已终止")

    # 2. 下载 artifacts
    download = "--no-download" not in sys.argv
    if download:
        print("\n[2] 下载 artifacts 到本地...")
        remote_art = f"{workdir}/artifacts"
        local_art = ROOT / "artifacts"
        local_art.mkdir(parents=True, exist_ok=True)

        sftp = c.open_sftp()
        try:
            n = sftp_get_recursive(sftp, remote_art, local_art)
            print(f"  已下载 {n} 个文件到 {local_art}")
        except Exception as e:
            print(f"  [warn] 下载异常: {e}")
        sftp.close()
    else:
        print("\n[2] 跳过下载 (--no-download)")

    # 3. 保存进度摘要
    print("\n[3] 保存进度摘要...")
    summary = run(f"find {workdir}/artifacts -name 'metrics.json' 2>/dev/null | wc -l")
    summary_file = ROOT / "artifacts" / "autodl_stop_summary.txt"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    content = f"""AutoDL 终止保存摘要
==================
远程: {user}@{host}:{port}
工作目录: {workdir}
metrics.json 数量: {summary.strip()}

核查命令: python _progress.py
"""
    summary_file.write_text(content, encoding="utf-8")
    print(f"  已保存: {summary_file}")

    c.close()
    print("\n[ok] 完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
