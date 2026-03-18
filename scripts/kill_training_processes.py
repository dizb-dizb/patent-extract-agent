"""
跨平台终止训练进程：train_bilstm_crf, train_seq_ner, train_fewshot_proto_span, train_span_ner
"""
from __future__ import annotations

import platform
import subprocess
import sys

PATTERNS = [
    "train_bilstm_crf",
    "train_seq_ner",
    "train_fewshot_proto_span",
    "train_span_ner",
    "run_full_experiment",
    "run_remaining",
    "run_gradient_isolate",
]


def main() -> None:
    system = platform.system()
    if system == "Windows":
        # 查找并终止包含训练脚本的 python 进程
        try:
            out = subprocess.run(
                ["wmic", "process", "where", "name='python.exe'", "get", "processid,commandline"],
                capture_output=True, text=True, timeout=10
            )
            if out.returncode == 0 and out.stdout:
                for line in out.stdout.strip().splitlines()[1:]:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[-1])
                            cmdline = " ".join(parts[:-1]) if len(parts) > 2 else ""
                            if any(p in cmdline for p in PATTERNS):
                                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
                                print(f"  [kill] PID {pid}")
                        except (ValueError, IndexError):
                            pass
        except Exception:
            # 回退：终止所有 python
            subprocess.run(["taskkill", "/F", "/IM", "python.exe"], capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "python3.exe"], capture_output=True)
            print("  [kill] 已终止所有 python 进程")
    else:
        # Linux / macOS
        for p in PATTERNS:
            subprocess.run(["pkill", "-f", p], capture_output=True)
        print("  [kill] 已发送 pkill 信号")
    print("[ok] 训练进程已清理")


if __name__ == "__main__":
    main()
