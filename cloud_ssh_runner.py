"""
SSH-based cloud runner for GPU platforms (RunPod / AutoDL / etc.).

Workflow:
1) Create a job folder locally with dataset + config + runner script
2) scp upload to remote workdir
3) ssh run remote command (python train_remote.py ...)
4) scp download artifacts to ./artifacts/cloud/<job_id>/
5) (optional) copy metrics.json into ./artifacts/metrics/ for report aggregation

Requirements:
- Windows OpenSSH available (ssh/scp in PATH)
- Remote machine has python and required ML deps (you manage your env/venv/conda)

Env vars:
  CLOUD_SSH_HOST           e.g. "xxx.autodl.com"
  CLOUD_SSH_PORT           e.g. "22" (optional)
  CLOUD_SSH_USER           e.g. "root" / "ubuntu"
  CLOUD_SSH_KEY_PATH       e.g. "C:\\Users\\me\\.ssh\\id_rsa" (optional)
  CLOUD_REMOTE_WORKDIR     e.g. "/root/patent-agent-jobs"
  CLOUD_REMOTE_PYTHON      e.g. "python3" (default: python3)
  CLOUD_REMOTE_SETUP_CMD   e.g. "source /opt/conda/bin/activate myenv" (optional)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ART_CLOUD = ROOT / "artifacts" / "cloud"
ART_METRICS = ROOT / "artifacts" / "metrics"
ART_CLOUD.mkdir(parents=True, exist_ok=True)
ART_METRICS.mkdir(parents=True, exist_ok=True)


@dataclass
class SshConfig:
    host: str
    port: int
    user: str
    key_path: str | None
    remote_workdir: str
    remote_python: str
    remote_setup_cmd: str | None


def load_ssh_config() -> SshConfig:
    host = (os.environ.get("CLOUD_SSH_HOST") or "").strip()
    user = (os.environ.get("CLOUD_SSH_USER") or "").strip()
    if not host:
        raise RuntimeError("Missing env CLOUD_SSH_HOST")
    if not user:
        raise RuntimeError("Missing env CLOUD_SSH_USER")
    port = int((os.environ.get("CLOUD_SSH_PORT") or "22").strip() or "22")
    key_path = (os.environ.get("CLOUD_SSH_KEY_PATH") or "").strip() or None
    remote_workdir = (os.environ.get("CLOUD_REMOTE_WORKDIR") or "").strip() or "/root/patent-agent-jobs"
    remote_python = (os.environ.get("CLOUD_REMOTE_PYTHON") or "").strip() or "python3"
    remote_setup_cmd = (os.environ.get("CLOUD_REMOTE_SETUP_CMD") or "").strip() or None
    return SshConfig(
        host=host,
        port=port,
        user=user,
        key_path=key_path,
        remote_workdir=remote_workdir,
        remote_python=remote_python,
        remote_setup_cmd=remote_setup_cmd,
    )


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, shell=False)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}")


def _ssh_base_args(cfg: SshConfig) -> list[str]:
    args = ["ssh", "-p", str(cfg.port)]
    if cfg.key_path:
        args += ["-i", cfg.key_path]
    # avoid interactive prompts in automation
    args += ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL"]
    return args


def _scp_base_args(cfg: SshConfig) -> list[str]:
    args = ["scp", "-P", str(cfg.port)]
    if cfg.key_path:
        args += ["-i", cfg.key_path]
    args += ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL"]
    return args


def make_job_bundle(dataset_path: Path, job_id: str, config: dict, mode: str = "proto_span") -> Path:
    """
    Create local job directory with dataset, config, scripts.
    mode: proto_span | span_ner | seq_ner
    """
    job_dir = ART_CLOUD / "_bundles" / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(dataset_path, job_dir / dataset_path.name)
    (job_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # Copy training scripts and fewshot package
    shutil.copy2(ROOT / "train_span_ner.py", job_dir / "train_span_ner.py")
    shutil.copy2(ROOT / "train_fewshot_proto_span.py", job_dir / "train_fewshot_proto_span.py")
    shutil.copy2(ROOT / "train_seq_ner.py", job_dir / "train_seq_ner.py")
    shutil.copy2(ROOT / "dataset_version.py", job_dir / "dataset_version.py")
    (job_dir / "fewshot").mkdir(exist_ok=True)
    for f in ["__init__.py", "episode_dataset.py", "model.py"]:
        src = ROOT / "fewshot" / f
        if src.exists():
            shutil.copy2(src, job_dir / "fewshot" / f)
    (job_dir / "configs").mkdir(exist_ok=True)
    if (ROOT / "configs" / "proto_default.json").exists():
        shutil.copy2(ROOT / "configs" / "proto_default.json", job_dir / "configs" / "proto_default.json")

    py_cmd = config.get("python", "python3")
    dataset = config.get("dataset_name") or dataset_path.name
    encoder = config.get("encoder") or "hfl/chinese-roberta-wwm-ext"
    epochs = config.get("epochs", 2)
    max_len = config.get("max_len", 256)
    out_dir = config.get("output_dir") or "run_out"

    if mode == "proto_span":
        run_cmd = f"{py_cmd} train_fewshot_proto_span.py --data {dataset} --encoder {encoder} --epochs {epochs} --max_len {max_len} --output_dir {out_dir} --n_way 5 --k_shot 5"
    elif mode == "seq_ner":
        run_cmd = f"{py_cmd} train_seq_ner.py --data {dataset} --encoder {encoder} --epochs {epochs} --max_len {max_len} --output_dir {out_dir}"
    else:
        run_cmd = f"{py_cmd} train_span_ner.py --data {dataset} --encoder {encoder} --epochs {epochs} --max_len {max_len} --output_dir {out_dir}"

    train_py = job_dir / "train_remote.py"
    train_py.write_text(
        (
            "import json, subprocess\n"
            "from pathlib import Path\n"
            "cfg = json.loads(Path('config.json').read_text(encoding='utf-8'))\n"
            f"run_cmd = {repr(run_cmd)}\n"
            "print('[info] running:', run_cmd)\n"
            "subprocess.check_call(run_cmd, shell=True)\n"
            "m = json.loads(Path(cfg.get('output_dir','run_out'), 'metrics.json').read_text(encoding='utf-8'))\n"
            "Path('metrics.json').write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding='utf-8')\n"
            "print('[ok] metrics.json ready')\n"
        ),
        encoding="utf-8",
    )
    return job_dir


def upload_job(cfg: SshConfig, job_dir: Path, remote_job_dir: str) -> None:
    # ensure remote base dir exists
    remote = f"{cfg.user}@{cfg.host}"
    mk = f"mkdir -p {shlex_quote(cfg.remote_workdir)} {shlex_quote(remote_job_dir)}"
    ssh_cmd = _ssh_base_args(cfg) + [remote, mk]
    _run(ssh_cmd)

    # upload directory contents (recursive)
    # Use trailing '/.' to copy directory contents (works across platforms)
    scp_cmd = _scp_base_args(cfg) + ["-r", str(job_dir) + "/.", f"{remote}:{remote_job_dir}/"]
    _run(scp_cmd)


def run_remote(cfg: SshConfig, remote_job_dir: str) -> None:
    remote = f"{cfg.user}@{cfg.host}"
    setup = (cfg.remote_setup_cmd + " && ") if cfg.remote_setup_cmd else ""
    # Save stdout/stderr to a log for debugging
    cmd = f"cd {shlex_quote(remote_job_dir)} && {setup}{cfg.remote_python} train_remote.py > run.log 2>&1"
    ssh_cmd = _ssh_base_args(cfg) + [remote, cmd]
    _run(ssh_cmd)


def download_artifacts(cfg: SshConfig, remote_job_dir: str, local_out: Path) -> None:
    remote = f"{cfg.user}@{cfg.host}"
    local_out.mkdir(parents=True, exist_ok=True)
    scp_cmd = _scp_base_args(cfg) + ["-r", f"{remote}:{remote_job_dir}/", str(local_out)]
    _run(scp_cmd)


def shlex_quote(s: str) -> str:
    # remote is assumed to be bash-like; basic safe quoting
    return "'" + s.replace("'", "'\"'\"'") + "'"


def main(mode: str = "proto_span") -> None:
    cfg = load_ssh_config()
    dataset = ROOT / "train_spans_augmented.jsonl"
    if not dataset.exists():
        dataset = ROOT / "train_spans_enhanced.jsonl"
    if not dataset.exists():
        dataset = ROOT / "train_spans.jsonl"
    if not dataset.exists():
        raise RuntimeError("Missing dataset: train_spans_enhanced.jsonl or train_spans.jsonl")

    job_id = f"sshjob_{int(time.time())}"
    remote_job_dir = f"{cfg.remote_workdir.rstrip('/')}/{job_id}"
    config = {
        "run_name": mode,
        "dataset_name": dataset.name,
        "python": cfg.remote_python,
        "encoder": "hfl/chinese-roberta-wwm-ext",
        "epochs": 2,
        "max_len": 256,
        "output_dir": "run_out",
    }

    job_dir = make_job_bundle(dataset, job_id, config, mode=mode)
    print("[info] job_dir:", job_dir)
    print("[info] mode:", mode)

    upload_job(cfg, job_dir, remote_job_dir)
    print("[ok] uploaded")
    run_remote(cfg, remote_job_dir)
    print("[ok] remote run done")

    local_out = ART_CLOUD / job_id
    download_artifacts(cfg, remote_job_dir, local_out)
    print("[ok] downloaded to:", local_out)

    metrics_src = local_out / "metrics.json"
    if not metrics_src.exists():
        metrics_src = local_out / "run_out" / "metrics.json"
    if metrics_src.exists():
        metrics_dst = ART_METRICS / f"cloud_{job_id}.json"
        metrics_dst.write_text(metrics_src.read_text(encoding="utf-8"), encoding="utf-8")
        print("[ok] copied metrics to:", metrics_dst)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="proto_span", choices=["proto_span", "seq_ner", "span_ner"], help="Training mode")
    a = ap.parse_args()
    main(mode=a.mode)

