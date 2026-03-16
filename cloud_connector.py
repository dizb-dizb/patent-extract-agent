"""
Cloud training connector (generic skeleton).

This module is designed to be adapted to your chosen platform (e.g., your own server,
RunPod, HF Spaces, etc.). It keeps credentials in env vars and communicates via HTTP.

It supports:
- submit_job: upload dataset/config and get job_id
- poll_job: check status/metrics
- download_artifacts: pull back metrics & artifacts to ./artifacts/

No external dependencies (stdlib only).
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts" / "cloud"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


@dataclass
class CloudConfig:
    base_url: str
    api_key: str


def _cfg() -> CloudConfig:
    base_url = (os.environ.get("CLOUD_TRAIN_URL") or "").strip().rstrip("/")
    api_key = (os.environ.get("CLOUD_TRAIN_API_KEY") or "").strip()
    if not base_url:
        raise RuntimeError("Missing env CLOUD_TRAIN_URL")
    if not api_key:
        raise RuntimeError("Missing env CLOUD_TRAIN_API_KEY")
    return CloudConfig(base_url=base_url, api_key=api_key)


def _request_json(method: str, url: str, payload: dict | None = None, timeout_s: int = 30) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8", errors="ignore"))


def submit_job(dataset_path: Path, config: dict) -> str:
    """
    Submit a job. The default implementation sends a JSON payload containing dataset content.
    For large datasets you should switch to pre-signed upload or multipart.
    Expected server response: {"job_id": "..."}
    """
    cfg = _cfg()
    dataset_text = dataset_path.read_text(encoding="utf-8")
    payload = {
        "api_key": cfg.api_key,
        "dataset_name": dataset_path.name,
        "dataset_text": dataset_text,
        "config": config,
        "submitted_at": int(time.time()),
    }
    obj = _request_json("POST", f"{cfg.base_url}/submit", payload)
    job_id = str(obj.get("job_id") or "")
    if not job_id:
        raise RuntimeError(f"Invalid submit response: {obj}")
    return job_id


def poll_job(job_id: str) -> dict:
    """
    Expected server response: {"status": "running|done|failed", "metrics": {...}}
    """
    cfg = _cfg()
    obj = _request_json("GET", f"{cfg.base_url}/jobs/{job_id}?api_key={urllib.parse.quote(cfg.api_key)}", None)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Invalid poll response: {obj}")
    return obj


def download_artifacts(job_id: str) -> Path:
    """
    Expected server response: {"files": [{"name":"metrics.json","content":"...base64 or text..."}]}
    This is a placeholder; adapt to your server.
    """
    cfg = _cfg()
    obj = _request_json("GET", f"{cfg.base_url}/jobs/{job_id}/artifacts?api_key={urllib.parse.quote(cfg.api_key)}", None)
    out_dir = ARTIFACTS / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def main() -> None:
    # Example usage (requires a real server):
    dataset = ROOT / "train_spans_enhanced.jsonl"
    if not dataset.exists():
        dataset = ROOT / "train_spans.jsonl"
    cfg = {"model": "roberta_wwm_ext", "epochs": 3}
    try:
        job_id = submit_job(dataset, cfg)
    except Exception as e:
        print(f"[fail] submit_job: {e}")
        return
    print("[ok] job_id:", job_id)


if __name__ == "__main__":
    main()

