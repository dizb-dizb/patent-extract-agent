"""
Benchmark runner: run baselines and aggregate metrics for report_generator.

With --run: invokes real training (train_seq_ner, train_fewshot_proto_span).
Without --run: writes stub metrics for baselines not yet implemented.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ART_DIR = ROOT / "artifacts" / "metrics"
ART_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Metrics:
    name: str
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    latency_ms: float | None = None
    cost_tokens: int | None = None
    cost_cny: float | None = None
    dataset: str | None = None
    created_at: int | None = None


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def run_stub(name: str, dataset_path: Path) -> Path:
    data = load_jsonl(dataset_path)
    payload = {
        "name": name,
        "precision": None,
        "recall": None,
        "f1": None,
        "latency_ms": None,
        "cost_tokens": None,
        "cost_cny": None,
        "dataset": str(dataset_path),
        "num_samples": len(data),
        "created_at": int(time.time()),
    }
    out = ART_DIR / f"{name}_{int(time.time())}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def run_train_script(script: str, args: list[str], out_name: str) -> Path | None:
    import sys
    cmd = [sys.executable, script] + args
    try:
        subprocess.run(cmd, cwd=str(ROOT), check=True, capture_output=True, text=True, timeout=3600)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[warn] {script} failed: {e}")
        return None
    out_dir = ROOT / "artifacts" / ("run_seq_ner" if "seq_ner" in script else "run_proto_span")
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        dst = ART_DIR / f"{out_name}_{int(time.time())}.json"
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return dst
    return None


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="Run real training scripts (requires torch/transformers)")
    ap.add_argument("--epochs", type=int, default=1, help="Epochs when --run (quick smoke)")
    args = ap.parse_args()

    spans = ROOT / "train_spans_augmented.jsonl"
    if not spans.exists():
        spans = ROOT / "train_spans_enhanced.jsonl"
    if not spans.exists():
        spans = ROOT / "train_spans.jsonl"
    if not spans.exists():
        print("[fail] missing dataset")
        return

    outs = []
    if args.run:
        r = run_train_script(
            "train_seq_ner.py",
            ["--data", str(spans.name), "--epochs", str(args.epochs), "--output_dir", "artifacts/run_seq_ner"],
            "baseline_roberta_seq",
        )
        if r:
            outs.append(r)
        r = run_train_script(
            "train_fewshot_proto_span.py",
            ["--data", str(spans.name), "--epochs", str(args.epochs), "--output_dir", "artifacts/run_proto_span", "--max_episodes", "100"],
            "proposed_proto_span",
        )
        if r:
            outs.append(r)

    for name in ["baseline_bilstm_crf", "baseline_bert_base_zh", "baseline_llm_fewshot"]:
        if not any(name in str(p) for p in outs):
            outs.append(run_stub(name, spans))

    print("[ok] metrics:")
    for p in outs:
        print(" -", p)


if __name__ == "__main__":
    main()

