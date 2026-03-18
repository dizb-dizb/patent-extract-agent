
"""
Generate/update sections in REPORT_设计思路.md from experiment artifacts.

This is intentionally simple:
- No training logic here.
- It only aggregates numbers and renders markdown tables/snippets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPORT_PATH = Path(__file__).resolve().parent / "REPORT_设计思路.md"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ABLATIONS_DIR = ARTIFACTS_DIR / "ablations"


@dataclass
class RunMetrics:
    name: str
    f1: float | None = None
    precision: float | None = None
    recall: float | None = None
    latency_ms: float | None = None
    cost_tokens: int | None = None
    cost_cny: float | None = None


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _safe_int(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def load_metrics_from_json(path: Path) -> RunMetrics:
    """
    Accepts flexible metrics json.
    Recommended schema (minimal):
    {
      "name": "roberta+aug",
      "f1": 0.812,
      "precision": 0.84,
      "recall": 0.79,
      "latency_ms": 23.1,
      "cost_tokens": 123456,
      "cost_cny": 12.34
    }
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    name = str(data.get("name") or path.stem)
    return RunMetrics(
        name=name,
        f1=_safe_float(data.get("f1")),
        precision=_safe_float(data.get("precision")),
        recall=_safe_float(data.get("recall")),
        latency_ms=_safe_float(data.get("latency_ms")),
        cost_tokens=_safe_int(data.get("cost_tokens")),
        cost_cny=_safe_float(data.get("cost_cny")),
    )


def render_baseline_table(rows: list[RunMetrics]) -> str:
    def fmt(v: float | int | None) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}".rstrip("0").rstrip(".")
        return str(v)

    header = (
        "| 方法 | P | R | F1 | 延迟(ms) | Token成本 | ￥成本 |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.name,
                    fmt(r.precision),
                    fmt(r.recall),
                    fmt(r.f1),
                    fmt(r.latency_ms),
                    fmt(r.cost_tokens),
                    fmt(r.cost_cny),
                ]
            )
            + " |"
        )
    return header + "\n".join(lines) + "\n"


def update_report(report_text: str, table_md: str) -> str:
    """
    Insert or replace an auto-generated block in the report.
    """
    start = "<!-- AUTO_METRICS_START -->"
    end = "<!-- AUTO_METRICS_END -->"
    block = (
        f"{start}\n\n"
        "### 自动汇总：实验基线对比\n\n"
        "以下表格由 `report_generator.py` 根据 `artifacts/metrics/*.json` 自动生成。\n\n"
        f"{table_md}\n"
        f"{end}\n"
    )
    if start in report_text and end in report_text:
        pre = report_text.split(start)[0]
        post = report_text.split(end, 1)[1]
        return pre + block + post
    return report_text.rstrip() + "\n\n" + block


def load_ablations_summary() -> list[RunMetrics]:
    """从 artifacts/ablations/summary.json 加载消融实验结果。"""
    summary_path = ABLATIONS_DIR / "summary.json"
    if not summary_path.exists():
        return []
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = data.get("runs") or []
    rows: list[RunMetrics] = []
    for r in runs:
        run_info = r.get("_run") or {}
        name = f"{run_info.get('dataset','')}_{run_info.get('mode','')}"
        if run_info.get("k_shot"):
            name += f"_{run_info.get('n_way',5)}w{run_info.get('k_shot')}k"
        rows.append(RunMetrics(
            name=name,
            f1=_safe_float(r.get("f1")),
            precision=_safe_float(r.get("precision")),
            recall=_safe_float(r.get("recall")),
            latency_ms=_safe_float(r.get("latency_ms")),
            cost_tokens=_safe_int(r.get("cost_tokens")),
            cost_cny=_safe_float(r.get("cost_cny")),
        ))
    return rows


def main() -> None:
    metrics_dir = ARTIFACTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metric_files = sorted(metrics_dir.glob("*.json"))
    rows: list[RunMetrics] = []
    for p in metric_files:
        try:
            rows.append(load_metrics_from_json(p))
        except Exception:
            continue

    ab_rows = load_ablations_summary()
    if ab_rows:
        rows = ab_rows + rows

    table_md = render_baseline_table(rows) if rows else (
        "| 方法 | P | R | F1 | 延迟(ms) | Token成本 | ￥成本 |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        "| (暂无数据) | - | - | - | - | - | - |\n"
    )

    report_text = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""
    updated = update_report(report_text, table_md)
    REPORT_PATH.write_text(updated, encoding="utf-8")
    print(f"[ok] updated report: {REPORT_PATH}")
    print(f"[info] metrics files: {len(rows)} (dir={metrics_dir})")


if __name__ == "__main__":
    main()

