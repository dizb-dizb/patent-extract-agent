#!/usr/bin/env python3
"""
实验状态检查 - 本地/AutoDL 通用
用法: python scripts/check_experiment_status.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 实验矩阵: (显示名, 日志名, 输出目录, 期望 epoch)
MATRIX = [
    ("B1", "BiLSTM-CRF", [
        ("fewnerd", "artifacts/run_bilstm_crf/fewnerd", 3),
        ("genia", "artifacts/run_bilstm_crf/genia", 5),
        ("chemdner", "artifacts/run_bilstm_crf/chemdner", 5),
    ]),
    ("B2", "RoBERTa-CRF", [
        ("fewnerd", "artifacts/run_seq_ner/fewnerd", 3),
        ("genia", "artifacts/run_seq_ner/genia", 5),
        ("chemdner", "artifacts/run_seq_ner/chemdner", 5),
    ]),
    ("B3", "BiLSTM-Proto", [
        ("fewnerd", "artifacts/run_proto_span_bilstm/fewnerd", 3),
        ("genia", "artifacts/run_proto_span_bilstm/genia", 5),
        ("chemdner", "artifacts/run_proto_span_bilstm/chemdner", 5),
    ]),
    ("B4", "RoBERTa-Proto", [
        ("fewnerd", "artifacts/run_proto_span/fewnerd", 3),
        ("genia", "artifacts/run_proto_span/genia", 5),
        ("chemdner", "artifacts/run_proto_span/chemdner", 5),
    ]),
    ("B5", "BiLSTM-Proto+Aug", [
        ("fewnerd", "artifacts/run_proto_span_bilstm_aug/fewnerd", 3),
        ("genia", "artifacts/run_proto_span_bilstm_aug/genia", 3),
        ("chemdner", "artifacts/run_proto_span_bilstm_aug/chemdner", 3),
    ]),
    ("Ours", "RoBERTa-Proto+Aug+SCL", [
        ("fewnerd", "artifacts/run_proto_span_aug/fewnerd", 3),
        ("genia", "artifacts/run_proto_span_aug/genia", 3),
        ("chemdner", "artifacts/run_proto_span_aug/chemdner", 3),
    ]),
]


def load_metrics(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    print("=" * 70)
    print("  专利术语小样本 NER - 实验状态检查")
    print("=" * 70)
    print(f"  项目根目录: {ROOT}")
    print()

    # 汇总表
    rows = []
    for bid, bname, datasets in MATRIX:
        for ds, out_rel, ep_max in datasets:
            out_path = ROOT / out_rel
            m_path = out_path / "metrics.json"
            m = load_metrics(m_path)
            if m:
                ep = m.get("epoch", "?")
                f1 = m.get("f1")
                ff = m.get("flat_f1", f1)
                f1_str = f"{f1:.4f}" if f1 is not None else "—"
                ff_str = f"{ff:.4f}" if ff is not None else "—"
                status = "✓ DONE"
            else:
                ep = "?"
                f1_str = ff_str = "—"
                status = "○ 未完成"
            rows.append((bid, ds, bname, status, f"{ep}/{ep_max}", f1_str, ff_str))

    # 打印矩阵
    print("▶ 实验矩阵")
    print("  " + "-" * 66)
    print(f"  {'Baseline':<8} {'fewnerd':<12} {'genia':<12} {'chemdner':<12}")
    print("  " + "-" * 66)
    by_baseline: dict[str, dict[str, str]] = {}
    for bid, ds, _, status, ep, f1, ff in rows:
        if bid not in by_baseline:
            by_baseline[bid] = {"fewnerd": "—", "genia": "—", "chemdner": "—"}
        by_baseline[bid][ds] = f1 if status == "✓ DONE" else "—"
    for bid, bname, datasets in MATRIX:
        r = by_baseline.get(bid, {})
        fn = r.get("fewnerd", "—")
        gn = r.get("genia", "—")
        cn = r.get("chemdner", "—")
        label = f"{bid}({bname[:14]})"[:20]
        print(f"  {label:<20} {fn:<12} {gn:<12} {cn:<12}")
    print()

    # 详细列表
    print("▶ 详细状态")
    for bid, ds, bname, status, ep, f1, ff in rows:
        print(f"  {bid} {ds:<10}  {status:<10}  ep:{ep:<6}  F1:{f1:<8} FlatF1:{ff}")
    print()

    # 战役三/四
    bwt_path = ROOT / "artifacts/continual/metrics.json"
    ood_path = ROOT / "artifacts/ood_oneshot/metrics.json"
    print("▶ 战役三 BWT 持续学习")
    if bwt_path.exists():
        m = load_metrics(bwt_path)
        if m:
            bwt = m.get("bwt")
            print(f"  BWT = {bwt:+.4f}" if bwt is not None else "  (无 bwt)")
        else:
            print("  (解析失败)")
    else:
        print("  (未执行)")
    print()
    print("▶ 战役四 Zero/One-shot OOD")
    if ood_path.exists():
        m = load_metrics(ood_path)
        if m:
            p, r, f = m.get("precision"), m.get("recall"), m.get("f1")
            print(f"  P={p:.4f} R={r:.4f} F1={f:.4f}" if f is not None else "  (无 f1)")
        else:
            print("  (解析失败)")
    else:
        print("  (未执行)")
    print("=" * 70)


if __name__ == "__main__":
    main()
