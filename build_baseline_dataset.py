"""
基础数据集一键构建脚本。

从原始论文/专利文本到 baseline 可用的 span 数据集，支持：
1) 从 input/*.txt 标注生成 train_ready.jsonl（需配置 API）
2) 从已有 train_ready.jsonl 展开嵌套并导出 span 格式
3) 联网检索证据块并增强（可选）
4) 划分 train/val/test 用于 baseline 评测

用法：
  python build_baseline_dataset.py [--from-annotate] [--skip-evidence] [--split 0.8,0.1,0.1]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
TRAIN_READY = ROOT / "train_ready.jsonl"
TRAIN_SPANS = ROOT / "train_spans.jsonl"
TRAIN_ENHANCED = ROOT / "train_spans_enhanced.jsonl"
TRAIN_AUGMENTED = ROOT / "train_spans_augmented.jsonl"
OUT_TRAIN = ROOT / "data_baseline" / "train.jsonl"
OUT_VAL = ROOT / "data_baseline" / "val.jsonl"
OUT_TEST = ROOT / "data_baseline" / "test.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def step1_annotate():
    """从 input/*.txt 调用标注 API 生成 train_ready.jsonl（需 DEEPSEEK_API_KEY 等）"""
    from patent_annotator.main import main as annotate_main
    annotate_main()


def step2_expand_nested():
    """从 train_ready.jsonl 展开嵌套子实体，输出 train_spans.jsonl"""
    from patent_annotator.nested_expand import expand_entities_with_subentities
    from patent_annotator.span_format import sentence_record_to_span_sample

    records = load_jsonl(TRAIN_READY)
    if not records:
        print("[skip] 无 train_ready.jsonl，跳过嵌套展开")
        return False

    lexicon: dict[str, str] = {}
    for rec in records:
        for e in rec.get("entities") or []:
            if isinstance(e, dict):
                t = (e.get("text") or "").strip()
                if t:
                    lexicon[t] = (e.get("label") or "term").strip()

    out: list[dict] = []
    for rec in records:
        sent = rec.get("sentence") or ""
        ents = rec.get("entities") or []
        expanded = expand_entities_with_subentities(sent, ents, lexicon)
        rec["entities"] = expanded
        out.append(sentence_record_to_span_sample(sent, expanded))

    # 写回 train_ready（含嵌套），供 patent_agent_pipeline 使用
    with open(TRAIN_READY, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_jsonl(TRAIN_SPANS, out)
    print(f"[ok] 嵌套展开完成: {TRAIN_SPANS} ({len(out)} 条)")
    return True


def step3_evidence():
    """联网检索证据块，输出 train_spans_enhanced.jsonl"""
    from patent_agent_pipeline import main as pipeline_main
    pipeline_main()


def step4_augment():
    """基于 evidence.snippet 做数据增强，输出 train_spans_augmented.jsonl"""
    from augment_from_evidence import main as augment_main
    augment_main()


def step5_split(split_ratio: tuple[float, float, float], seed: int, source: Path):
    """划分 train/val/test"""
    records = load_jsonl(source)
    if not records:
        print(f"[fail] 无数据: {source}")
        return

    tr, va, te = split_ratio
    n = len(records)
    random.seed(seed)
    idx = list(range(n))
    random.shuffle(idx)
    n_train = int(n * tr)
    n_val = int(n * va)
    n_test = n - n_train - n_val

    train_rec = [records[idx[i]] for i in range(n_train)]
    val_rec = [records[idx[n_train + i]] for i in range(n_val)]
    test_rec = [records[idx[n_train + n_val + i]] for i in range(n_test)]

    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_TRAIN, train_rec)
    write_jsonl(OUT_VAL, val_rec)
    write_jsonl(OUT_TEST, test_rec)

    print(f"[ok] 划分完成: train={len(train_rec)} val={len(val_rec)} test={len(test_rec)}")
    print(f"     输出: {OUT_TRAIN.parent}")


def main() -> None:
    ap = argparse.ArgumentParser(description="基础数据集构建")
    ap.add_argument("--from-annotate", action="store_true", help="从 input 重新标注（需 API）")
    ap.add_argument("--skip-evidence", action="store_true", help="跳过联网证据检索与增强")
    ap.add_argument("--split", type=str, default="0.8,0.1,0.1", help="train,val,test 比例")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    split_ratio = tuple(float(x) for x in args.split.split(","))
    if len(split_ratio) != 3:
        raise ValueError("--split 需为 3 个小数，如 0.8,0.1,0.1")

    if args.from_annotate:
        print("[1/5] 从 input 标注...")
        step1_annotate()
    else:
        if not TRAIN_READY.exists():
            print("[warn] 无 train_ready.jsonl，请先运行: python -m patent_annotator.main")
            print("       或从已有 train_ready.jsonl 开始（若已存在）")

    print("[2/5] 嵌套展开...")
    step2_expand_nested()

    if not args.skip_evidence:
        print("[3/5] 联网证据检索...")
        step3_evidence()
        print("[4/5] 证据增强...")
        step4_augment()
        source = TRAIN_AUGMENTED if TRAIN_AUGMENTED.exists() else TRAIN_ENHANCED
    else:
        source = TRAIN_SPANS

    print("[5/5] 划分 train/val/test...")
    step5_split(split_ratio, args.seed, source)

    print("\n[done] 基础数据集已就绪，可用于 baseline 训练：")
    print(f"  - 训练: {OUT_TRAIN}")
    print(f"  - 验证: {OUT_VAL}")
    print(f"  - 测试: {OUT_TEST}")


if __name__ == "__main__":
    main()
