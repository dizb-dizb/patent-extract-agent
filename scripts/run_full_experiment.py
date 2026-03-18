"""
一体化实验脚本：从数据准备到训练、能力测试，一键跑完全部实验。

实验矩阵：
  B1  BiLSTM  + CRF (BIO)     original  -- bilstm_crf
  B2  RoBERTa + CRF (BIO)     original  -- seq
  B3  BiLSTM  + Span (Proto)  original  -- fewshot + bilstm
  B4  RoBERTa + Span (Proto)  original  -- fewshot
  B5  BiLSTM  + Span (Proto)  augmented -- fewshot + bilstm + augmented
  Ours RoBERTa+ Span (Proto)  augmented -- fewshot + augmented

能力测试：
  战役一/二  Flat F1 / Nested F1  (已含在 fewshot 训练输出中)
  战役三     BWT                   run_continual.py
  战役四     Zero/One-shot OOD     run_ood_oneshot.py

用法：
  python scripts/run_full_experiment.py --fast           # B1-B4, original data, 跳过 evidence
  python scripts/run_full_experiment.py                  # B1-B5+Ours, full pipeline
  python scripts/run_full_experiment.py --datasets fewnerd --fast
  python scripts/run_full_experiment.py --epochs 3 --n_way 5 --k_shot 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ROOT / "data" / "benchmarks"
UNIFIED_DIR = ROOT / "data" / "dataset" / "unified"
SPLIT_DIR = ROOT / "data" / "dataset" / "split"
ARTIFACTS = ROOT / "artifacts" / "experiments"

DATASETS = ["fewnerd", "genia", "chemdner"]

# 收敛导向超参：保证模型充分训练
EPOCHS_B1 = {"fewnerd": 8, "genia": 12, "chemdner": 15}   # BiLSTM 随机初始化，小数据集需更多 epochs
EPOCHS_B2 = 5                                             # RoBERTa 预训练，5 epochs 足够
EPOCHS_PROTO = 8                                           # Proto-Span  episodic，需充分曝光
MAX_EPISODES_PROTO = {"fewnerd": 1000, "genia": 800, "chemdner": 800}
N_EVAL = 80
EPOCHS_BWT = 3
MAX_EPISODES_BWT = 300
# 数据集特定：chemdner 仅 1 类实体，需 n_way=1；GENIA 生物医学文本需 max_len=256 避免截断
N_WAY_PROTO = {"fewnerd": 5, "genia": 5, "chemdner": 1}
MAX_LEN_PROTO = 256


def run(cmd: list[str], desc: str) -> int:
    print(f"\n{'='*60}")
    print(f"[run] {desc}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*60}")
    ret = subprocess.run(cmd, cwd=str(ROOT))
    if ret.returncode != 0:
        print(f"[warn] {desc} exited with code {ret.returncode}")
    return ret.returncode


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def step_download(datasets: list[str]) -> None:
    run(
        [sys.executable, "scripts/download_benchmarks.py", ",".join(datasets)],
        "下载基准数据集",
    )


def step_convert(datasets: list[str]) -> None:
    for ds in datasets:
        script = f"scripts/convert_{ds}.py"
        if (ROOT / script).exists():
            run([sys.executable, script], f"转换 {ds}")
        else:
            print(f"[skip] {script} 不存在")


def step_build_dataset(datasets: list[str]) -> None:
    run(
        [sys.executable, "scripts/build_dataset.py", "--datasets", ",".join(datasets)],
        "构建合并数据集",
    )


def step_convert_with_evidence(limit: int = 0) -> None:
    cmd = [sys.executable, "scripts/convert_with_evidence.py"]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    run(cmd, "证据链检索 (convert_with_evidence)")


def step_split_evidence() -> None:
    run([sys.executable, "scripts/split_evidence_by_dataset.py"], "按数据集拆分 evidence")


def step_augment(datasets: list[str]) -> None:
    run(
        [
            sys.executable,
            "augment_from_evidence.py",
            "--input-dir", str(SPLIT_DIR),
            "--output-dir", str(SPLIT_DIR),
        ],
        "数据增强 (augment_from_evidence)",
    )


def run_baseline(
    dataset: str,
    mode: str,
    data_strategy: str = "original",
    encoder_type: str = "transformer",
    epochs: int = 2,
    n_way: int = 5,
    k_shot: int = 5,
    seed: int = 42,
    extra: list[str] | None = None,
    max_episodes: int = 0,
    n_eval: int = 0,
) -> dict | None:
    cmd = [
        sys.executable, "scripts/run_baseline.py",
        "--dataset", dataset,
        "--mode", mode,
        "--data-strategy", data_strategy,
        "--encoder-type", encoder_type,
        "--epochs", str(epochs),
        "--n_way", str(n_way),
        "--k_shot", str(k_shot),
        "--seed", str(seed),
    ]
    if max_episodes > 0:
        cmd.extend(["--max-episodes", str(max_episodes)])
    if n_eval > 0:
        cmd.extend(["--n-eval", str(n_eval)])
    if extra:
        cmd.extend(extra)
    desc = f"{dataset} {mode} encoder={encoder_type} data={data_strategy}"
    run(cmd, desc)

    # Extract encoder name from extra args to determine correct artifacts path
    encoder_name = ""
    if extra:
        for i, v in enumerate(extra):
            if v == "--encoder" and i + 1 < len(extra):
                encoder_name = extra[i + 1]
    roberta_sfx = "_roberta" if "roberta" in encoder_name.lower() else ""

    # Determine metrics path
    if mode == "bilstm_crf":
        metrics_path = ROOT / "artifacts" / "run_bilstm_crf" / dataset / "metrics.json"
    elif mode == "seq":
        metrics_path = ROOT / "artifacts" / f"run_seq_ner{roberta_sfx}" / dataset / "metrics.json"
    elif mode == "fewshot":
        aug_sfx = "_aug" if data_strategy == "augmented" else ""
        frozen_sfx = "_frozen" if extra and "--freeze-encoder" in extra else ""
        if encoder_type == "bilstm":
            base = f"run_proto_span_bilstm{aug_sfx}"
        else:
            base = f"run_proto_span{roberta_sfx}{aug_sfx}{frozen_sfx}"
        metrics_path = ROOT / "artifacts" / base / dataset / "metrics.json"
    else:
        aug_sfx = "_aug" if data_strategy == "augmented" else ""
        metrics_path = ROOT / "artifacts" / f"run_span_ner{aug_sfx}" / dataset / "metrics.json"

    m = load_metrics(metrics_path)
    if m:
        m["_run"] = {
            "dataset": dataset, "mode": mode,
            "encoder_type": encoder_type, "data_strategy": data_strategy,
            "encoder": encoder_name or encoder_type,
        }
    return m


def run_continual(encoder: str = "bert-base-cased", epochs: int = 1, max_episodes: int = 200) -> dict | None:
    cmd = [
        sys.executable, "scripts/run_continual.py",
        "--encoder", encoder,
        "--epochs_per_task", str(epochs),
        "--max_episodes", str(max_episodes),
    ]
    run(cmd, "战役三: BWT 持续学习")
    return load_metrics(ROOT / "artifacts" / "continual" / "metrics.json")


def run_ood(train_ds: str, test_ds: str, encoder: str = "bert-base-cased", k_shot: int = 1) -> dict | None:
    train_path = BENCHMARKS / train_ds / "train.jsonl"
    test_path = BENCHMARKS / test_ds / "test.jsonl"
    if not train_path.exists() or not test_path.exists():
        print(f"[skip] OOD: data not found ({train_path} / {test_path})")
        return None
    model_path = ROOT / "artifacts" / "run_proto_span" / train_ds / "model.pt"
    cmd = [
        sys.executable, "scripts/run_ood_oneshot.py",
        "--train_data", str(train_path),
        "--test_data", str(test_path),
        "--encoder", encoder,
        "--k_shot", str(k_shot),
    ]
    if model_path.exists():
        cmd.extend(["--model", str(model_path)])
    run(cmd, f"战役四: OOD {k_shot}-shot ({train_ds} -> {test_ds})")
    m = load_metrics(ROOT / "artifacts" / "ood_oneshot" / "metrics.json")
    if m:
        m["_run"] = {"train_ds": train_ds, "test_ds": test_ds, "k_shot": k_shot}
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="一体化实验脚本 B1-B5+Ours + 四大能力测试")
    ap.add_argument("--fast", action="store_true",
                    help="快速模式: 跳过 evidence 步骤, 只跑 B1-B4 (original data)")
    ap.add_argument("--datasets", type=str, default="fewnerd,genia,chemdner",
                    help="逗号分隔数据集, 如 fewnerd 或 fewnerd,genia")
    ap.add_argument("--epochs", type=int, default=0,
                    help="统一 epochs 覆盖 (0=使用收敛配置)")
    ap.add_argument("--n_way", type=int, default=5)
    ap.add_argument("--k_shot", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder", type=str, default="bert-base-cased",
                    help="Transformer encoder for B2/B4/Ours (BERT group)")
    ap.add_argument("--roberta-encoder", type=str, default="roberta-base",
                    help="RoBERTa encoder for B2r/B4r/Ours-r baselines")
    ap.add_argument("--skip-roberta", action="store_true",
                    help="跳过 RoBERTa 基线，只跑 BERT 组")
    ap.add_argument("--multi-gpu", action="store_true",
                    help="B2/B4/Ours 使用 DataParallel 多卡")
    ap.add_argument("--skip-data", action="store_true",
                    help="跳过数据准备步骤 (已有数据)")
    ap.add_argument("--reset", action="store_true",
                    help="清空 artifacts 后重新运行 (慎用)")
    ap.add_argument("--evidence-limit", type=int, default=0,
                    help="convert_with_evidence --limit N (0=不限制)")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    results: list[dict] = []
    t0 = time.time()

    # 收敛配置 vs 统一 epochs
    use_convergence = args.epochs == 0
    if use_convergence:
        print("[config] 使用收敛导向超参 (EPOCHS_B1/B2/PROTO, MAX_EPISODES)")

    # ── Reset: 清空 artifacts ───────────────────────────────────────────────
    if args.reset:
        import shutil
        arts = ROOT / "artifacts"
        if arts.exists():
            print(f"\n[reset] 清空 {arts}")
            for d in arts.iterdir():
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
                    print(f"  removed {d.name}")
        arts.mkdir(parents=True, exist_ok=True)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # ── Step 1-3: Data preparation ──────────────────────────────────────────
    if not args.skip_data:
        print("\n[phase] 数据准备")
        step_download(datasets)
        step_convert(datasets)
        step_build_dataset(datasets)

        if not args.fast:
            print("\n[phase] Evidence 增强 (可能耗时较长)")
            step_convert_with_evidence(limit=args.evidence_limit)
            step_split_evidence()
            step_augment(datasets)

    # ── Step 4: Baselines ────────────────────────────────────────────────────
    print("\n[phase] 基线实验")
    kw_base = dict(n_way=args.n_way, k_shot=args.k_shot, seed=args.seed)

    for ds in datasets:
        ep_b1 = EPOCHS_B1.get(ds, 8) if use_convergence else args.epochs
        ep_b2 = EPOCHS_B2 if use_convergence else args.epochs
        ep_proto = EPOCHS_PROTO if use_convergence else args.epochs
        max_ep = MAX_EPISODES_PROTO.get(ds, 800) if use_convergence else 0
        n_ev = N_EVAL if use_convergence else 0
        n_way_proto = N_WAY_PROTO.get(ds, 5)
        extra_max_len = ["--max-len", str(MAX_LEN_PROTO)]
        kw_proto = {**kw_base, "n_way": n_way_proto}  # fewshot 用数据集特定 n_way

        # B1: BiLSTM + CRF
        m = run_baseline(ds, "bilstm_crf", epochs=ep_b1, **kw_base)
        if m:
            m["baseline"] = "B1"
            results.append(m)

        # B2: RoBERTa + CRF (BIO)
        extra_b2 = ["--encoder", args.encoder]
        if args.multi_gpu:
            extra_b2.append("--multi-gpu")
        m = run_baseline(ds, "seq", encoder_type="transformer", epochs=ep_b2,
                         extra=extra_b2, **kw_base)
        if m:
            m["baseline"] = "B2"
            results.append(m)

        # B-Span: BERT + Span classifier, no proto (证明原型网络优势的对照基线)
        extra_bspan = ["--encoder", args.encoder]
        if args.multi_gpu:
            extra_bspan.append("--multi_gpu")
        m = run_baseline(ds, "supervised", encoder_type="transformer",
                         epochs=ep_b2, extra=extra_bspan, **kw_base)
        if m:
            m["baseline"] = "B-Span"
            results.append(m)

        # B3: BiLSTM + Span (Proto), original
        m = run_baseline(ds, "fewshot", encoder_type="bilstm", epochs=ep_proto,
                         max_episodes=max_ep, n_eval=n_ev,
                         extra=extra_max_len, **kw_proto)
        if m:
            m["baseline"] = "B3"
            results.append(m)

        # B4: BERT + Span (Proto), original
        extra_b4 = ["--encoder", args.encoder] + extra_max_len
        if args.multi_gpu:
            extra_b4.append("--multi-gpu")
        m = run_baseline(ds, "fewshot", encoder_type="transformer", epochs=ep_proto,
                         max_episodes=max_ep, n_eval=n_ev,
                         extra=extra_b4, **kw_proto)
        if m:
            m["baseline"] = "B4"
            results.append(m)

        # B4f: BERT-Proto + encoder 冻结（未微调模型+原型网络对照）
        extra_b4f = ["--encoder", args.encoder, "--freeze-encoder"] + extra_max_len
        if args.multi_gpu:
            extra_b4f.append("--multi-gpu")
        m = run_baseline(ds, "fewshot", encoder_type="transformer", epochs=ep_proto,
                         max_episodes=max_ep, n_eval=n_ev,
                         extra=extra_b4f, **kw_proto)
        if m:
            m["baseline"] = "B4f"
            results.append(m)

        if not args.fast:
            # B5: BiLSTM + Span (Proto), augmented
            m = run_baseline(ds, "fewshot", data_strategy="augmented",
                             encoder_type="bilstm", epochs=ep_proto,
                             max_episodes=max_ep, n_eval=n_ev,
                             extra=extra_max_len, **kw_proto)
            if m:
                m["baseline"] = "B5"
                results.append(m)

            # B-Span+Aug: BERT Span 无原型 + 增强数据（对照：证明原型网络在增强数据下的优势）
            extra_bspan_aug = ["--encoder", args.encoder]
            if args.multi_gpu:
                extra_bspan_aug.append("--multi-gpu")
            m = run_baseline(ds, "supervised", data_strategy="augmented",
                             encoder_type="transformer", epochs=ep_b2,
                             extra=extra_bspan_aug, **kw_base)
            if m:
                m["baseline"] = "B-Span+Aug"
                results.append(m)

            # Ours: BERT + Span (Proto), augmented
            extra_ours = ["--encoder", args.encoder] + extra_max_len
            if args.multi_gpu:
                extra_ours.append("--multi-gpu")
            m = run_baseline(ds, "fewshot", data_strategy="augmented",
                             encoder_type="transformer", epochs=ep_proto,
                             max_episodes=max_ep, n_eval=n_ev,
                             extra=extra_ours, **kw_proto)
            if m:
                m["baseline"] = "Ours"
                results.append(m)

        # ── RoBERTa group ────────────────────────────────────────────────────
        if not args.skip_roberta:
            # B2r: RoBERTa + CRF (BIO)
            extra_b2r = ["--encoder", args.roberta_encoder]
            if args.multi_gpu:
                extra_b2r.append("--multi-gpu")
            m = run_baseline(ds, "seq", encoder_type="transformer", epochs=ep_b2,
                             extra=extra_b2r, **kw_base)
            if m:
                m["baseline"] = "B2r"
                results.append(m)

            # B4r: RoBERTa + Span (Proto), original
            extra_b4r = ["--encoder", args.roberta_encoder] + extra_max_len
            if args.multi_gpu:
                extra_b4r.append("--multi-gpu")
            m = run_baseline(ds, "fewshot", encoder_type="transformer", epochs=ep_proto,
                             max_episodes=max_ep, n_eval=n_ev,
                             extra=extra_b4r, **kw_proto)
            if m:
                m["baseline"] = "B4r"
                results.append(m)

            # B4rf: RoBERTa-Proto + encoder 冻结（未微调模型+原型网络对照）
            extra_b4rf = ["--encoder", args.roberta_encoder, "--freeze-encoder"] + extra_max_len
            if args.multi_gpu:
                extra_b4rf.append("--multi-gpu")
            m = run_baseline(ds, "fewshot", encoder_type="transformer", epochs=ep_proto,
                             max_episodes=max_ep, n_eval=n_ev,
                             extra=extra_b4rf, **kw_proto)
            if m:
                m["baseline"] = "B4rf"
                results.append(m)

            if not args.fast:
                # Ours-r: RoBERTa + Span (Proto), augmented
                extra_oursr = ["--encoder", args.roberta_encoder] + extra_max_len
                if args.multi_gpu:
                    extra_oursr.append("--multi-gpu")
                m = run_baseline(ds, "fewshot", data_strategy="augmented",
                                 encoder_type="transformer", epochs=ep_proto,
                                 max_episodes=max_ep, n_eval=n_ev,
                                 extra=extra_oursr, **kw_proto)
                if m:
                    m["baseline"] = "Ours-r"
                    results.append(m)

    # ── Step 5: Capability tests ─────────────────────────────────────────────
    print("\n[phase] 能力测试")

    # 战役三: BWT
    ep_bwt = EPOCHS_BWT if use_convergence else max(1, args.epochs)
    bwt = run_continual(encoder=args.encoder, epochs=ep_bwt, max_episodes=MAX_EPISODES_BWT)
    if bwt:
        bwt["capability"] = "BWT"
        results.append(bwt)

    # 战役四: OOD 1-shot (fewnerd -> genia, fewnerd -> chemdner)
    for test_ds in [d for d in datasets if d != "fewnerd"]:
        if "fewnerd" in datasets:
            ood = run_ood("fewnerd", test_ds, encoder=args.encoder, k_shot=1)
            if ood:
                ood["capability"] = "OOD_1shot"
                results.append(ood)
    # Also 0-shot (k_shot=0 treated as k_shot=1 with random support, included for completeness)
    if "fewnerd" in datasets and "genia" in datasets:
        ood0 = run_ood("fewnerd", "genia", encoder=args.encoder, k_shot=5)
        if ood0:
            ood0["capability"] = "OOD_5shot"
            results.append(ood0)

    # ── Save summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    summary = {
        "runs": results,
        "datasets": datasets,
        "fast_mode": args.fast,
        "elapsed_seconds": round(elapsed, 1),
        "created_at": int(time.time()),
    }
    summary_path = ARTIFACTS / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Print table ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  实验汇总  (elapsed {elapsed/60:.1f} min)")
    print(f"{'='*70}")
    header = f"  {'Baseline':<8} {'Dataset':<12} {'Encoder':<12} {'Data':<12} {'F1':>6} {'FlatF1':>8}"
    print(header)
    print(f"  {'-'*68}")
    for r in results:
        run_info = r.get("_run") or {}
        baseline = r.get("baseline") or r.get("capability") or "?"
        ds_str = run_info.get("dataset") or run_info.get("test_ds") or "?"
        enc_str = run_info.get("encoder_type", r.get("encoder_type", "?"))[:10]
        data_str = run_info.get("data_strategy", "?")[:10]
        f1 = r.get("f1") or r.get("bwt") or 0.0
        flat_f1 = r.get("flat_f1") or r.get("f1") or 0.0
        print(f"  {baseline:<8} {ds_str:<12} {enc_str:<12} {data_str:<12} {f1:>6.4f} {flat_f1:>8.4f}")
    print(f"{'='*70}")
    print(f"[ok] 汇总已写入: {summary_path}")


if __name__ == "__main__":
    main()
