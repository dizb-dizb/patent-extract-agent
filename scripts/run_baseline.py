"""
统一 Baseline 入口：在指定数据集上运行 train_span_ner、train_seq_ner 或 train_fewshot_proto_span。

用法：
  python scripts/run_baseline.py --dataset fewnerd --mode supervised
  python scripts/run_baseline.py --dataset fewnerd --mode seq  # B2 BIO 基线
  python scripts/run_baseline.py --dataset chemdner --mode fewshot --n_way 5 --k_shot 5
  python scripts/run_baseline.py --dataset unified --mode fewshot  # 需先运行 build_dataset.py
  python scripts/run_baseline.py --dataset fewnerd --mode fewshot --data-strategy augmented  # Ours
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ROOT / "data" / "benchmarks"
UNIFIED_DIR = ROOT / "data" / "dataset" / "unified"
SPLIT_DIR = ROOT / "data" / "dataset" / "split"

DATASET_ENCODERS = {
    "fewnerd": "bert-base-cased",
    "genia": "bert-base-cased",
    "chemdner": "bert-base-cased",
    "unified": "bert-base-cased",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, choices=["fewnerd", "genia", "chemdner", "unified"])
    ap.add_argument("--mode", type=str,
                    choices=["supervised", "fewshot", "seq", "bilstm_crf"],
                    default="supervised",
                    help="supervised=span_ner; fewshot=proto_span; seq=BIO/BERT (B2); bilstm_crf=BiLSTM+CRF (B1)")
    ap.add_argument("--data-strategy", type=str, choices=["original", "augmented"], default="original",
                    help="original=基准数据; augmented=Agent 增强数据 (Ours)")
    ap.add_argument("--n_way", type=int, default=5)
    ap.add_argument("--k_shot", type=int, default=5)
    ap.add_argument("--encoder", type=str, default="")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-episodes", type=int, default=0, help="Fewshot only: episodes per epoch (0=use default 500)")
    ap.add_argument("--n-eval", type=int, default=0, help="Fewshot only: eval episodes (0=use default 50)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--multi-gpu", action="store_true", help="Use DataParallel for multi-GPU")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="Fewshot only: 冻结 encoder，仅训练 span_proj（未微调模型+原型网络）")
    ap.add_argument("--train-labels", type=str, default="", help="Meta-train labels (comma-separated)")
    ap.add_argument("--test-labels", type=str, default="", help="Meta-test labels (comma-separated)")
    ap.add_argument("--output-suffix", type=str, default="",
                    help="追加到 artifacts 目录名，如 _isolate 用于小样本类别隔离实验")
    ap.add_argument("--max-train-samples", type=int, default=0,
                    help="限制训练样本数 (0=不限制)，用于 10/100/1000 梯度实验")
    ap.add_argument("--encoder-type", type=str, default="transformer", choices=["transformer", "bilstm"],
                    help="transformer=BERT/RoBERTa; bilstm=randomly-init BiLSTM (B3/B5)")
    ap.add_argument("--bilstm-embed-dim", type=int, default=100)
    ap.add_argument("--bilstm-hidden", type=int, default=256)
    ap.add_argument("--bilstm-layers", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=0, help="Fewshot only: max token length (0=use 256)")
    ap.add_argument("--batch-size", type=int, default=0, help="Supervised only: batch size (0=use default 4)")
    ap.add_argument("--num-workers", type=int, default=0, help="Supervised only: DataLoader workers (4-8 拉满 GPU)")
    ap.add_argument("--batch-episodes", type=int, default=0, help="Fewshot only: batch_episodes (0=use default 4)")
    args = ap.parse_args()

    if args.dataset == "unified":
        data_dir = UNIFIED_DIR
        train_path = data_dir / "train.jsonl"
        val_path = data_dir / "val.jsonl"
        if not train_path.exists():
            print(f"[fail] {train_path} 不存在。请先运行: python scripts/build_dataset.py")
            sys.exit(1)
    else:
        if args.data_strategy == "augmented":
            data_dir = SPLIT_DIR
            train_path = data_dir / f"{args.dataset}_train_augmented.jsonl"
            val_path = data_dir / f"{args.dataset}_val_with_evidence.jsonl"
            if not val_path.exists():
                val_path = BENCHMARKS / args.dataset / "val.jsonl"
            if not train_path.exists():
                print(f"[fail] {train_path} 不存在。请先运行: python scripts/split_evidence_by_dataset.py && python augment_from_evidence.py --input-dir data/dataset/split --output-dir data/dataset/split")
                sys.exit(1)
        else:
            data_dir = BENCHMARKS / args.dataset
            train_path = data_dir / "train.jsonl"
            val_path = data_dir / "val.jsonl"
            if not train_path.exists():
                print(f"[fail] {train_path} not found. Run download + convert first.")
                sys.exit(1)

    encoder = args.encoder or DATASET_ENCODERS.get(args.dataset, "bert-base-cased")

    out_sfx = getattr(args, "output_suffix", "") or ""
    max_samples = getattr(args, "max_train_samples", 0) or 0
    if args.mode == "bilstm_crf":
        out_dir = ROOT / "artifacts" / f"run_bilstm_crf{out_sfx}" / args.dataset
        cmd = [
            sys.executable,
            str(ROOT / "train_bilstm_crf.py"),
            "--data", str(train_path),
            "--output_dir", str(out_dir),
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
        ]
        if val_path.exists():
            cmd.extend(["--val", str(val_path)])
        if max_samples > 0:
            cmd.extend(["--max_train_samples", str(max_samples)])
    elif args.mode == "supervised":
        aug_sfx = "_aug" if args.data_strategy == "augmented" else ""
        out_dir = ROOT / "artifacts" / f"run_span_ner{aug_sfx}{out_sfx}" / args.dataset
        cmd = [
            sys.executable,
            str(ROOT / "train_span_ner.py"),
            "--data", str(train_path),
            "--encoder", encoder,
            "--output_dir", str(out_dir),
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
        ]
        if val_path.exists():
            cmd.extend(["--val", str(val_path)])
        if args.multi_gpu:
            cmd.append("--multi_gpu")
        if max_samples > 0:
            cmd.extend(["--max_train_samples", str(max_samples)])
        if getattr(args, "batch_size", 0) > 0:
            cmd.extend(["--batch_size", str(args.batch_size)])
        if getattr(args, "num_workers", 0) > 0:
            cmd.extend(["--num_workers", str(args.num_workers)])
    elif args.mode == "seq":
        roberta_sfx = "_roberta" if "roberta" in encoder.lower() else ""
        out_dir = ROOT / "artifacts" / f"run_seq_ner{roberta_sfx}{out_sfx}" / args.dataset
        cmd = [
            sys.executable,
            str(ROOT / "train_seq_ner.py"),
            "--data", str(train_path),
            "--encoder", encoder,
            "--output_dir", str(out_dir),
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
        ]
        if args.multi_gpu:
            cmd.append("--multi_gpu")
        if max_samples > 0:
            cmd.extend(["--max_train_samples", str(max_samples)])
    else:
        # fewshot: proto_span, supports both transformer and bilstm encoder
        enc_type = getattr(args, "encoder_type", "transformer")
        if enc_type == "bilstm":
            suffix = "_bilstm"
        elif "roberta" in encoder.lower():
            suffix = "_roberta"
        else:
            suffix = ""
        if args.data_strategy == "augmented":
            suffix += "_aug"
        if getattr(args, "freeze_encoder", False):
            suffix += "_frozen"
        out_dir = ROOT / "artifacts" / f"run_proto_span{suffix}{out_sfx}" / args.dataset
        cmd = [
            sys.executable,
            str(ROOT / "train_fewshot_proto_span.py"),
            "--data", str(train_path),
            "--encoder", encoder,
            "--output_dir", str(out_dir),
            "--n_way", str(args.n_way),
            "--k_shot", str(args.k_shot),
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
            "--encoder_type", enc_type,
            "--bilstm_embed_dim", str(args.bilstm_embed_dim),
            "--bilstm_hidden", str(args.bilstm_hidden),
            "--bilstm_layers", str(args.bilstm_layers),
        ]
        if args.max_episodes > 0:
            cmd.extend(["--max_episodes", str(args.max_episodes)])
        if args.n_eval > 0:
            cmd.extend(["--n_eval", str(args.n_eval)])
        if val_path.exists():
            cmd.extend(["--val", str(val_path)])
        if args.multi_gpu:
            cmd.append("--multi_gpu")
        if args.max_len > 0:
            cmd.extend(["--max_len", str(args.max_len)])
        if getattr(args, "freeze_encoder", False):
            cmd.append("--freeze_encoder")
        if args.train_labels:
            cmd.extend(["--train_labels", args.train_labels])
        if args.test_labels:
            cmd.extend(["--test_labels", args.test_labels])
        if max_samples > 0:
            cmd.extend(["--max_train_samples", str(max_samples)])
        if getattr(args, "batch_episodes", 0) > 0:
            cmd.extend(["--batch_episodes", str(args.batch_episodes)])

    print(f"[run] {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=str(ROOT))
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
