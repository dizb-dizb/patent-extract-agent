"""
持续学习与 BWT（后向转移率）评测。

流水线任务：先学 chemdner，再学 genia（混入 chemdner 温习样本）。
BWT = 学完 task2 后在 task1 上的表现 - 学完 task1 后在 task1 上的表现。
负值表示灾难性遗忘。

用法：
  python scripts/run_continual.py
  python scripts/run_continual.py --rehearsal_ratio 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer

from dataset_version import dataset_version
from fewshot.episode_dataset import EpisodicSpanDataset, load_jsonl
from fewshot.model import PrototypicalSpanNER, char_span_to_token_span
from train_fewshot_proto_span import process_episode

BENCHMARKS = ROOT / "data" / "benchmarks"
SPLIT_DIR = ROOT / "data" / "dataset" / "split"
ARTIFACTS = ROOT / "artifacts" / "continual"
SPECIAL_OFFSETS = {(0, 0)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _tokenize(tokenizer, text: str, max_len: int):
    return tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_len,
        padding=False,
    )


def eval_dataset(
    model: PrototypicalSpanNER,
    data_path: Path,
    tokenizer,
    max_len: int,
    device: torch.device,
    n_way: int,
    k_shot: int,
    n_eval: int,
) -> float:
    """Evaluate model on dataset, return F1."""
    samples = load_jsonl(data_path)
    if not samples:
        return 0.0
    ds = EpisodicSpanDataset(
        samples,
        n_way=n_way,
        k_shot=k_shot,
        query_per_class=5,
        neg_ratio=0.3,
        max_span_width=12,
        max_episodes=n_eval,
        seed=42,
    )
    model.eval()
    preds: list[set] = []
    golds: list[set] = []
    with torch.no_grad():
        for _ in range(n_eval):
            ep = ds.sample_episode()
            if not ep.support_contexts:
                continue
            all_offsets = []
            all_ids = []
            all_attn = []
            for ctx in ep.support_contexts:
                enc = _tokenize(tokenizer, ctx, max_len)
                all_offsets.append([(int(a), int(b)) for a, b in enc["offset_mapping"]])
                all_ids.append(enc["input_ids"])
                all_attn.append(enc["attention_mask"])
            max_l = max(len(x) for x in all_ids)
            pad_id = tokenizer.pad_token_id or 0
            input_ids = torch.tensor(
                [x + [pad_id] * (max_l - len(x)) for x in all_ids],
                dtype=torch.long,
                device=device,
            )
            attn = torch.tensor(
                [x + [0] * (max_l - len(x)) for x in all_attn],
                dtype=torch.long,
                device=device,
            )
            hs = model(input_ids, attn)
            n_class = len(ep.label_names)
            support_emb_list = []
            support_labels_list = []
            for (ctx_idx, cs, ce), lab in zip(ep.support_spans, ep.support_labels):
                ts_te = char_span_to_token_span(all_offsets[ctx_idx], cs, ce)
                if ts_te is None:
                    continue
                ts, te = ts_te
                emb = model.span_embedding(
                    hs[ctx_idx],
                    torch.tensor([[ts, te]], device=device, dtype=torch.long),
                )
                support_emb_list.append(emb.squeeze(0))
                support_labels_list.append(lab)
            if not support_emb_list:
                continue
            support_emb = torch.stack(support_emb_list, dim=0)
            support_labels_t = torch.tensor(support_labels_list, device=device, dtype=torch.long)
            prototypes = model.compute_prototypes(support_emb, support_labels_t)
            gold_set = set()
            pred_set = set()
            for (ctx_idx, cs, ce), lab in zip(ep.query_spans, ep.query_labels):
                if lab < n_class:
                    gold_set.add((ctx_idx, cs, ce, lab))
                ts_te = char_span_to_token_span(all_offsets[ctx_idx], cs, ce)
                if ts_te is None:
                    continue
                ts, te = ts_te
                emb = model.span_embedding(
                    hs[ctx_idx],
                    torch.tensor([[ts, te]], device=device, dtype=torch.long),
                )
                logits = model.compute_logits(emb, prototypes)
                pred_id = logits.argmax(dim=-1).item()
                if pred_id < n_class:
                    pred_set.add((ctx_idx, cs, ce, pred_id))
            preds.append(pred_set)
            golds.append(gold_set)
    if not preds:
        return 0.0
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task1", type=str, default="chemdner")
    ap.add_argument("--task2", type=str, default="genia")
    ap.add_argument("--rehearsal_ratio", type=float, default=0.2)
    ap.add_argument("--encoder", type=str, default="bert-base-cased")
    ap.add_argument("--n_way", type=int, default=5)
    ap.add_argument("--k_shot", type=int, default=5)
    ap.add_argument("--epochs_per_task", type=int, default=1)
    ap.add_argument("--max_episodes", type=int, default=200)
    ap.add_argument("--n_eval", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def data_path(ds: str) -> Path:
        p = BENCHMARKS / ds / "train.jsonl"
        if not p.exists():
            p = SPLIT_DIR / f"{ds}_train_with_evidence.jsonl"
        return p

    t1_path = data_path(args.task1)
    t2_path = data_path(args.task2)
    if not t1_path.exists() or not t2_path.exists():
        print(f"[fail] missing data: {t1_path} or {t2_path}")
        sys.exit(1)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)

    # Task 1: train on chemdner
    print(f"[info] Task 1: training on {args.task1}")
    t1_samples = load_jsonl(t1_path)
    ds_t1 = EpisodicSpanDataset(
        t1_samples,
        n_way=args.n_way,
        k_shot=args.k_shot,
        max_episodes=args.max_episodes,
        seed=args.seed,
    )
    model = PrototypicalSpanNER(args.encoder, n_classes=args.n_way).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    for ep in range(1, args.epochs_per_task + 1):
        model.train()
        for _ in range(args.max_episodes):
            episode = ds_t1.sample_episode()
            loss, _ = process_episode(model, episode, tokenizer, 256, device)
            if loss.requires_grad:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

    f1_t1_after_t1 = eval_dataset(
        model, t1_path, tokenizer, 256, device, args.n_way, args.k_shot, args.n_eval
    )
    print(f"[eval] {args.task1} after task1: F1={f1_t1_after_t1:.4f}")

    # Rehearsal buffer: sample from task1
    n_rehearsal = max(1, int(len(t1_samples) * args.rehearsal_ratio))
    rehearsal = random.sample(t1_samples, min(n_rehearsal, len(t1_samples)))

    # Task 2: train on genia + rehearsal
    print(f"[info] Task 2: training on {args.task2} + {len(rehearsal)} rehearsal")
    t2_samples = load_jsonl(t2_path)
    mixed = t2_samples + rehearsal
    random.shuffle(mixed)
    ds_t2 = EpisodicSpanDataset(
        mixed,
        n_way=args.n_way,
        k_shot=args.k_shot,
        max_episodes=args.max_episodes,
        seed=args.seed + 1,
    )
    for ep in range(1, args.epochs_per_task + 1):
        model.train()
        for _ in range(args.max_episodes):
            episode = ds_t2.sample_episode()
            loss, _ = process_episode(model, episode, tokenizer, 256, device)
            if loss.requires_grad:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

    f1_t1_after_t2 = eval_dataset(
        model, t1_path, tokenizer, 256, device, args.n_way, args.k_shot, args.n_eval
    )
    f1_t2_after_t2 = eval_dataset(
        model, t2_path, tokenizer, 256, device, args.n_way, args.k_shot, args.n_eval
    )
    print(f"[eval] {args.task1} after task2: F1={f1_t1_after_t2:.4f}")
    print(f"[eval] {args.task2} after task2: F1={f1_t2_after_t2:.4f}")

    bwt = f1_t1_after_t2 - f1_t1_after_t1
    print(f"[BWT] {bwt:.4f} (negative = forgetting)")

    out = {
        "task1": args.task1,
        "task2": args.task2,
        "f1_t1_after_t1": f1_t1_after_t1,
        "f1_t1_after_t2": f1_t1_after_t2,
        "f1_t2_after_t2": f1_t2_after_t2,
        "bwt": bwt,
        "rehearsal_ratio": args.rehearsal_ratio,
        "created_at": int(time.time()),
    }
    (ARTIFACTS / "metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ok] saved to {ARTIFACTS}")


if __name__ == "__main__":
    main()
