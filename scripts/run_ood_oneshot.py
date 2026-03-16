"""
Zero/One-shot OOD 评测：在未见类别上，仅给 1 个支撑样本，评估泛化能力。

用法：
  python scripts/run_ood_oneshot.py --train_data data/benchmarks/fewnerd/train.jsonl --test_data data/benchmarks/genia/test.jsonl --k_shot 1
  python scripts/run_ood_oneshot.py --model artifacts/run_proto_span/fewnerd/model.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer

from fewshot.episode_dataset import EpisodicSpanDataset, load_jsonl
from fewshot.model import PrototypicalSpanNER, char_span_to_token_span

ARTIFACTS = ROOT / "artifacts" / "ood_oneshot"
SPECIAL_OFFSETS = {(0, 0)}


def _tokenize(tokenizer, text: str, max_len: int):
    return tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_len,
        padding=False,
    )


def eval_ood(
    model: PrototypicalSpanNER,
    test_samples: list[dict],
    tokenizer,
    max_len: int,
    device: torch.device,
    n_way: int,
    k_shot: int,
    n_eval: int,
    seed: int,
) -> dict:
    """Evaluate on OOD test set with K-shot support."""
    ds = EpisodicSpanDataset(
        test_samples,
        n_way=n_way,
        k_shot=k_shot,
        query_per_class=5,
        neg_ratio=0.3,
        max_span_width=12,
        max_episodes=n_eval,
        seed=seed,
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
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data", type=str, default="")
    ap.add_argument("--test_data", type=str, required=True)
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--encoder", type=str, default="bert-base-cased")
    ap.add_argument("--n_way", type=int, default=5)
    ap.add_argument("--k_shot", type=int, default=1)
    ap.add_argument("--n_eval", type=int, default=50)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_path = Path(args.test_data)
    if not test_path.is_absolute():
        test_path = ROOT / test_path
    if not test_path.exists():
        print(f"[fail] test data not found: {test_path}")
        sys.exit(1)

    test_samples = load_jsonl(test_path)
    if not test_samples:
        print("[fail] empty test set")
        sys.exit(1)

    model = PrototypicalSpanNER(args.encoder, n_classes=args.n_way).to(device)
    if args.model:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        if model_path.exists():
            state = torch.load(model_path, map_location=device)
            if hasattr(model, "module"):
                model.module.load_state_dict(state, strict=False)
            else:
                model.load_state_dict(state, strict=False)
            print(f"[info] loaded model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
    metrics = eval_ood(
        model, test_samples, tokenizer, args.max_len, device,
        args.n_way, args.k_shot, args.n_eval, args.seed,
    )

    print(f"[OOD {args.k_shot}-shot] P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = {
        "k_shot": args.k_shot,
        "test_data": str(test_path),
        "model": args.model,
        **metrics,
    }
    (ARTIFACTS / "metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ok] saved to {ARTIFACTS}")


if __name__ == "__main__":
    main()
