#!/usr/bin/env python3
"""
解耦二阶段（stage1 BCE span pretrain + stage2 Projector prototypical fine-tune）后的验证评测。

目标：加载 `projector_stage2.pt`（内含 encoder/span_proj/projector），在官方 `test.jsonl`（或 `val.jsonl`）
上跑 N-way K-shot episode，并输出 metrics.json（含 P/R/F1 + flat_f1）。

说明：评测使用 EpisodicSpanDataset 与 train_fewshot_proto_span 相同的原型推理逻辑，但加 tqdm 进度条。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fewshot.episode_dataset import EpisodicSpanDataset, load_jsonl
from fewshot.model import char_span_to_token_span
from fewshot.projector_proto import ProjectorSpanProto

SPECIAL_OFFSETS = {(0, 0)}


def _dataset_version(path: Path) -> str:
    """Short content hash for reproducibility (no external dataset_version.py)."""
    text = path.read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _tokenize(tokenizer, text: str, max_len: int):
    return tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_len,
        padding=False,
    )


def _flatten_spans_per_ctx(spans: set[tuple[int, int, int, int]]) -> set[tuple[int, int, int, int]]:
    """Keep only longest span per overlapping group (Flat F1)."""

    by_ctx: dict[int, list[tuple[int, int, int]]] = {}
    for ctx_idx, cs, ce, lab in spans:
        by_ctx.setdefault(ctx_idx, []).append((cs, ce, lab))

    out: set[tuple[int, int, int, int]] = set()
    for ctx_idx, items in by_ctx.items():
        # sort by length desc then keep non-overlapping
        items_sorted = sorted(items, key=lambda x: (x[1] - x[0]), reverse=True)
        kept: list[tuple[int, int, int]] = []
        for cs, ce, lab in items_sorted:
            if any(max(cs, ks) < min(ce, ke) for ks, ke, _ in kept):
                continue
            kept.append((cs, ce, lab))
        for cs, ce, lab in kept:
            out.add((ctx_idx, cs, ce, lab))
    return out


def micro_prf(
    pred: list[set[tuple[int, int, int, int]]],
    gold: list[set[tuple[int, int, int, int]]],
    flat: bool = False,
) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for p, g in zip(pred, gold):
        if flat:
            p = _flatten_spans_per_ctx(p)
            g = _flatten_spans_per_ctx(g)
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def evaluate_episodes(
    model: ProjectorSpanProto,
    episode_iter: EpisodicSpanDataset,
    tokenizer,
    max_len: int,
    device: torch.device,
    n_eval: int,
) -> tuple[float, float, float, float]:
    """Run n_eval sampled episodes and return (P,R,F1,flatF1)."""

    model.eval()
    preds: list[set[tuple[int, int, int, int]]] = []
    golds: list[set[tuple[int, int, int, int]]] = []

    for _ in tqdm(range(n_eval), desc="eval", leave=False):
        ep = episode_iter.sample_episode(use_test_labels=False)
        ctx_list = ep.support_contexts
        if not ctx_list:
            continue

        # Remap episode-local class ids to [0..n_class-1].
        # EpisodicSpanDataset stores label ids in the global label_to_id space,
        # but prototypical heads expect labels in the episode-local contiguous space.
        global_ids = [episode_iter.label_to_id[l] for l in ep.label_names]
        local_map = {gid: i for i, gid in enumerate(global_ids)}

        # tokenize all support contexts together
        all_offsets: list[list[tuple[int, int]]] = []
        all_ids: list[list[int]] = []
        all_attn: list[list[int]] = []
        for ctx in ctx_list:
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

        with torch.no_grad():
            hs = model(input_ids, attn)  # (N_ctx, T, H)

        n_class = len(ep.label_names)

        # build support embeddings -> prototypes
        support_emb_list: list[torch.Tensor] = []
        support_labels_list: list[int] = []
        for (ctx_idx, cs, ce), lab in zip(ep.support_spans, ep.support_labels):
            if int(lab) not in local_map:
                continue
            offsets = all_offsets[ctx_idx]
            ts_te = char_span_to_token_span(offsets, cs, ce)
            if ts_te is None:
                continue
            ts, te = ts_te
            emb = model.span_embedding(hs[ctx_idx], torch.tensor([[ts, te]], device=device, dtype=torch.long))
            support_emb_list.append(emb.squeeze(0))
            support_labels_list.append(local_map[int(lab)])

        if not support_emb_list:
            continue

        support_emb = torch.stack(support_emb_list, dim=0)
        support_labels_t = torch.tensor(support_labels_list, device=device, dtype=torch.long)
        prototypes = model.compute_prototypes(support_emb, support_labels_t, n_class=n_class)

        # build pred/gold sets with projected prototypes
        gold_set: set[tuple[int, int, int, int]] = set()
        pred_set: set[tuple[int, int, int, int]] = set()

        for (ctx_idx, cs, ce), lab in zip(ep.query_spans, ep.query_labels):
            # gold
            if lab < n_class:
                # remap global label id -> local label id
                gold_lab = local_map.get(int(lab), int(lab))
                gold_set.add((ctx_idx, cs, ce, int(gold_lab)))

            # pred
            offsets = all_offsets[ctx_idx]
            ts_te = char_span_to_token_span(offsets, cs, ce)
            if ts_te is None:
                continue
            ts, te = ts_te
            emb = model.span_embedding(hs[ctx_idx], torch.tensor([[ts, te]], device=device, dtype=torch.long))
            logits = model.compute_logits(emb, prototypes)  # (1, n_class+1)
            pred_id = logits.argmax(dim=-1).item()
            if pred_id < n_class:
                pred_set.add((ctx_idx, cs, ce, int(pred_id)))

        golds.append(gold_set)
        preds.append(pred_set)

    if not preds:
        return 0.0, 0.0, 0.0, 0.0
    prec, rec, f1 = micro_prf(preds, golds, flat=False)
    _, _, flat_f1 = micro_prf(preds, golds, flat=True)
    return prec, rec, f1, flat_f1


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval projector stage2 via episodic prototypical span NER.")
    ap.add_argument("--dataset", type=str, required=True, choices=["genia", "fewnerd", "chemdner"])
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    ap.add_argument("--encoder", type=str, default="roberta-base")
    ap.add_argument("--stage2_ckpt", type=str, required=True, help="projector_stage2.pt")
    ap.add_argument("--output_dir", type=str, default="", help="defaults to artifacts/two_stage_decoupled/<ds>_eval")
    ap.add_argument("--n_way", type=int, default=5)
    ap.add_argument("--k_shot", type=int, default=5)
    ap.add_argument("--n_eval", type=int, default=100, help="number of episodes to sample for evaluation")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--max_span_width", type=int, default=12)
    ap.add_argument("--query_per_class", type=int, default=5)
    ap.add_argument("--neg_ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = ROOT / "data" / "benchmarks" / args.dataset / ("test.jsonl" if args.split == "test" else "val.jsonl")
    if not data_path.exists():
        raise FileNotFoundError(f"missing eval split: {data_path}")

    stage2_ckpt = Path(args.stage2_ckpt)
    if not stage2_ckpt.is_absolute():
        stage2_ckpt = ROOT / stage2_ckpt
    if not stage2_ckpt.exists():
        raise FileNotFoundError(f"missing stage2 ckpt: {stage2_ckpt}")

    out_dir = Path(args.output_dir) if args.output_dir else (ROOT / "artifacts" / f"two_stage_decoupled_eval_{args.dataset}_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(data_path)
    episode_ds = EpisodicSpanDataset(
        samples,
        n_way=args.n_way,
        k_shot=args.k_shot,
        query_per_class=args.query_per_class,
        neg_ratio=args.neg_ratio,
        max_span_width=args.max_span_width,
        max_episodes=args.n_eval,
        seed=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)

    model = ProjectorSpanProto(
        encoder_name=args.encoder,
        n_classes=args.n_way,
        metric="cosine",
    ).to(device)

    ckpt = torch.load(stage2_ckpt, map_location="cpu")
    # stage2 ckpt format (from train_stage2_projector_proto.py)
    if "encoder" in ckpt:
        model.encoder.load_state_dict(ckpt["encoder"], strict=False)
    if "span_proj" in ckpt:
        model.span_proj.load_state_dict(ckpt["span_proj"], strict=False)
    if "projector" in ckpt:
        model.projector.load_state_dict(ckpt["projector"], strict=False)

    prec, rec, f1, flat_f1 = evaluate_episodes(
        model=model,
        episode_iter=episode_ds,
        tokenizer=tokenizer,
        max_len=args.max_len,
        device=device,
        n_eval=args.n_eval,
    )

    metrics = {
        "name": "two_stage_decoupled_projector_eval",
        "dataset": args.dataset,
        "split": args.split,
        "structure": "projector_span_proto",
        "encoder": args.encoder,
        "stage2_ckpt": str(stage2_ckpt),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "flat_f1": flat_f1,
        "best_f1": f1,
        "epochs": 0,
        "elapsed_s": None,
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "n_eval_episodes": args.n_eval,
        "dataset_version": _dataset_version(data_path),
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] eval done -> {out_dir / 'metrics.json'}")
    print(f"[metrics] P={prec:.4f} R={rec:.4f} F1={f1:.4f} flat_f1={flat_f1:.4f}")


if __name__ == "__main__":
    main()

