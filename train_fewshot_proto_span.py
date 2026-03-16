"""
Few-shot prototypical span NER training (episodic).

N-way K-shot: sample N classes, K support per class, query from same + negatives.
Output: metrics.json with P/R/F1, dataset_version.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset_version import dataset_version
from fewshot.episode_dataset import EpisodicSpanDataset, load_jsonl
from fewshot.model import PrototypicalSpanNER, char_span_to_token_span

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


def _scl_loss(emb: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """Supervised contrastive loss: pull same-class together, push different-class apart."""
    if emb.size(0) < 2:
        return torch.tensor(0.0, device=emb.device)
    emb_n = torch.nn.functional.normalize(emb, dim=-1)
    sim = torch.mm(emb_n, emb_n.t()) / temperature
    mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask = mask.float()
    exp_sim = torch.exp(sim)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    mask_diag = mask - torch.eye(emb.size(0), device=emb.device)
    pos_count = mask_diag.sum(dim=1).clamp(min=1)
    loss = -(mask_diag * log_prob).sum(dim=1) / pos_count
    return loss.mean()


def process_episode(
    model: PrototypicalSpanNER,
    episode,
    tokenizer,
    max_len: int,
    device: torch.device,
    scl_weight: float = 0.0,
    scl_temperature: float = 0.07,
) -> tuple[torch.Tensor, int]:
    """Encode episode, compute loss. Returns (loss, num_query)."""
    ctx_list = episode.support_contexts
    if not ctx_list:
        return torch.tensor(0.0, device=device), 0

    all_input_ids = []
    all_attn = []
    all_offsets = []

    for ctx in ctx_list:
        enc = _tokenize(tokenizer, ctx, max_len)
        all_input_ids.append(enc["input_ids"])
        all_attn.append(enc["attention_mask"])
        all_offsets.append([(int(a), int(b)) for a, b in enc["offset_mapping"]])

    max_len_actual = max(len(x) for x in all_input_ids)
    pad_id = tokenizer.pad_token_id or 0
    input_ids = torch.tensor(
        [x + [pad_id] * (max_len_actual - len(x)) for x in all_input_ids],
        dtype=torch.long,
        device=device,
    )
    attn = torch.tensor(
        [x + [0] * (max_len_actual - len(x)) for x in all_attn],
        dtype=torch.long,
        device=device,
    )

    with torch.set_grad_enabled(model.training):
        hs = model(input_ids, attn)

    n_class = len(episode.label_names)
    support_spans: list[tuple[int, int, int]] = []
    support_labels: list[int] = []

    for (ctx_idx, cs, ce), lab in zip(episode.support_spans, episode.support_labels):
        offsets = all_offsets[ctx_idx]
        ts_te = char_span_to_token_span(offsets, cs, ce)
        if ts_te is None:
            continue
        ts, te = ts_te
        support_spans.append((ctx_idx, ts, te))
        support_labels.append(lab)

    if not support_spans:
        return torch.tensor(0.0, device=device), 0

    span_emb_list = []
    for ctx_idx, ts, te in support_spans:
        h = hs[ctx_idx]
        emb = model.span_embedding(h, torch.tensor([[ts, te]], device=device, dtype=torch.long))
        span_emb_list.append(emb.squeeze(0))
    support_emb = torch.stack(span_emb_list, dim=0)
    support_labels_t = torch.tensor(support_labels, device=device, dtype=torch.long)

    prototypes = model.compute_prototypes(support_emb, support_labels_t)

    query_emb_list = []
    query_labels = []
    for (ctx_idx, cs, ce), lab in zip(episode.query_spans, episode.query_labels):
        offsets = all_offsets[ctx_idx]
        ts_te = char_span_to_token_span(offsets, cs, ce)
        if ts_te is None:
            continue
        ts, te = ts_te
        h = hs[ctx_idx]
        emb = model.span_embedding(h, torch.tensor([[ts, te]], device=device, dtype=torch.long))
        query_emb_list.append(emb.squeeze(0))
        query_labels.append(lab)

    if not query_emb_list:
        return torch.tensor(0.0, device=device), 0

    query_emb = torch.stack(query_emb_list, dim=0)
    query_labels_t = torch.tensor(query_labels, device=device, dtype=torch.long)

    logits = model.compute_logits(query_emb, prototypes)
    loss = torch.nn.functional.cross_entropy(logits, query_labels_t)
    if scl_weight > 0 and query_emb.size(0) >= 2:
        scl = _scl_loss(query_emb, query_labels_t, scl_temperature)
        loss = loss + scl_weight * scl
    return loss, len(query_labels)


def _flatten_spans_per_ctx(spans: set[tuple[int, int, int, int]]) -> set[tuple[int, int, int, int]]:
    """Keep only longest span per overlapping group (Flat F1)."""
    by_ctx: dict[int, list[tuple[int, int, int]]] = {}
    for ctx_idx, cs, ce, lab in spans:
        if ctx_idx not in by_ctx:
            by_ctx[ctx_idx] = []
        by_ctx[ctx_idx].append((cs, ce, lab))
    out: set[tuple[int, int, int, int]] = set()
    for ctx_idx, items in by_ctx.items():
        items_sorted = sorted(items, key=lambda x: (x[1] - x[0]), reverse=True)
        kept: list[tuple[int, int, int]] = []
        for cs, ce, lab in items_sorted:
            if any(cs < ke < ce or cs < kd < ce or (ke <= cs and ce <= kd) for ke, kd, _ in kept):
                continue
            kept.append((cs, ce, lab))
        for cs, ce, lab in kept:
            out.add((ctx_idx, cs, ce, lab))
    return out


def micro_prf(
    pred: list[set[tuple[int, int, int, int]]], gold: list[set[tuple[int, int, int, int]]],
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


def eval_episodes(
    model: PrototypicalSpanNER,
    episode_iter,
    tokenizer,
    max_len: int,
    device: torch.device,
    n_eval: int,
    use_test_labels: bool = False,
) -> tuple[float, float, float, float]:
    model.eval()
    preds: list[set[tuple[int, int, int]]] = []
    golds: list[set[tuple[int, int, int]]] = []
    with torch.no_grad():
        for i in range(n_eval):
            ep = episode_iter.sample_episode(use_test_labels=use_test_labels) if hasattr(episode_iter, "sample_episode") else next(episode_iter)
            ctx_list = ep.support_contexts
            if not ctx_list:
                continue
            all_offsets = []
            all_ids = []
            all_attn = []
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

            hs = model(input_ids, attn)
            n_class = len(ep.label_names)

            support_emb_list = []
            support_labels_list = []
            for (ctx_idx, cs, ce), lab in zip(ep.support_spans, ep.support_labels):
                ts_te = char_span_to_token_span(all_offsets[ctx_idx], cs, ce)
                if ts_te is None:
                    continue
                ts, te = ts_te
                emb = model.span_embedding(hs[ctx_idx], torch.tensor([[ts, te]], device=device, dtype=torch.long))
                support_emb_list.append(emb.squeeze(0))
                support_labels_list.append(lab)
            if not support_emb_list:
                continue
            support_emb = torch.stack(support_emb_list, dim=0)
            support_labels_t = torch.tensor(support_labels_list, device=device, dtype=torch.long)
            prototypes = model.compute_prototypes(support_emb, support_labels_t)

            gold_set: set[tuple[int, int, int, int]] = set()
            for (ctx_idx, cs, ce), lab in zip(ep.query_spans, ep.query_labels):
                if lab < n_class:
                    gold_set.add((ctx_idx, cs, ce, lab))

            pred_set: set[tuple[int, int, int, int]] = set()
            for (ctx_idx, cs, ce), lab in zip(ep.query_spans, ep.query_labels):
                ts_te = char_span_to_token_span(all_offsets[ctx_idx], cs, ce)
                if ts_te is None:
                    continue
                ts, te = ts_te
                emb = model.span_embedding(hs[ctx_idx], torch.tensor([[ts, te]], device=device, dtype=torch.long))
                logits = model.compute_logits(emb, prototypes)
                pred_id = logits.argmax(dim=-1).item()
                if pred_id < n_class:
                    pred_set.add((ctx_idx, cs, ce, pred_id))

            golds.append(gold_set)
            preds.append(pred_set)

    if not preds:
        return 0.0, 0.0, 0.0, 0.0
    prec, rec, f1 = micro_prf(preds, golds, flat=False)
    flat_prec, flat_rec, flat_f1 = micro_prf(preds, golds, flat=True)
    return prec, rec, f1, flat_f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="train_spans_augmented.jsonl")
    ap.add_argument("--val", type=str, default="", help="Optional separate val file")
    ap.add_argument("--config", type=str, default="configs/proto_default.json")
    ap.add_argument("--encoder", type=str, default="hfl/chinese-roberta-wwm-ext")
    ap.add_argument("--output_dir", type=str, default="artifacts/run_proto_span")
    ap.add_argument("--n_way", type=int, default=5)
    ap.add_argument("--k_shot", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_episodes", type=int, default=4)
    ap.add_argument("--max_episodes", type=int, default=500)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--n_eval", type=int, default=50)
    ap.add_argument("--multi_gpu", action="store_true", help="Use DataParallel for multi-GPU")
    ap.add_argument("--scl_weight", type=float, default=0.0, help="Supervised contrastive loss weight (0=disabled)")
    ap.add_argument("--scl_temperature", type=float, default=0.07)
    ap.add_argument("--train_labels", type=str, default="", help="Comma-separated meta-train labels (strict class split)")
    ap.add_argument("--test_labels", type=str, default="", help="Comma-separated meta-test labels (unseen at train)")
    # BiLSTM encoder options (B3/B5)
    ap.add_argument("--encoder_type", type=str, default="transformer", choices=["transformer", "bilstm"],
                    help="transformer=BERT/RoBERTa (default); bilstm=randomly-init BiLSTM (B3/B5)")
    ap.add_argument("--bilstm_embed_dim", type=int, default=100)
    ap.add_argument("--bilstm_hidden", type=int, default=256)
    ap.add_argument("--bilstm_layers", type=int, default=2)
    args = ap.parse_args()

    set_seed(args.seed)
    root = Path(__file__).resolve().parent
    data_path = root / args.data if not Path(args.data).is_absolute() else Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"missing data: {data_path}")

    samples = load_jsonl(data_path)
    if not samples:
        raise RuntimeError("empty dataset")

    val_path = Path(args.val) if args.val else None
    if val_path and not val_path.is_absolute():
        val_path = root / val_path
    if val_path and val_path.exists():
        val_s = load_jsonl(val_path)
        train_s = samples
    else:
        random.shuffle(samples)
        n_train = int(len(samples) * args.train_ratio)
        train_s = samples[:n_train]
        val_s = samples[n_train:] if n_train < len(samples) else samples[-max(1, len(samples) // 10) :]

    train_labels_list = [x.strip() for x in args.train_labels.split(",") if x.strip()] or None
    test_labels_list = [x.strip() for x in args.test_labels.split(",") if x.strip()] or None

    ds_train = EpisodicSpanDataset(
        train_s,
        n_way=args.n_way,
        k_shot=args.k_shot,
        query_per_class=5,
        neg_ratio=0.3,
        max_span_width=12,
        max_episodes=args.max_episodes,
        seed=args.seed,
        train_labels=train_labels_list,
        test_labels=test_labels_list,
    )
    ds_val = EpisodicSpanDataset(
        val_s,
        n_way=args.n_way,
        k_shot=args.k_shot,
        query_per_class=5,
        neg_ratio=0.3,
        max_span_width=12,
        max_episodes=args.n_eval * 2,
        seed=args.seed + 1,
        train_labels=train_labels_list,
        test_labels=test_labels_list,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.encoder_type == "bilstm":
        # Build word vocab from training data for BiLSTM
        from train_bilstm_crf import build_vocab as _build_vocab, simple_tokenize as _simple_tokenize

        bilstm_word_to_id, bilstm_vocab = _build_vocab(train_s)
        bilstm_vocab_size = len(bilstm_vocab)

        class _BiLSTMTokenizerWrapper:
            """Wrap whitespace tokenizer to match HuggingFace tokenizer API."""
            pad_token_id = 0

            def __call__(self, text: str, return_offsets_mapping=True,
                         truncation=True, max_length=256, padding=False):
                tokens_info = _simple_tokenize(text)[:max_length]
                input_ids = [bilstm_word_to_id.get(t.lower(), 1) for t, _, _ in tokens_info]
                attn = [1] * len(input_ids)
                offsets = [(ts, te) for _, ts, te in tokens_info]
                return {
                    "input_ids": input_ids,
                    "attention_mask": attn,
                    "offset_mapping": offsets,
                }

        tokenizer = _BiLSTMTokenizerWrapper()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
        bilstm_vocab_size = 30000

    model = PrototypicalSpanNER(
        args.encoder,
        n_classes=args.n_way,
        metric=args.metric,
        encoder_type=args.encoder_type,
        bilstm_vocab_size=bilstm_vocab_size,
        bilstm_embed_dim=args.bilstm_embed_dim,
        bilstm_hidden=args.bilstm_hidden,
        bilstm_layers=args.bilstm_layers,
    ).to(device)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_ver = dataset_version(data_path)
    (out_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_queries = 0
        pbar = tqdm(range(args.max_episodes), desc=f"ep{ep}", leave=False)
        for step in pbar:
            episode = ds_train.sample_episode(use_test_labels=False)
            loss, nq = process_episode(
                model, episode, tokenizer, args.max_len, device,
                scl_weight=args.scl_weight, scl_temperature=args.scl_temperature,
            )
            if nq == 0:
                continue
            total_loss += loss.item()
            total_queries += nq
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.set_postfix(loss=loss.item())

        prec, rec, f1, flat_f1 = eval_episodes(
            model, ds_val, tokenizer, args.max_len, device, args.n_eval,
            use_test_labels=bool(test_labels_list),
        )
        metrics = {
            "name": "proto_span_fewshot",
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "nested_micro_f1": f1,
            "flat_f1": flat_f1,
            "encoder": args.encoder,
            "encoder_type": args.encoder_type,
            "data": str(data_path),
            "dataset_version": ds_ver,
            "epoch": ep,
            "n_way": args.n_way,
            "k_shot": args.k_shot,
            "created_at": int(time.time()),
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] ep={ep} P={prec:.4f} R={rec:.4f} F1={f1:.4f} FlatF1={flat_f1:.4f}")

    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"[ok] saved to {out_dir}")


if __name__ == "__main__":
    main()
