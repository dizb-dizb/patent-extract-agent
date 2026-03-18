"""
Few-shot prototypical span NER training (episodic).

N-way K-shot: sample N classes, K support per class, query from same + negatives.
Output: metrics.json with P/R/F1, dataset_version.
"""

from __future__ import annotations

import argparse
import json
import math
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


def process_episode(
    model: PrototypicalSpanNER,
    episode,
    tokenizer,
    max_len: int,
    device: torch.device,
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

    # Unwrap DataParallel to access custom methods
    _m = model.module if hasattr(model, "module") else model

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
        emb = _m.span_embedding(h, torch.tensor([[ts, te]], device=device, dtype=torch.long))
        span_emb_list.append(emb.squeeze(0))
    support_emb = torch.stack(span_emb_list, dim=0)
    support_labels_t = torch.tensor(support_labels, device=device, dtype=torch.long)

    prototypes = _m.compute_prototypes(support_emb, support_labels_t)

    query_emb_list = []
    query_labels = []
    for (ctx_idx, cs, ce), lab in zip(episode.query_spans, episode.query_labels):
        offsets = all_offsets[ctx_idx]
        ts_te = char_span_to_token_span(offsets, cs, ce)
        if ts_te is None:
            continue
        ts, te = ts_te
        h = hs[ctx_idx]
        emb = _m.span_embedding(h, torch.tensor([[ts, te]], device=device, dtype=torch.long))
        query_emb_list.append(emb.squeeze(0))
        query_labels.append(lab)

    if not query_emb_list:
        return torch.tensor(0.0, device=device), 0

    query_emb = torch.stack(query_emb_list, dim=0)
    query_labels_t = torch.tensor(query_labels, device=device, dtype=torch.long)

    logits = _m.compute_logits(query_emb, prototypes)
    # clamp labels to valid range (logits has n_classes+1 columns incl. NONE)
    query_labels_t = query_labels_t.clamp(0, logits.size(-1) - 1)
    loss = torch.nn.functional.cross_entropy(logits, query_labels_t)
    return loss, len(query_labels)


def _flatten_spans_per_ctx(spans: set[tuple[int, int, int, int]]) -> set[tuple[int, int, int, int]]:
    """Keep only longest span per overlapping group (Flat F1).

    Two spans (a,b) and (c,d) overlap iff max(a,c) < min(b,d).
    We greedily keep the longest span and discard anything that overlaps with it.
    """
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
            if any(max(cs, ks) < min(ce, ke) for ks, ke, _ in kept):
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
            _em = model.module if hasattr(model, "module") else model
            n_class = len(ep.label_names)

            support_emb_list = []
            support_labels_list = []
            for (ctx_idx, cs, ce), lab in zip(ep.support_spans, ep.support_labels):
                ts_te = char_span_to_token_span(all_offsets[ctx_idx], cs, ce)
                if ts_te is None:
                    continue
                ts, te = ts_te
                emb = _em.span_embedding(hs[ctx_idx], torch.tensor([[ts, te]], device=device, dtype=torch.long))
                support_emb_list.append(emb.squeeze(0))
                support_labels_list.append(lab)
            if not support_emb_list:
                continue
            support_emb = torch.stack(support_emb_list, dim=0)
            support_labels_t = torch.tensor(support_labels_list, device=device, dtype=torch.long)
            prototypes = _em.compute_prototypes(support_emb, support_labels_t)

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
                emb = _em.span_embedding(hs[ctx_idx], torch.tensor([[ts, te]], device=device, dtype=torch.long))
                logits = _em.compute_logits(emb, prototypes)
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
    ap.add_argument("--encoder", type=str, default="roberta-base")
    ap.add_argument("--output_dir", type=str, default="artifacts/run_proto_span")
    ap.add_argument("--n_way", type=int, default=5)
    ap.add_argument("--k_shot", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30, help="Total training epochs")
    ap.add_argument("--warmup_ratio", type=float, default=0.1, help="Fraction of total episodes for LR warmup")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_episodes", type=int, default=4)
    ap.add_argument("--max_episodes", type=int, default=500)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--n_eval", type=int, default=50)
    ap.add_argument("--multi_gpu", action="store_true", help="Use DataParallel for multi-GPU")
    ap.add_argument("--train_labels", type=str, default="", help="Comma-separated meta-train labels (strict class split)")
    ap.add_argument("--test_labels", type=str, default="", help="Comma-separated meta-test labels (unseen at train)")
    ap.add_argument("--max_train_samples", type=int, default=0,
                    help="限制训练样本数 (0=不限制)。隔离实验下为 meta-train 类型句子上限，评测用独立 meta-test 句池")
    ap.add_argument(
        "--eval-pool-max",
        type=int,
        default=8000,
        help="隔离实验：meta-test 评测句池最大条数（从全量中含 test_labels 的句子抽取）",
    )
    # BiLSTM encoder options (B3/B5)
    ap.add_argument("--encoder_type", type=str, default="transformer", choices=["transformer", "bilstm"],
                    help="transformer=BERT/RoBERTa (default); bilstm=randomly-init BiLSTM (B3/B5)")
    ap.add_argument("--bilstm_embed_dim", type=int, default=100)
    ap.add_argument("--bilstm_hidden", type=int, default=256)
    ap.add_argument("--bilstm_layers", type=int, default=2)
    ap.add_argument("--freeze_encoder", action="store_true",
                    help="冻结 encoder（预训练权重不微调），仅训练 span_proj，用于原型网络+未训练模型对照")
    args = ap.parse_args()

    set_seed(args.seed)
    root = Path(__file__).resolve().parent
    data_path = root / args.data if not Path(args.data).is_absolute() else Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"missing data: {data_path}")

    samples = load_jsonl(data_path)
    if not samples:
        raise RuntimeError("empty dataset")

    train_labels_list = [x.strip() for x in args.train_labels.split(",") if x.strip()] or None
    test_labels_list = [x.strip() for x in args.test_labels.split(",") if x.strip()] or None

    def _labels_in_sample(s: dict) -> set[str]:
        out: set[str] = set()
        for sp in s.get("spans") or []:
            if isinstance(sp, dict):
                out.add(str(sp.get("label") or "term").strip() or "term")
        return out

    val_path = Path(args.val) if args.val else None
    if val_path and not val_path.is_absolute():
        val_path = root / val_path

    # --- 类别隔离 + 梯度 n：n=仅 meta-train 句量；评测= meta-test 类型句池（train+可选 val，与 --val 传入兼容）---
    if train_labels_list and test_labels_list and args.max_train_samples > 0:
        tl = set(train_labels_list)
        te = set(test_labels_list)
        pool_tr = [s for s in samples if _labels_in_sample(s) & tl]
        strict_tr = [s for s in pool_tr if not (_labels_in_sample(s) & te)]
        if len(strict_tr) >= args.max_train_samples:
            pool_tr = strict_tr
            print("[isolate] meta-train 句池：优先选用不含 meta-test 类型标注的句子（减少标注泄漏）")
        random.shuffle(pool_tr)
        train_s = pool_tr[: args.max_train_samples]
        if not train_s and pool_tr:
            train_s = pool_tr[: min(len(pool_tr), args.max_train_samples)]
        if not train_s:
            random.shuffle(samples)
            train_s = samples[: min(args.max_train_samples, len(samples))]
            print(f"[warn] 无含 meta-train 类型的句子，回退为随机 {len(train_s)} 句")
        print(
            f"[isolate] meta-train: {len(train_s)} 句 (目标上限 n={args.max_train_samples})，"
            f"仅含类型 {sorted(tl)[:5]}{'...' if len(tl) > 5 else ''}"
        )

        pool_source_te = list(samples)
        if val_path and val_path.exists():
            pool_source_te.extend(load_jsonl(val_path))
            print("[isolate] 评测池合并 train + 外部 val 中含 meta-test 类型的句子")
        pool_te = [s for s in pool_source_te if _labels_in_sample(s) & te]
        ctx_key = lambda s: (s.get("context") or "")[:800]
        tr_ctx = {ctx_key(s) for s in train_s}
        val_s = [s for s in pool_te if ctx_key(s) not in tr_ctx]
        random.shuffle(val_s)
        val_s = val_s[: args.eval_pool_max]
        if len(val_s) < 50:
            val_s = (pool_te[: args.eval_pool_max] if pool_te else [])[: args.eval_pool_max]
        if not val_s and pool_te:
            val_s = pool_te[: args.eval_pool_max]
            print("[warn] meta-test 池与训练去重后过少，使用含 test 类型全池子集")
        print(
            f"[isolate] meta-test 评测池: {len(val_s)} 句（含 test_labels，与训练句不重复上下文），"
            f"类型示例 {sorted(te)[:5]}{'...' if len(te) > 5 else ''}"
        )
    elif val_path and val_path.exists():
        val_s = load_jsonl(val_path)
        if args.max_train_samples > 0:
            random.shuffle(samples)
            train_s = samples[: args.max_train_samples]
            print(f"[data] 外部 val + 训练限 n={len(train_s)}")
        else:
            train_s = samples
    else:
        if args.max_train_samples > 0:
            random.shuffle(samples)
            samples = samples[: args.max_train_samples]
            print(f"[data] 限制训练样本数: {len(samples)}")
        random.shuffle(samples)
        n_train = int(len(samples) * args.train_ratio)
        train_s = samples[:n_train]
        val_s = samples[n_train:] if n_train < len(samples) else samples[-max(1, len(samples) // 10) :]

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
    if getattr(args, "freeze_encoder", False) and args.encoder_type == "transformer":
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("[freeze] 已冻结 encoder，仅训练 span_proj")
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * args.max_episodes
    warmup_steps = int(total_steps * args.warmup_ratio)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_ver = dataset_version(data_path)
    (out_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    best_f1 = 0.0
    best_state = None
    global_step = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_queries = 0
        pbar = tqdm(range(args.max_episodes), desc=f"ep{ep}", leave=False)
        for step in pbar:
            episode = ds_train.sample_episode(use_test_labels=False)
            loss, nq = process_episode(
                model, episode, tokenizer, args.max_len, device,
            )
            if nq == 0:
                continue
            total_loss += loss.item()
            total_queries += nq
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        prec, rec, f1, flat_f1 = eval_episodes(
            model, ds_val, tokenizer, args.max_len, device, args.n_eval,
            use_test_labels=bool(test_labels_list),
        )
        cur_lr = scheduler.get_last_lr()[0]
        if f1 > best_f1:
            best_f1 = f1
            _m = model.module if hasattr(model, "module") else model
            best_state = {k: v.cpu().clone() for k, v in _m.state_dict().items()}
            torch.save(_m.state_dict(), out_dir / "model.pt")

        metrics = {
            "name": "proto_span_fewshot",
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "best_f1": best_f1,
            "nested_micro_f1": f1,
            "flat_f1": flat_f1,
            "lr": cur_lr,
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
        mark = " ★" if f1 >= best_f1 else ""
        print(f"[eval] ep={ep} lr={cur_lr:.2e} P={prec:.4f} R={rec:.4f} F1={f1:.4f} FlatF1={flat_f1:.4f}{mark}")

    model_pt = out_dir / "model.pt"
    if best_state is None:
        torch.save((model.module if hasattr(model, "module") else model).state_dict(), model_pt)
    model_pt.unlink(missing_ok=True)
    print(f"[ok] best_f1={best_f1:.4f}  saved to {out_dir} (已释放 model.pt，保留 metrics.json)")


if __name__ == "__main__":
    main()
