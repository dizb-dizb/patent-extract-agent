#!/usr/bin/env python3
"""
阶段二：冻结底座的度量微调 (Frozen Metric Fine-tuning)

使用极低/低资源梯度集（Subset-100 / Subset-1000），以 N-way K-shot 情景模拟方式训练。
冻结 RoBERTa + Span 提议层，仅训练投影模块 (Projector)。
Loss: 原型网络交叉熵；可选监督对比损失 (SCL)，默认关闭 (--scl_weight 0)。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys_path = str(ROOT)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from fewshot.episode_dataset import EpisodicSpanDataset, load_jsonl
from fewshot.projector_proto import ProjectorSpanProto, supervised_contrastive_loss

SPECIAL_OFFSETS = {(0, 0)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def char_span_to_token_span(offsets: list[tuple[int, int]], cs: int, ce: int) -> tuple[int, int] | None:
    if ce <= cs:
        return None
    valid = [(i, a, b) for i, (a, b) in enumerate(offsets) if (a, b) not in SPECIAL_OFFSETS and b > a]
    if not valid:
        return None
    ts = te = None
    for i, a, b in valid:
        if ts is None and a <= cs < b:
            ts = i
        if a < ce <= b:
            te = i + 1
            break
        if a >= ce:
            break
    if ts is None or te is None or te <= ts:
        for i, a, b in valid:
            if a == cs:
                ts = i
            if b == ce and ts is not None and i >= ts:
                te = i + 1
                break
    if ts is None or te is None or te <= ts:
        return None
    return (ts, te)


def _tokenize(tokenizer, text: str, max_len: int):
    return tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_len,
        padding=False,
    )


def process_episode(
    model: ProjectorSpanProto,
    episode,
    tokenizer,
    max_len: int,
    device: torch.device,
    scl_weight: float,
    scl_temp: float,
    use_amp: bool = False,
) -> tuple[torch.Tensor, int]:
    """Encode episode, compute CE + SCL. Returns (loss, num_query)."""
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

    with torch.amp.autocast("cuda", enabled=use_amp), torch.set_grad_enabled(model.training):
        hs = model(input_ids, attn)
        span_emb_list = []
        for ctx_idx, ts, te in support_spans:
            h = hs[ctx_idx]
            emb = _m.span_embedding(h, torch.tensor([[ts, te]], device=device, dtype=torch.long))
            span_emb_list.append(emb.squeeze(0))
        support_emb = torch.stack(span_emb_list, dim=0)
        support_labels_t = torch.tensor(support_labels, device=device, dtype=torch.long)
        prototypes = _m.compute_prototypes(support_emb, support_labels_t, n_class=n_class)

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
        query_labels_t = query_labels_t.clamp(0, logits.size(-1) - 1)
        loss_ce = torch.nn.functional.cross_entropy(logits.float(), query_labels_t)

        loss_scl = torch.tensor(0.0, device=device)
        if scl_weight > 0:
            all_emb = torch.cat([support_emb, query_emb], dim=0)
            all_labels = torch.cat([support_labels_t, query_labels_t], dim=0)
            if all_emb.size(0) >= 2:
                loss_scl = supervised_contrastive_loss(all_emb.float(), all_labels, temperature=scl_temp)

        loss = loss_ce + scl_weight * loss_scl
        return loss, len(query_labels)


def main() -> None:
    ap = argparse.ArgumentParser(description="阶段二：冻结底座，仅训练 Projector（CE + SCL）")
    ap.add_argument("--data", type=str, required=True, help="train_100.jsonl 或 train_1000.jsonl")
    ap.add_argument("--stage1_ckpt", type=str, required=True, help="encoder_span_proj.pt 路径")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--encoder", type=str, default="roberta-base")
    ap.add_argument("--n_way", type=int, default=5)
    ap.add_argument("--k_shot", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_episodes", type=int, default=4)
    ap.add_argument("--max_episodes", type=int, default=500)
    ap.add_argument("--fp16", action="store_true", help="混合精度加速")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    ap.add_argument(
        "--scl_weight",
        type=float,
        default=0.0,
        help="SCL 损失权重；0=不使用 SCL，仅原型网络 CE（推荐与论文对照实验）",
    )
    ap.add_argument("--scl_temp", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"data not found: {data_path}")

    ckpt_path = Path(args.stage1_ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"stage1 ckpt not found: {ckpt_path}")

    samples = load_jsonl(data_path)
    if not samples:
        raise RuntimeError("empty dataset")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    episode_ds = EpisodicSpanDataset(
        samples,
        n_way=args.n_way,
        k_shot=args.k_shot,
        query_per_class=5,
        neg_ratio=0.3,
        max_span_width=12,
        max_episodes=args.max_episodes,
        seed=args.seed,
    )
    n_classes = max(args.n_way, len(episode_ds.label_names))

    model = ProjectorSpanProto(
        encoder_name=args.encoder,
        n_classes=n_classes,
        projector_dim=0,
        metric=args.metric,
    )
    model.load_stage1_ckpt(str(ckpt_path))
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_ep = 0
        it = iter(episode_ds)
        pbar = tqdm(
            range(args.max_episodes),
            desc=f"proj_ep{ep}",
            leave=True,
            dynamic_ncols=False,
            ncols=110,
            file=sys.stdout,
        )
        for step_idx, _ in enumerate(pbar, start=1):
            batch_loss = []
            for _ in range(args.batch_episodes):
                try:
                    episode = next(it)
                except StopIteration:
                    it = iter(episode_ds)
                    episode = next(it)
                loss, _ = process_episode(
                    model, episode, tokenizer, args.max_len, device,
                    scl_weight=args.scl_weight,
                    scl_temp=args.scl_temp,
                    use_amp=args.fp16 and device.type == "cuda",
                )
                batch_loss.append(loss)
            if not batch_loss:
                continue
            loss = torch.stack(batch_loss).mean()
            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += loss.item()
            n_ep += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if step_idx == 1 or step_idx % 20 == 0 or step_idx == args.max_episodes:
                pct = int(step_idx * 100 / max(1, args.max_episodes))
                print(
                    f"[STAGE2][ep {ep}/{args.epochs}] step {step_idx}/{args.max_episodes} ({pct}%) "
                    f"loss={loss.item():.4f}"
                )
        avg = total_loss / n_ep if n_ep else 0
        print(f"epoch {ep} loss={avg:.4f}")

    ckpt_out = out_dir / "projector_stage2.pt"
    torch.save({
        "projector": model.projector.state_dict(),
        "encoder": model.encoder.state_dict(),
        "span_proj": model.span_proj.state_dict(),
    }, ckpt_out)
    (out_dir / "config_stage2.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ok] 阶段二完成 -> {ckpt_out}")


if __name__ == "__main__":
    main()
