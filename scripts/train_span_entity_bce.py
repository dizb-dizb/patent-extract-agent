#!/usr/bin/env python3
"""
阶段一：领域通用跨度预训练 (Class-Agnostic Span Pre-training)

使用高资源梯度集（如 Subset-10000），RoBERTa + Span 提议层，二分类（实体 vs 非实体）。
Loss: BCE。输出 encoder_span_proj.pt 供阶段二加载。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
SPECIAL_OFFSETS = {(0, 0)}


def load_jsonl(path: Path) -> list[dict]:
    out = []
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


def sample_negative_spans(
    seq_len: int,
    gold_spans: set[tuple[int, int]],
    max_width: int,
    k: int,
    valid_mask: list[bool],
) -> list[tuple[int, int]]:
    neg = []
    for _ in range(k * 3):
        if len(neg) >= k:
            break
        s = random.randint(0, max(0, seq_len - 2))
        w = random.randint(1, max_width)
        e = min(seq_len, s + w)
        if e <= s or (s, e) in gold_spans:
            continue
        if not (valid_mask[s] and valid_mask[e - 1]):
            continue
        neg.append((s, e))
    return neg[:k]


class SpanEntityBCE(nn.Module):
    """RoBERTa + Span 提议层 + 二分类头（实体 vs 非实体）"""

    def __init__(self, encoder_name: str, hidden_dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        h = self.encoder.config.hidden_size
        self.span_proj = nn.Sequential(
            nn.Linear(h * 3, h),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
        )
        self.binary_head = nn.Linear(h, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def span_logits(self, token_emb: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        h = token_emb
        starts = spans[:, 0]
        ends = spans[:, 1] - 1
        h_start = h[starts]
        h_end = h[ends]
        pooled = []
        for i in range(spans.size(0)):
            s, e = int(spans[i, 0].item()), int(spans[i, 1].item())
            pooled.append(h[s:e].mean(dim=0))
        h_mean = torch.stack(pooled, dim=0)
        feat = torch.cat([h_start, h_end, h_mean], dim=-1)
        z = self.span_proj(feat)
        return self.binary_head(z).squeeze(-1)


class BCESpanDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer, max_len: int):
        self.samples = samples
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        text = str(s.get("context") or "")
        enc = self.tok(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_len,
            padding=False,
        )
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
        gold = []
        for sp in s.get("spans") or []:
            if not isinstance(sp, dict):
                continue
            cs, ce = int(sp.get("start", 0)), int(sp.get("end", 0))
            ts_te = char_span_to_token_span(offsets, cs, ce)
            if ts_te:
                gold.append(ts_te)
        return {"input_ids": input_ids, "attention_mask": attn, "offsets": offsets, "gold": gold}


def collate(batch: list[dict], pad_id: int) -> dict:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attn = []
    offsets = []
    gold = []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_id] * pad)
        attn.append(x["attention_mask"] + [0] * pad)
        offsets.append(x["offsets"] + [(0, 0)] * pad)
        gold.append(x["gold"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "offsets": offsets,
        "gold": gold,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="阶段一：BCE 二分类跨度预训练")
    ap.add_argument("--data", type=str, required=True, help="train_10000.jsonl 等高资源子集")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--encoder", type=str, default="roberta-base")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--max_span_width", type=int, default=12)
    ap.add_argument("--neg_per_gold", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", help="混合精度加速（RTX 4090 等）")
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers，0=主进程")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"data not found: {data_path}")

    samples = load_jsonl(data_path)
    if not samples:
        raise RuntimeError("empty dataset")

    tok = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
    pad_id = tok.pad_token_id or 0
    ds = BCESpanDataset(samples, tok, args.max_len)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpanEntityBCE(args.encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda") if args.fp16 and device.type == "cuda" else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batch = 0
        use_amp = args.fp16 and device.type == "cuda"
        pbar = tqdm(
            dl,
            desc=f"bce_ep{ep}",
            leave=True,
            dynamic_ncols=False,
            ncols=110,
            file=sys.stdout,
        )
        total_steps = len(dl) if len(dl) > 0 else 1
        for step_idx, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                hs = model(input_ids, attn)
            loss_all = []
            for bi in range(hs.size(0)):
                tok_emb = hs[bi]
                offsets = batch["offsets"][bi]
                valid_mask = [(a, b) not in SPECIAL_OFFSETS and b > a for (a, b) in offsets]
                gold = batch["gold"][bi]
                if not gold:
                    continue
                # gold may contain list items from json decode; normalize to tuple for hashing.
                pos_spans = [tuple(x) for x in gold]
                gold_sp = set(pos_spans)
                neg_spans = sample_negative_spans(
                    seq_len=tok_emb.size(0),
                    gold_spans=gold_sp,
                    max_width=args.max_span_width,
                    k=len(pos_spans) * args.neg_per_gold,
                    valid_mask=valid_mask,
                )
                spans = pos_spans + neg_spans
                y = [1.0] * len(pos_spans) + [0.0] * len(neg_spans)
                spans_t = torch.tensor(spans, dtype=torch.long, device=device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model.span_logits(tok_emb, spans_t)
                y_t = torch.tensor(y, dtype=torch.float32, device=device)
                loss_all.append(bce(logits.float(), y_t))
            if not loss_all:
                continue
            loss = torch.stack(loss_all).mean()
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
            n_batch += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            # 后台日志可见的稳定进度行（避免 tqdm 回写导致 tail 看不清）
            if step_idx == 1 or step_idx % 10 == 0 or step_idx == total_steps:
                pct = int(step_idx * 100 / max(1, total_steps))
                print(
                    f"[BCE][ep {ep}/{args.epochs}] step {step_idx}/{total_steps} ({pct}%) "
                    f"loss={loss.item():.4f}"
                )
        avg = total_loss / n_batch if n_batch else 0
        print(f"epoch {ep} loss={avg:.4f}")

    ckpt_path = out_dir / "encoder_span_proj.pt"
    torch.save({
        "encoder": model.encoder.state_dict(),
        "span_proj": model.span_proj.state_dict(),
    }, ckpt_path)
    (out_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] 阶段一完成 -> {ckpt_path}")


if __name__ == "__main__":
    main()
