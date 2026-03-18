"""
Span-based NER trainer (supports nested/overlapping spans).

Input dataset jsonl (one per line):
  {"context": "...", "spans": [{"start":0,"end":2,"label":"X","text":"..", ...}, ...]}

Model:
  - Encoder: HuggingFace Transformer (default: hfl/chinese-roberta-wwm-ext)
  - Span representation: concat([h_start, h_end, mean_pool(span_tokens)])
  - Classifier over labels + NONE

Training:
  - For each sentence: use all gold spans + sampled negative spans (random)
  - Loss: cross entropy

Eval:
  - Enumerate candidate spans (up to max_span_width), score, keep label!=NONE and prob>=threshold
  - Metric: micro P/R/F1 on exact match of (char_start, char_end, label)

Outputs:
  - metrics.json (in output_dir)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from dataset_version import dataset_version

SPECIAL_OFFSETS = {(0, 0)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
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


def build_label_vocab(samples: list[dict]) -> list[str]:
    labs: set[str] = set()
    for s in samples:
        for sp in s.get("spans") or []:
            if isinstance(sp, dict):
                labs.add(str(sp.get("label") or "term"))
    # stable order
    return ["NONE"] + sorted(labs)


def _tokenize(tokenizer, text: str, max_len: int):
    return tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_len,
        padding=False,
    )


def char_span_to_token_span(offsets: list[tuple[int, int]], char_start: int, char_end: int) -> tuple[int, int] | None:
    """
    Map a char span [char_start, char_end) to token indices [ts, te) (inclusive-exclusive).
    We find tokens whose offsets overlap the span boundaries.
    """
    if char_end <= char_start:
        return None
    # ignore special tokens with (0,0)
    valid = [(i, a, b) for i, (a, b) in enumerate(offsets) if (a, b) not in SPECIAL_OFFSETS and b > a]
    if not valid:
        return None
    ts = None
    te = None
    for i, a, b in valid:
        if ts is None and a <= char_start < b:
            ts = i
        if a < char_end <= b:
            te = i + 1
            break
        if a >= char_end:
            break
    if ts is None or te is None or te <= ts:
        # fallback: exact match on boundaries
        for i, a, b in valid:
            if a == char_start:
                ts = i
            if b == char_end and ts is not None and i >= ts:
                te = i + 1
                break
    if ts is None or te is None or te <= ts:
        return None
    return (ts, te)


@dataclass
class EncodedSample:
    text: str
    input_ids: list[int]
    attention_mask: list[int]
    offsets: list[tuple[int, int]]
    gold: list[tuple[int, int, int]]  # token span + label_id
    gold_char: list[tuple[int, int, int]]  # char span + label_id


class SpanDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer, label_to_id: dict[str, int], max_len: int):
        self.items: list[EncodedSample] = []
        for s in samples:
            text = str(s.get("context") or "")
            enc = _tokenize(tokenizer, text, max_len=max_len)
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
            gold: list[tuple[int, int, int]] = []
            gold_char: list[tuple[int, int, int]] = []
            for sp in s.get("spans") or []:
                if not isinstance(sp, dict):
                    continue
                cs = int(sp.get("start", 0))
                ce = int(sp.get("end", 0))
                lab = str(sp.get("label") or "term")
                lid = label_to_id.get(lab, 0)
                ts_te = char_span_to_token_span(offsets, cs, ce)
                if ts_te is None:
                    continue
                ts, te = ts_te
                gold.append((ts, te, lid))
                gold_char.append((cs, ce, lid))
            self.items.append(
                EncodedSample(
                    text=text,
                    input_ids=input_ids,
                    attention_mask=attn,
                    offsets=offsets,
                    gold=gold,
                    gold_char=gold_char,
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> EncodedSample:
        return self.items[idx]


def collate(batch: list[EncodedSample], pad_id: int):
    max_len = max(len(x.input_ids) for x in batch)
    input_ids = []
    attn = []
    offsets = []
    gold = []
    gold_char = []
    texts = []
    for x in batch:
        pad = max_len - len(x.input_ids)
        input_ids.append(x.input_ids + [pad_id] * pad)
        attn.append(x.attention_mask + [0] * pad)
        offsets.append(x.offsets + [(0, 0)] * pad)
        gold.append(x.gold)
        gold_char.append(x.gold_char)
        texts.append(x.text)
    return {
        "texts": texts,
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "offsets": offsets,
        "gold": gold,
        "gold_char": gold_char,
    }


class SpanNER(nn.Module):
    def __init__(self, encoder_name: str, num_labels: int, hidden_dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(h * 3, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state  # (B, T, H)

    def span_logits(self, token_emb: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        """
        token_emb: (T, H)
        spans: (N, 2) token indices [start, end) in token space
        """
        h = token_emb
        starts = spans[:, 0]
        ends = spans[:, 1] - 1
        h_start = h[starts]  # (N, H)
        h_end = h[ends]      # (N, H)
        # mean pool within span (clamp for safety)
        pooled = []
        for i in range(spans.size(0)):
            s = int(spans[i, 0].item())
            e = int(spans[i, 1].item())
            pooled.append(h[s:e].mean(dim=0))
        h_mean = torch.stack(pooled, dim=0)
        feat = torch.cat([h_start, h_end, h_mean], dim=-1)
        feat = self.dropout(feat)
        return self.classifier(feat)  # (N, C)


def sample_negative_spans(
    seq_len: int,
    gold_spans: set[tuple[int, int]],
    max_width: int,
    k: int,
    valid_mask: list[bool],
) -> list[tuple[int, int]]:
    neg: list[tuple[int, int]] = []
    tries = 0
    while len(neg) < k and tries < k * 20:
        tries += 1
        s = random.randint(0, seq_len - 2)
        w = random.randint(1, max_width)
        e = min(seq_len, s + w)
        if e <= s + 0:
            continue
        if (s, e) in gold_spans:
            continue
        if not (valid_mask[s] and valid_mask[e - 1]):
            continue
        neg.append((s, e))
    return neg


def enumerate_candidate_spans(seq_len: int, max_width: int, valid_mask: list[bool], limit: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for s in range(seq_len):
        if not valid_mask[s]:
            continue
        for w in range(1, max_width + 1):
            e = s + w
            if e > seq_len:
                break
            if not valid_mask[e - 1]:
                continue
            spans.append((s, e))
            if len(spans) >= limit:
                return spans
    return spans


def spans_token_to_char(spans: list[tuple[int, int, int]], offsets: list[tuple[int, int]]) -> set[tuple[int, int, int]]:
    out: set[tuple[int, int, int]] = set()
    for ts, te, lid in spans:
        if te <= ts:
            continue
        cs = offsets[ts][0]
        ce = offsets[te - 1][1]
        if (cs, ce) in SPECIAL_OFFSETS or ce <= cs:
            continue
        out.add((cs, ce, lid))
    return out


def micro_prf(pred: list[set[tuple[int, int, int]]], gold: list[set[tuple[int, int, int]]]) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for p, g in zip(pred, gold):
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="train_spans_augmented.jsonl")
    ap.add_argument("--val", type=str, default="", help="Optional separate val file; if set, --train_ratio ignored")
    ap.add_argument("--encoder", type=str, default="roberta-base")
    ap.add_argument("--output_dir", type=str, default="artifacts/run_span_ner")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--max_span_width", type=int, default=12)
    ap.add_argument("--neg_per_gold", type=int, default=3)
    ap.add_argument("--eval_span_limit", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (4-8 for max GPU)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--multi_gpu", action="store_true", help="Use DataParallel for multi-GPU")
    ap.add_argument("--max_train_samples", type=int, default=0,
                    help="限制训练样本数 (0=不限制)，用于 10/100/1000 梯度实验")
    args = ap.parse_args()

    set_seed(args.seed)
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"missing data: {data_path}")

    samples = load_jsonl(data_path)
    if not samples:
        raise RuntimeError("empty dataset")
    if args.max_train_samples > 0:
        random.shuffle(samples)
        samples = samples[: args.max_train_samples]
        print(f"[data] 限制训练样本数: {len(samples)}")

    val_path = Path(args.val) if args.val else None
    if val_path and not val_path.is_absolute():
        val_path = Path(__file__).resolve().parent / val_path
    if val_path and val_path.exists():
        val_s = load_jsonl(val_path)
        train_s = samples
    else:
        random.shuffle(samples)
        n_train = int(len(samples) * args.train_ratio)
        train_s = samples[:n_train]
        val_s = samples[n_train:] if n_train < len(samples) else samples[-max(1, len(samples)//10):]

    labels = build_label_vocab(train_s + val_s)
    label_to_id = {l: i for i, l in enumerate(labels)}
    id_to_label = {i: l for l, i in label_to_id.items()}

    tok = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    train_ds = SpanDataset(train_s, tok, label_to_id, max_len=args.max_len)
    val_ds = SpanDataset(val_s, tok, label_to_id, max_len=args.max_len)

    # num_workers>0 预取数据，避免 GPU 等待；pin_memory 加速 CPU->GPU 传输
    kw = dict(batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, pad_id))
    if args.num_workers > 0:
        kw.update(num_workers=args.num_workers, pin_memory=True)
    elif device.type == "cuda":
        kw.update(pin_memory=True)
    dl_train = DataLoader(train_ds, **kw)
    dl_val = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=lambda b: collate(b, pad_id))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = SpanNER(args.encoder, num_labels=len(labels)).to(device)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    # training (DataParallel 包装后需通过 .module 访问自定义方法)
    _model = model.module if hasattr(model, "module") else model
    model.train()
    for ep in range(1, args.epochs + 1):
        pbar = tqdm(dl_train, desc=f"train ep{ep}", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            hs = model(input_ids, attn)  # (B,T,H)
            loss_all = []
            for bi in range(hs.size(0)):
                tok_emb = hs[bi]
                offsets = batch["offsets"][bi]
                valid_mask = [(a, b) not in SPECIAL_OFFSETS and b > a for (a, b) in offsets]
                gold = batch["gold"][bi]
                if not gold:
                    continue
                gold_sp = {(ts, te) for ts, te, _ in gold}
                pos_spans = [(ts, te) for ts, te, _ in gold]
                pos_labels = [lid for _, _, lid in gold]
                neg_spans = sample_negative_spans(
                    seq_len=tok_emb.size(0),
                    gold_spans=gold_sp,
                    max_width=args.max_span_width,
                    k=len(pos_spans) * args.neg_per_gold,
                    valid_mask=valid_mask,
                )
                spans = pos_spans + neg_spans
                y = pos_labels + [0] * len(neg_spans)  # 0 = NONE
                spans_t = torch.tensor(spans, dtype=torch.long, device=device)
                logits = _model.span_logits(tok_emb, spans_t)
                y_t = torch.tensor(y, dtype=torch.long, device=device)
                loss = nn.functional.cross_entropy(logits, y_t)
                loss_all.append(loss)
            if not loss_all:
                continue
            loss = torch.stack(loss_all).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        # eval
        model.eval()
        preds: list[set[tuple[int, int, int]]] = []
        golds: list[set[tuple[int, int, int]]] = []
        with torch.no_grad():
            for batch in tqdm(dl_val, desc=f"eval ep{ep}", leave=False):
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                out = model(input_ids, attn)
                hs = out[0] if isinstance(out, tuple) else out
                if hs.dim() == 3:
                    hs = hs[0]  # (T,H)
                offsets = batch["offsets"][0]
                valid_mask = [(a, b) not in SPECIAL_OFFSETS and b > a for (a, b) in offsets]
                cand = enumerate_candidate_spans(hs.size(0), args.max_span_width, valid_mask, limit=args.eval_span_limit)
                if not cand:
                    preds.append(set())
                    golds.append(set(batch["gold_char"][0]))
                    continue
                spans_t = torch.tensor(cand, dtype=torch.long, device=device)
                logits = _model.span_logits(hs, spans_t)
                prob = torch.softmax(logits, dim=-1)
                conf, lid = torch.max(prob, dim=-1)
                chosen: list[tuple[int, int, int]] = []
                for i in range(spans_t.size(0)):
                    lab_id = int(lid[i].item())
                    if lab_id == 0:
                        continue
                    if float(conf[i].item()) < args.threshold:
                        continue
                    ts = int(spans_t[i, 0].item())
                    te = int(spans_t[i, 1].item())
                    chosen.append((ts, te, lab_id))
                preds.append(spans_token_to_char(chosen, offsets))
                golds.append(set(batch["gold_char"][0]))

        prec, rec, f1 = micro_prf(preds, golds)
        ds_ver = dataset_version(data_path)
        metrics = {
            "name": "span_ner",
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "encoder": args.encoder,
            "data": str(data_path),
            "dataset_version": ds_ver,
            "epoch": ep,
            "num_labels": len(labels) - 1,
            "created_at": int(time.time()),
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] ep={ep} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
        model.train()

    # save model (DataParallel 需保存 .module)，得到实验数据后释放以省磁盘
    model_pt = out_dir / "model.pt"
    torch.save(_model.state_dict(), model_pt)
    (out_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    model_pt.unlink(missing_ok=True)
    print(f"[ok] saved to {out_dir} (已释放 model.pt，保留 metrics.json)")


if __name__ == "__main__":
    main()

