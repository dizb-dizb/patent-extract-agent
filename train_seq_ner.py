"""
RoBERTa/BERT BIO sequence labeling baseline.

Input: train_spans*.jsonl (context + spans)
Converts to token-level BIO, trains token classification head.
Output: metrics.json (P/R/F1)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from dataset_version import dataset_version

SPECIAL_OFFSETS = {(0, 0)}


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


def spans_to_bio_labels(context: str, spans: list[dict]) -> list[str]:
    n = len(context or "")
    labels = ["O"] * n
    norm_spans: list[tuple[int, int, str]] = []
    for e in spans or []:
        if not isinstance(e, dict):
            continue
        start = int(e.get("start", 0))
        end = int(e.get("end", 0))
        if start < 0 or end <= start or end > n:
            continue
        lab = str(e.get("label", "TERM")).strip() or "TERM"
        norm_spans.append((start, end, lab))
    norm_spans.sort(key=lambda x: (x[1] - x[0]), reverse=True)
    for start, end, lab in norm_spans:
        if any(labels[i] != "O" for i in range(start, end)):
            continue
        labels[start] = f"B-{lab}"
        for i in range(start + 1, end):
            labels[i] = f"I-{lab}"
    return labels


def build_label_vocab(samples: list[dict]) -> list[str]:
    labs: set[str] = set()
    for s in samples:
        for sp in s.get("spans") or []:
            if isinstance(sp, dict):
                lab = str(sp.get("label") or "TERM")
                labs.add(f"B-{lab}")
                labs.add(f"I-{lab}")
    return ["O"] + sorted(labs)


class BioDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer, label_to_id: dict[str, int], max_len: int):
        self.items: list[dict] = []
        for s in samples:
            ctx = str(s.get("context") or "")
            if not ctx:
                continue
            enc = tokenizer(
                ctx,
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_len,
                padding=False,
            )
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
            char_labels = spans_to_bio_labels(ctx, s.get("spans") or [])
            token_labels = []
            for a, b in offsets:
                if (a, b) in SPECIAL_OFFSETS or b <= a:
                    token_labels.append(-100)
                else:
                    token_labels.append(label_to_id.get(char_labels[a], label_to_id["O"]))
            self.items.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": token_labels,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def collate(batch: list[dict], pad_id: int, label_pad: int = -100) -> dict:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attn = []
    labels = []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_id] * pad)
        attn.append(x["attention_mask"] + [0] * pad)
        labels.append(x["labels"] + [label_pad] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def extract_spans_from_bio(labels: list[str], offsets: list[tuple[int, int]]) -> set[tuple[int, int, str]]:
    spans: set[tuple[int, int, str]] = set()
    i = 0
    while i < len(labels):
        if labels[i].startswith("B-"):
            lab = labels[i][2:]
            start = offsets[i][0] if i < len(offsets) else 0
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{lab}":
                j += 1
            end = offsets[j - 1][1] if j <= len(offsets) else start + 1
            spans.add((start, end, lab))
            i = j
        else:
            i += 1
    return spans


def micro_prf(
    pred: list[set[tuple[int, int, str]]], gold: list[set[tuple[int, int, str]]]
) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for p, g in zip(pred, gold):
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="train_spans_augmented.jsonl")
    ap.add_argument("--encoder", type=str, default="hfl/chinese-roberta-wwm-ext")
    ap.add_argument("--output_dir", type=str, default="artifacts/run_seq_ner")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--multi_gpu", action="store_true", help="Use DataParallel for multi-GPU")
    args = ap.parse_args()

    set_seed(args.seed)
    root = Path(__file__).resolve().parent
    data_path = root / args.data if not Path(args.data).is_absolute() else Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"missing data: {data_path}")

    samples = load_jsonl(data_path)
    if not samples:
        raise RuntimeError("empty dataset")

    random.shuffle(samples)
    n_train = int(len(samples) * args.train_ratio)
    train_s = samples[:n_train]
    val_s = samples[n_train:] if n_train < len(samples) else samples[-max(1, len(samples) // 10) :]

    labels = build_label_vocab(samples)
    label_to_id = {l: i for i, l in enumerate(labels)}
    id_to_label = {i: l for l, i in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
    pad_id = tokenizer.pad_token_id or 0

    train_ds = BioDataset(train_s, tokenizer, label_to_id, args.max_len)
    val_ds = BioDataset(val_s, tokenizer, label_to_id, args.max_len)

    model = AutoModelForTokenClassification.from_pretrained(
        args.encoder,
        num_labels=len(labels),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_ver = dataset_version(data_path)

    from torch.utils.data import DataLoader
    dl_train = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
    )
    dl_val = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=lambda b: collate(b, pad_id))

    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in tqdm(dl_train, desc=f"ep{ep}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels_t = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels_t)
            opt.zero_grad()
            loss = out.loss if out.loss.dim() == 0 else out.loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        preds: list[set[tuple[int, int, str]]] = []
        golds: list[set[tuple[int, int, str]]] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dl_val):
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                out = model(input_ids=input_ids, attention_mask=attn)
                logits = out.logits[0]
                pred_ids = logits.argmax(dim=-1).tolist()
                # get offsets from tokenizer for this sample
                idx = min(batch_idx, len(val_s) - 1)
                ctx = val_s[idx].get("context") or ""
                enc = tokenizer(ctx, return_offsets_mapping=True, truncation=True, max_length=args.max_len)
                offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
                pred_labels = [id_to_label.get(pid, "O") for pid in pred_ids[: len(offsets)]]
                pred_spans = extract_spans_from_bio(pred_labels, offsets)
                gold_char = spans_to_bio_labels(ctx, val_s[idx].get("spans") or [])
                gold_spans: set[tuple[int, int, str]] = set()
                i = 0
                while i < len(gold_char):
                    if gold_char[i].startswith("B-"):
                        lab = gold_char[i][2:]
                        start = i
                        j = i + 1
                        while j < len(gold_char) and gold_char[j] == f"I-{lab}":
                            j += 1
                        end = j
                        gold_spans.add((start, end, lab))
                        i = j
                    else:
                        i += 1
                preds.append(pred_spans)
                golds.append(gold_spans)

        prec, rec, f1 = micro_prf(preds, golds)
        metrics = {
            "name": "seq_ner_roberta",
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "encoder": args.encoder,
            "data": str(data_path),
            "dataset_version": ds_ver,
            "epoch": ep,
            "created_at": int(time.time()),
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] ep={ep} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[ok] saved to {out_dir}")


if __name__ == "__main__":
    main()
