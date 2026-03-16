"""
BiLSTM-CRF BIO sequence labeling baseline (B1).

Randomly-initialized word embeddings + BiLSTM + CRF (pytorch-crf).
No pretrained model — intentional weakest baseline to show the gap between
traditional NLP and large pretrained encoders.

Input : span JSONL  {"context": "...", "spans": [...]}
Output: artifacts/run_bilstm_crf/{dataset}/metrics.json  (flat_f1, P/R/F1)

Usage:
  python train_bilstm_crf.py --data data/benchmarks/fewnerd/train.jsonl
  python train_bilstm_crf.py --data data/benchmarks/genia/train.jsonl --output_dir artifacts/run_bilstm_crf/genia
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from torchcrf import CRF
except ImportError:
    raise ImportError("pytorch-crf not installed. Run: pip install pytorch-crf")

from dataset_version import dataset_version

SPECIAL_TOKENS = ["<PAD>", "<UNK>"]
PAD_IDX = 0
UNK_IDX = 1


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


def simple_tokenize(text: str) -> list[str]:
    """Whitespace tokenization; returns list of (token, char_start, char_end)."""
    tokens: list[tuple[str, int, int]] = []
    start = 0
    for part in text.split(" "):
        if part:
            tokens.append((part, start, start + len(part)))
        start += len(part) + 1
    return tokens


def spans_to_bio(
    context: str, spans: list[dict]
) -> tuple[list[str], list[tuple[int, int]]]:
    """Convert char-level spans to word-level BIO tags via whitespace tokenization."""
    tokens_info = simple_tokenize(context)
    if not tokens_info:
        return [], []

    n = len(context)
    char_labels = ["O"] * n
    norm: list[tuple[int, int, str]] = []
    for sp in spans or []:
        if not isinstance(sp, dict):
            continue
        s = int(sp.get("start", 0))
        e = int(sp.get("end", 0))
        if s < 0 or e <= s or e > n:
            continue
        lab = str(sp.get("label") or "TERM").strip() or "TERM"
        norm.append((s, e, lab))
    norm.sort(key=lambda x: (x[1] - x[0]), reverse=True)
    for s, e, lab in norm:
        if any(char_labels[i] != "O" for i in range(s, e)):
            continue
        char_labels[s] = f"B-{lab}"
        for i in range(s + 1, e):
            char_labels[i] = f"I-{lab}"

    word_tags: list[str] = []
    offsets: list[tuple[int, int]] = []
    for tok, ts, te in tokens_info:
        tag = char_labels[ts] if ts < len(char_labels) else "O"
        word_tags.append(tag)
        offsets.append((ts, te))
    return word_tags, offsets


def build_vocab(samples: list[dict]) -> tuple[dict[str, int], list[str]]:
    """Build word vocabulary from samples."""
    counter: dict[str, int] = {}
    for s in samples:
        ctx = str(s.get("context") or "")
        for tok, _, _ in simple_tokenize(ctx):
            counter[tok.lower()] = counter.get(tok.lower(), 0) + 1
    vocab = SPECIAL_TOKENS + sorted(counter.keys())
    word_to_id = {w: i for i, w in enumerate(vocab)}
    return word_to_id, vocab


def build_label_vocab(samples: list[dict]) -> list[str]:
    labs: set[str] = set()
    for s in samples:
        for sp in s.get("spans") or []:
            if isinstance(sp, dict):
                lab = str(sp.get("label") or "TERM")
                labs.add(f"B-{lab}")
                labs.add(f"I-{lab}")
    return ["O"] + sorted(labs)


class BioCRFDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        word_to_id: dict[str, int],
        label_to_id: dict[str, int],
        max_len: int,
    ):
        self.items: list[dict] = []
        for s in samples:
            ctx = str(s.get("context") or "")
            if not ctx.strip():
                continue
            tokens_info = simple_tokenize(ctx)
            if not tokens_info:
                continue
            tags, offsets = spans_to_bio(ctx, s.get("spans") or [])
            tokens = [t for t, _, _ in tokens_info]

            # truncate
            tokens = tokens[:max_len]
            tags = tags[:max_len]
            offsets = offsets[:max_len]

            ids = [word_to_id.get(tok.lower(), UNK_IDX) for tok in tokens]
            label_ids = [label_to_id.get(tag, label_to_id["O"]) for tag in tags]
            self.items.append({
                "input_ids": ids,
                "labels": label_ids,
                "offsets": offsets,
                "context": ctx,
                "spans": s.get("spans") or [],
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def collate_fn(batch: list[dict]) -> dict:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, masks = [], [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [PAD_IDX] * pad)
        labels.append(x["labels"] + [0] * pad)
        masks.append([1] * len(x["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "mask": torch.tensor(masks, dtype=torch.bool),
        "_meta": [{"offsets": x["offsets"], "context": x["context"], "spans": x["spans"]} for x in batch],
    }


class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embed_dim: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[list[int]]]:
        emb = self.dropout(self.embedding(input_ids))
        lstm_out, _ = self.lstm(emb)
        lstm_out = self.dropout(lstm_out)
        emissions = self.linear(lstm_out)

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
            return loss, []
        else:
            preds = self.crf.decode(emissions, mask=mask)
            return torch.tensor(0.0), preds


def extract_spans_from_bio(
    tags: list[str], offsets: list[tuple[int, int]]
) -> set[tuple[int, int, str]]:
    spans: set[tuple[int, int, str]] = set()
    i = 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            lab = tags[i][2:]
            s = offsets[i][0] if i < len(offsets) else 0
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{lab}":
                j += 1
            e = offsets[j - 1][1] if (j - 1) < len(offsets) else s + 1
            spans.add((s, e, lab))
            i = j
        else:
            i += 1
    return spans


def micro_prf(
    pred: list[set[tuple[int, int, str]]],
    gold: list[set[tuple[int, int, str]]],
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
    ap = argparse.ArgumentParser(description="BiLSTM-CRF BIO baseline (B1)")
    ap.add_argument("--data", type=str, default="data/benchmarks/fewnerd/train.jsonl")
    ap.add_argument("--val", type=str, default="", help="Optional separate val file")
    ap.add_argument("--output_dir", type=str, default="artifacts/run_bilstm_crf/fewnerd")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--embed_dim", type=int, default=100)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    root = Path(__file__).resolve().parent
    data_path = Path(args.data) if Path(args.data).is_absolute() else root / args.data
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
        val_s = samples[n_train:] if n_train < len(samples) else samples[-max(1, len(samples) // 10):]

    word_to_id, vocab = build_vocab(train_s)
    label_list = build_label_vocab(train_s)
    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for l, i in label_to_id.items()}

    train_ds = BioCRFDataset(train_s, word_to_id, label_to_id, args.max_len)
    val_ds = BioCRFDataset(val_s, word_to_id, label_to_id, args.max_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMCRF(
        vocab_size=len(vocab),
        num_labels=len(label_list),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)

    dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dl_val = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_ver = dataset_version(data_path)

    best_f1 = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(dl_train, desc=f"ep{ep}", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels_t = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            opt.zero_grad()
            loss, _ = model(input_ids, mask, labels=labels_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        preds_list: list[set[tuple[int, int, str]]] = []
        golds_list: list[set[tuple[int, int, str]]] = []
        with torch.no_grad():
            for batch in dl_val:
                input_ids = batch["input_ids"].to(device)
                mask = batch["mask"].to(device)
                _, pred_seqs = model(input_ids, mask)
                meta = batch["_meta"]
                for i, pred_ids in enumerate(pred_seqs):
                    offsets = meta[i]["offsets"]
                    ctx = meta[i]["context"]
                    gold_spans_raw = meta[i]["spans"]
                    pred_tags = [id_to_label.get(pid, "O") for pid in pred_ids[:len(offsets)]]
                    pred_spans = extract_spans_from_bio(pred_tags, offsets)
                    # gold spans from char-level
                    gold_tags, gold_offsets = spans_to_bio(ctx, gold_spans_raw)
                    gold_spans = extract_spans_from_bio(gold_tags, gold_offsets)
                    preds_list.append(pred_spans)
                    golds_list.append(gold_spans)

        prec, rec, f1 = micro_prf(preds_list, golds_list)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), out_dir / "model.pt")

        metrics = {
            "name": "bilstm_crf",
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "flat_f1": f1,
            "epoch": ep,
            "data": str(data_path),
            "dataset_version": ds_ver,
            "created_at": int(time.time()),
        }
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[eval] ep={ep} loss={total_loss:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")

    print(f"[ok] best_f1={best_f1:.4f}  saved to {out_dir}")


if __name__ == "__main__":
    main()
