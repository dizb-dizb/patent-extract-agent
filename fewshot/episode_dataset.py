"""
N-way K-shot episodic dataset for prototypical span NER.

Builds support/query sets from span-annotated samples.
Supports nested/overlapping spans.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def build_label_to_spans(samples: list[dict]) -> dict[str, list[tuple[str, int, int, str]]]:
    """
    label -> [(context, char_start, char_end, text), ...]
    """
    lab_to_items: dict[str, list[tuple[str, int, int, str]]] = {}
    for s in samples:
        ctx = str(s.get("context") or "")
        if not ctx:
            continue
        for sp in s.get("spans") or []:
            if not isinstance(sp, dict):
                continue
            lab = str(sp.get("label") or "term").strip()
            if not lab:
                continue
            cs = int(sp.get("start", 0))
            ce = int(sp.get("end", 0))
            text = str(sp.get("text", ""))
            if ce <= cs:
                continue
            if lab not in lab_to_items:
                lab_to_items[lab] = []
            lab_to_items[lab].append((ctx, cs, ce, text))
    return lab_to_items


def sample_negative_spans(
    text: str,
    gold_set: set[tuple[int, int]],
    max_width: int,
    k: int,
    min_len: int = 1,
) -> list[tuple[int, int]]:
    """Sample k negative (non-entity) spans from text."""
    n = len(text)
    if n < 2:
        return []
    neg: list[tuple[int, int]] = []
    tries = 0
    while len(neg) < k and tries < k * 30:
        tries += 1
        s = random.randint(0, max(0, n - 2))
        w = random.randint(min_len, max_width)
        e = min(n, s + w)
        if e <= s:
            continue
        if (s, e) in gold_set:
            continue
        if (s, e) in neg:
            continue
        neg.append((s, e))
    return neg


@dataclass
class Episode:
    """Single N-way K-shot episode."""

    support_contexts: list[str] = field(default_factory=list)
    support_spans: list[tuple[int, int, int]] = field(default_factory=list)  # (ctx_idx, char_s, char_e)
    support_labels: list[int] = field(default_factory=list)  # label_id in [0..N-1]
    query_contexts: list[str] = field(default_factory=list)
    query_spans: list[tuple[int, int, int]] = field(default_factory=list)
    query_labels: list[int] = field(default_factory=list)  # N for NONE/negative
    label_names: list[str] = field(default_factory=list)  # N class names
    neg_in_query: int = 0  # count of negatives in query_labels (label_id=N)


class EpisodicSpanDataset:
    """
    Builds N-way K-shot episodes from span-annotated jsonl.
    """

    def __init__(
        self,
        samples: list[dict],
        n_way: int = 5,
        k_shot: int = 5,
        query_per_class: int = 5,
        neg_ratio: float = 0.3,
        max_span_width: int = 12,
        max_episodes: int = 1000,
        seed: int = 42,
        train_labels: list[str] | None = None,
        test_labels: list[str] | None = None,
    ):
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_per_class = query_per_class
        self.neg_ratio = neg_ratio
        self.max_span_width = max_span_width
        self.max_episodes = max_episodes
        self.rng = random.Random(seed)
        self.train_labels = train_labels
        self.test_labels = test_labels

        lab_to_spans = build_label_to_spans(samples)
        self.label_names = sorted(lab_to_spans.keys())
        self.label_to_id = {l: i for i, l in enumerate(self.label_names)}
        self.label_to_spans = lab_to_spans

        # Meta-train/meta-test: restrict labels. train_labels for training, test_labels for eval.
        if train_labels is not None:
            pool = [l for l in train_labels if l in lab_to_spans and len(lab_to_spans[l]) >= k_shot]
        else:
            pool = [l for l in self.label_names if len(lab_to_spans[l]) >= k_shot]
        self.valid_labels = pool if pool else self.label_names

    def __len__(self) -> int:
        return self.max_episodes

    def __iter__(self) -> "EpisodicSpanDataset":
        return self

    def __next__(self) -> Episode:
        return self.sample_episode()

    def sample_episode(self, use_test_labels: bool = False) -> Episode:
        # Sample N labels (from test_labels if use_test_labels, else valid_labels)
        pool = self.test_labels if (use_test_labels and self.test_labels) else self.valid_labels
        if not pool:
            pool = self.valid_labels
        if len(pool) < self.n_way:
            chosen = list(pool)
        else:
            chosen = self.rng.sample(pool, self.n_way)
        label_ids = [self.label_to_id[l] for l in chosen]

        support_ctx: list[str] = []
        support_spans: list[tuple[int, int, int]] = []
        support_labels: list[int] = []

        for lab in chosen:
            items = self.label_to_spans[lab]
            if len(items) < self.k_shot:
                picked = items
            else:
                picked = self.rng.sample(items, self.k_shot)
            for ctx, cs, ce, _ in picked:
                idx = len(support_ctx)
                if ctx not in support_ctx:
                    support_ctx.append(ctx)
                    idx = len(support_ctx) - 1
                else:
                    idx = support_ctx.index(ctx)
                support_spans.append((idx, cs, ce))
                support_labels.append(self.label_to_id[lab])

        # Unified contexts list for episode (support + query)
        all_ctx: list[str] = []
        ctx_to_idx: dict[str, int] = {}
        support_spans_fixed: list[tuple[int, int, int]] = []
        support_labels_fixed: list[int] = []

        def get_idx(c: str) -> int:
            if c not in ctx_to_idx:
                ctx_to_idx[c] = len(all_ctx)
                all_ctx.append(c)
            return ctx_to_idx[c]

        for (idx, cs, ce), lab in zip(support_spans, support_labels):
            ctx = support_ctx[idx] if idx < len(support_ctx) else ""
            support_spans_fixed.append((get_idx(ctx), cs, ce))
            support_labels_fixed.append(lab)

        support_set = {(all_ctx[s[0]], s[1], s[2]) for s in support_spans_fixed}
        gold_set_per_ctx: dict[str, set[tuple[int, int]]] = {}

        # Query: sample from same N classes
        query_spans: list[tuple[int, int, int]] = []
        query_labels: list[int] = []
        n_query_pos = self.n_way * self.query_per_class
        n_neg = max(1, int(n_query_pos * self.neg_ratio))

        for lab in chosen:
            items = self.label_to_spans[lab]
            if not items:
                continue
            need = self.query_per_class
            candidates = [x for x in items if (x[0], x[1], x[2]) not in support_set]
            if not candidates:
                candidates = items
            if len(candidates) < need:
                picked = candidates
            else:
                picked = self.rng.sample(candidates, need)
            for ctx, cs, ce, _ in picked:
                if ctx not in gold_set_per_ctx:
                    gold_set_per_ctx[ctx] = set()
                gold_set_per_ctx[ctx].add((cs, ce))
                query_spans.append((get_idx(ctx), cs, ce))
                query_labels.append(self.label_to_id[lab])

        # Add negative spans
        ctx_list = list(gold_set_per_ctx.keys()) or (all_ctx[:1] if all_ctx else [])
        for _ in range(n_neg):
            if not ctx_list:
                break
            ctx = self.rng.choice(ctx_list)
            gold_set = gold_set_per_ctx.get(ctx, set())
            negs = sample_negative_spans(ctx, gold_set, self.max_span_width, 1)
            if negs:
                cs, ce = negs[0]
                query_spans.append((get_idx(ctx), cs, ce))
                query_labels.append(self.n_way)

        return Episode(
            support_contexts=all_ctx,
            support_spans=support_spans_fixed,
            support_labels=support_labels_fixed,
            query_contexts=all_ctx,
            query_spans=query_spans,
            query_labels=query_labels,
            label_names=chosen,
            neg_in_query=sum(1 for q in query_labels if q == self.n_way),
        )
