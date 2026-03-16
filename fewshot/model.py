"""
Prototypical span NER: encoder + span representation + prototype head.
Supports cosine / euclidean distance.

Encoder types:
  - "transformer" (default): HuggingFace AutoModel (BERT/RoBERTa)
  - "bilstm": randomly-initialized Embedding + BiLSTM (B3/B5 baselines)
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from transformers import AutoModel

MetricType = Literal["cosine", "euclidean"]
EncoderType = Literal["transformer", "bilstm"]

SPECIAL_OFFSETS = {(0, 0)}


def char_span_to_token_span(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int] | None:
    if char_end <= char_start:
        return None
    valid = [(i, a, b) for i, (a, b) in enumerate(offsets) if (a, b) not in SPECIAL_OFFSETS and b > a]
    if not valid:
        return None
    ts = te = None
    for i, a, b in valid:
        if ts is None and a <= char_start < b:
            ts = i
        if a < char_end <= b:
            te = i + 1
            break
        if a >= char_end:
            break
    if ts is None or te is None or te <= ts:
        for i, a, b in valid:
            if a == char_start:
                ts = i
            if b == char_end and ts is not None and i >= ts:
                te = i + 1
                break
    if ts is None or te is None or te <= ts:
        return None
    return (ts, te)


class _BiLSTMEncoder(nn.Module):
    """Randomly-initialized embedding + BiLSTM encoder for B3/B5 ablations."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = self.drop(self.embedding(input_ids))
        lengths = attention_mask.sum(dim=-1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # pad to original length if shorter due to packing
        T = input_ids.size(1)
        if out.size(1) < T:
            pad = torch.zeros(out.size(0), T - out.size(1), out.size(2), device=out.device)
            out = torch.cat([out, pad], dim=1)
        return out


class PrototypicalSpanNER(nn.Module):
    """
    Span encoder + prototype head.
    - Encoder: RoBERTa/BERT (encoder_type="transformer") or BiLSTM (encoder_type="bilstm")
    - Span repr: [h_start, h_end, mean_pool]
    - Prototypes: mean of support span vectors per class
    - Metric: cosine or euclidean (negative distance -> logits)
    """

    def __init__(
        self,
        encoder_name: str,
        n_classes: int = 10,
        hidden_dropout: float = 0.1,
        metric: MetricType = "cosine",
        encoder_type: EncoderType = "transformer",
        bilstm_vocab_size: int = 30000,
        bilstm_embed_dim: int = 100,
        bilstm_hidden: int = 256,
        bilstm_layers: int = 2,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.n_classes = n_classes
        self.metric = metric
        self.dropout = nn.Dropout(hidden_dropout)

        if encoder_type == "bilstm":
            self.encoder = _BiLSTMEncoder(
                vocab_size=bilstm_vocab_size,
                embed_dim=bilstm_embed_dim,
                hidden_dim=bilstm_hidden,
                num_layers=bilstm_layers,
                dropout=hidden_dropout,
            )
            h = bilstm_hidden
        else:
            self.encoder = AutoModel.from_pretrained(encoder_name)
            h = self.encoder.config.hidden_size

        self.hidden_size = h
        self.span_proj = nn.Sequential(
            nn.Linear(h * 3, h),
            nn.GELU(),
            self.dropout,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "bilstm":
            return self.encoder(input_ids, attention_mask)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

    def span_embedding(self, token_emb: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        """
        token_emb: (T, H)
        spans: (N, 2) token indices [start, end)
        returns: (N, H) projected span vectors
        """
        h = token_emb
        starts = spans[:, 0]
        ends = spans[:, 1] - 1
        h_start = h[starts]
        h_end = h[ends]
        pooled = []
        for i in range(spans.size(0)):
            s = int(spans[i, 0].item())
            e = int(spans[i, 1].item())
            pooled.append(h[s:e].mean(dim=0))
        h_mean = torch.stack(pooled, dim=0)
        feat = torch.cat([h_start, h_end, h_mean], dim=-1)
        return self.span_proj(feat)

    def compute_prototypes(
        self, span_emb: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        span_emb: (M, H)
        labels: (M,) in [0..n_classes-1], NONE excluded
        returns: (n_classes, H)
        """
        protos = []
        for c in range(self.n_classes):
            mask = labels == c
            if mask.any():
                protos.append(span_emb[mask].mean(dim=0))
            else:
                protos.append(span_emb.mean(dim=0))
        return torch.stack(protos, dim=0)

    def compute_logits(
        self,
        query_emb: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        query_emb: (Q, H)
        prototypes: (N, H)
        returns: (Q, N+1) logits, last dim is NONE.
        NONE wins when query is far from all prototypes (max sim < threshold).
        """
        if self.metric == "cosine":
            qn = nn.functional.normalize(query_emb, dim=-1)
            pn = nn.functional.normalize(prototypes, dim=-1)
            sim = torch.mm(qn, pn.t())
            class_logits = sim
            # NONE when max sim < 0.5 (far from all prototypes)
            none_logit = torch.full((query_emb.size(0), 1), 0.5, device=query_emb.device, dtype=query_emb.dtype)
        else:
            q = query_emb.unsqueeze(1)
            p = prototypes.unsqueeze(0)
            dist = (q - p).pow(2).sum(dim=-1).sqrt()
            class_logits = -dist
            # NONE when best class score (-min_dist) < -1.0 (far from all prototypes)
            none_logit = torch.full((query_emb.size(0), 1), -1.0, device=query_emb.device, dtype=query_emb.dtype)
        return torch.cat([class_logits, none_logit], dim=-1)
