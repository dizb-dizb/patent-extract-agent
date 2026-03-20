"""
阶段二：冻结底座的度量微调 (Frozen Metric Fine-tuning)

冻结 RoBERTa + Span 提议层，仅训练投影模块 (Projector)。
投影器将 span 嵌入映射到度量空间，配合原型网络 + SCL 实现同类聚拢、异类推远。
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from transformers import AutoModel

MetricType = Literal["cosine", "euclidean"]


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    SCL: 同类聚拢、异类推远。
    embeddings: (N, D), labels: (N,) in [0..C-1]
    """
    n = embeddings.size(0)
    if n < 2:
        return torch.tensor(0.0, device=embeddings.device)
    emb_n = nn.functional.normalize(embeddings, dim=-1)
    sim = torch.mm(emb_n, emb_n.t()) / temperature  # (N, N)
    mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_pos = mask_pos.float()
    mask_pos = mask_pos - torch.eye(n, device=embeddings.device)
    mask_pos = mask_pos.clamp(min=0)
    exp_sim = torch.exp(sim)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    mean_log_prob = (mask_pos * log_prob).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
    return -mean_log_prob.mean()


class ProjectorSpanProto(nn.Module):
    """
    冻结 encoder + span_proj，仅训练 projector。
    投影器将 span 嵌入映射到度量空间，用于原型网络分类。
    """

    def __init__(
        self,
        encoder_name: str,
        n_classes: int = 10,
        projector_dim: int = 0,
        metric: MetricType = "cosine",
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        h = self.encoder.config.hidden_size
        self.span_proj = nn.Sequential(
            nn.Linear(h * 3, h),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
        )
        proj_dim = projector_dim or h
        self.projector = nn.Sequential(
            nn.Linear(h, proj_dim),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
        )
        self.proj_dim = proj_dim
        self.n_classes = n_classes
        self.metric = metric
        self._frozen = False

    def freeze_encoder_and_span_proj(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.span_proj.parameters():
            p.requires_grad = False
        self._frozen = True

    def load_stage1_ckpt(self, ckpt_path: str | None) -> None:
        if not ckpt_path:
            return
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "encoder" in ckpt:
            self.encoder.load_state_dict(ckpt["encoder"], strict=True)
        if "span_proj" in ckpt:
            self.span_proj.load_state_dict(ckpt["span_proj"], strict=True)
        self.freeze_encoder_and_span_proj()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def span_embedding(self, token_emb: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
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
        return self.projector(z)

    def compute_prototypes(
        self,
        span_emb: torch.Tensor,
        labels: torch.Tensor,
        n_class: int | None = None,
    ) -> torch.Tensor:
        n_class = n_class or self.n_classes
        protos = []
        for c in range(n_class):
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
        if self.metric == "cosine":
            qn = nn.functional.normalize(query_emb, dim=-1)
            pn = nn.functional.normalize(prototypes, dim=-1)
            sim = torch.mm(qn, pn.t())
            none_logit = torch.full((query_emb.size(0), 1), 0.5, device=query_emb.device, dtype=query_emb.dtype)
        else:
            q = query_emb.unsqueeze(1)
            p = prototypes.unsqueeze(0)
            dist = (q - p).pow(2).sum(dim=-1).sqrt()
            sim = -dist
            none_logit = torch.full((query_emb.size(0), 1), -1.0, device=query_emb.device, dtype=query_emb.dtype)
        return torch.cat([sim, none_logit], dim=-1)
