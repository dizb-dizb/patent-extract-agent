"""专利/论文实体标注的 Pydantic 模型定义（扁平实体，无 children）。"""
from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class NestedEntity(BaseModel):
    """论文术语实体（扁平）：text, label, start, end。"""

    text: str = Field(..., description="术语对应的原文片段")
    label: str = Field(
        ...,
        description="由术语词典生成的 label，如 term_0, term_1",
    )
    start: int = Field(..., ge=0, description="在原文中的起始字符索引（含）")
    end: int = Field(..., ge=0, description="在原文中的结束字符索引（不含）")


class PaperAnnotationResult(BaseModel):
    """论文术语标注结果（顶层实体列表）。"""

    entities: List[NestedEntity] = Field(
        default_factory=list,
        description="识别出的术语列表",
    )
