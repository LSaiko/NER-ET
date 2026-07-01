"""Pydantic v2 request/response models."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class NERRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str = Field(min_length=10, max_length=50_000,
                      description="Document text to extract entities from")


class EntityItem(BaseModel):
    label: Literal["PERSON", "ORG", "LOC", "DATE"]
    text: str
    score: float = Field(ge=0.0, le=1.0)
    start: int
    end: int


class CanonicalEntity(BaseModel):
    label: Literal["PERSON", "ORG", "LOC", "DATE"]
    canonical_text: str
    variants: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class EntityCounts(BaseModel):
    PERSON: int = 0
    ORG: int = 0
    LOC: int = 0
    DATE: int = 0


class ResolvedEntities(BaseModel):
    canonical_entities: list[CanonicalEntity]
    entity_counts: EntityCounts
    resolution_notes: str


class NERResponse(BaseModel):
    entities: list[EntityItem]          # raw BERT + spaCy output
    resolved: ResolvedEntities | None   # Claude output; None on fallback
    claude_invoked: bool
    model: str = "dslim/bert-base-NER"
    char_count: int
    entity_count: int
