"""
Pydantic models for the manifest produced by ParserAgent.
Imported by ParserAgent for validation and NarrativeScriptAgent for typed access.
"""

from typing import Literal

from pydantic import BaseModel, Field


class KeySection(BaseModel):
    id: int
    heading: str
    summary: str
    key_stats: list[str] = Field(default_factory=list)
    page: int


class Manifest(BaseModel):
    title: str
    type: Literal["corporate", "financial", "research", "project", "other"]
    total_pages: int
    key_sections: list[KeySection] = Field(min_length=1)
    overall_summary: str
    sentiment: Literal["positive", "cautious", "negative", "neutral"]
    sentiment_reason: str = ""  # why this sentiment — used in narration prompt
