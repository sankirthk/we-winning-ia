"""
Pydantic model for the knowledge base produced by KnowledgeBaseAgent.
Never shown to the user — injected into LiveAgent system context for deep Q&A.
Generic schema that works for any report type (financial, research, compliance, etc.)
"""

from pydantic import BaseModel, Field


class KnowledgeBase(BaseModel):
    document_title: str

    # Detailed explanation of each key finding — the "why" behind the headline
    deep_findings: list[str] = Field(default_factory=list)

    # Specific facts, numbers, or named entities worth knowing verbatim
    key_facts: list[str] = Field(default_factory=list)

    # Notable causes of failures, risks, or negative outcomes explained in detail
    risks_and_failures: list[str] = Field(default_factory=list)

    # What worked, why it worked, and the conditions that made it succeed
    successes_and_rationale: list[str] = Field(default_factory=list)

    # Domain-specific terminology defined in plain language
    definitions: dict[str, str] = Field(default_factory=dict)

    # Full verbatim extraction of the most analysis-dense sections (discussion, conclusion, etc.)
    expert_detail: str
