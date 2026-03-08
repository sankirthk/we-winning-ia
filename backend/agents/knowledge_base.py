"""
KnowledgeBaseAgent
IN:  session["file_path"]      — local path to the uploaded PDF
OUT: session["knowledge_base"] — deep knowledge dict for LiveAgent system context

Runs in parallel with ParserAgent via ParallelAgent in pipeline.py.
Output is never shown to the user — it is injected into the LiveAgent system prompt
so it can answer detailed follow-up questions with expert-level accuracy.
Model: gemini-3.1-pro-preview via Vertex AI
"""

import json
import os
import re
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types
from pydantic import ValidationError

from models.knowledge_base import KnowledgeBase
from tools.gemini import build_client, GEMINI_MODEL
from tools.job_store import update_job

PROMPT = """
You are an expert analyst building a knowledge base from this document.
Your output will be used by a live AI assistant to answer detailed user questions — so prioritize depth, precision, and completeness over brevity.

Extract the following as JSON:

{
  "document_title": "<title of the document>",
  "deep_findings": [
    "<detailed explanation of a key finding — include the mechanism, context, and significance. Not just what happened, but why.>"
  ],
  "key_facts": [
    "<specific fact, number, date, name, or verbatim claim worth knowing exactly>"
  ],
  "risks_and_failures": [
    "<detailed explanation of a failure, risk, problem, or negative outcome — include root cause and contributing factors>"
  ],
  "successes_and_rationale": [
    "<what worked, why it worked, and what conditions enabled it>"
  ],
  "definitions": {
    "<term>": "<plain-language definition as used in this document>"
  },
  "expert_detail": "<a long, dense paragraph extracting the most analysis-heavy content from the discussion, conclusion, or methodology sections — write as if briefing a domain expert>"
}

Rules:
- Be exhaustive on deep_findings and key_facts — include everything a user might ask about
- definitions should cover any jargon, acronyms, or domain-specific terms used in the document
- expert_detail should be several paragraphs if needed — do not summarize, extract
- Return ONLY valid JSON. No markdown fences, no explanation.
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


class KnowledgeBaseAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        job_id = ctx.session.state["job_id"]
        file_path = ctx.session.state["file_path"]

        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        client = build_client()

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                PROMPT,
            ],
        )

        raw = _extract_json(response.text)

        try:
            kb = KnowledgeBase.model_validate(raw)
        except ValidationError as e:
            raise ValueError(f"KnowledgeBaseAgent: validation failed:\n{e}") from e

        ctx.session.state["knowledge_base"] = kb.model_dump()
        update_job(job_id, knowledge_base=kb.model_dump())

        yield Event(
            author=self.name,
            content=f"Knowledge base built: {len(kb.deep_findings)} findings, {len(kb.key_facts)} facts, {len(kb.definitions)} definitions",
        )


knowledge_base_agent = KnowledgeBaseAgent(name="KnowledgeBaseAgent")
