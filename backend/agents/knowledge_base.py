"""
KnowledgeBaseAgent
IN:  session["file_path"]      — local path to the uploaded PDF
OUT: session["knowledge_base"] — deep knowledge dict for LiveAgent system context

Runs in parallel with ParserAgent via ParallelAgent in pipeline.py.
Output is never shown to the user — it is injected into the LiveAgent system prompt
so it can answer detailed follow-up questions with expert-level accuracy.
Model: gemini-3.1-pro-preview via Vertex AI
"""

import asyncio
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
from tools.gemini import build_client, GEMINI_MODEL, generate_with_retry
from tools.storage import read_bytes, save_cache
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
    data = json.loads(text)
    # Coerce list[dict] → list[str] for fields that expect strings.
    # gemini-2.5-pro sometimes returns richer objects even when the prompt says strings.
    for field in ("deep_findings", "key_facts", "risks_and_failures", "successes_and_rationale"):
        if field in data and isinstance(data[field], list):
            data[field] = [
                " ".join(str(v) for v in item.values()) if isinstance(item, dict) else str(item)
                for item in data[field]
            ]
    return data


class KnowledgeBaseAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        job_id = ctx.session.state["job_id"]
        file_path = ctx.session.state["file_path"]

        print(f"\n[KnowledgeBaseAgent] ▶ Starting", flush=True)
        pdf_bytes = read_bytes(file_path)
        print(f"[KnowledgeBaseAgent]   PDF read: {len(pdf_bytes):,} bytes from {file_path!r}", flush=True)

        client = build_client()
        print(f"[KnowledgeBaseAgent]   Sending PDF to Gemini...", flush=True)

        # Wrap blocking HTTP call in to_thread so ParallelAgent can truly interleave
        response = await asyncio.to_thread(
            generate_with_retry,
            client,
            GEMINI_MODEL,
            [types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), PROMPT],
        )

        print(f"[KnowledgeBaseAgent]   Gemini responded, validating...", flush=True)
        raw = _extract_json(response.text)

        try:
            kb = KnowledgeBase.model_validate(raw)
        except ValidationError as e:
            print(f"[KnowledgeBaseAgent] ❌ Validation failed: {e}", flush=True)
            raise ValueError(f"KnowledgeBaseAgent: validation failed:\n{e}") from e

        print(f"[KnowledgeBaseAgent] ✅ Done — {len(kb.deep_findings)} findings, {len(kb.key_facts)} facts, {len(kb.definitions)} definitions", flush=True)
        kb_dict = kb.model_dump()
        ctx.session.state["knowledge_base"] = kb_dict
        update_job(job_id, knowledge_base=kb_dict)
        if ctx.session.state.get("pdf_hash"):
            save_cache(ctx.session.state["pdf_hash"], "knowledge_base", kb_dict)
            print(f"[KnowledgeBaseAgent]   Cached knowledge_base for pdf_hash={ctx.session.state['pdf_hash'][:8]}...", flush=True)

        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=f"Knowledge base built: {len(kb.deep_findings)} findings, {len(kb.key_facts)} facts, {len(kb.definitions)} definitions")]),
        )


knowledge_base_agent = KnowledgeBaseAgent(name="KnowledgeBaseAgent")


def run_knowledge_base(file_path: str, job_id: str) -> dict:
    """
    Standalone blocking function — call via asyncio.to_thread for true parallelism.
    Identical logic to KnowledgeBaseAgent but without ADK scaffolding.
    """
    print(f"\n[run_knowledge_base] ▶ Starting", flush=True)

    pdf_bytes = read_bytes(file_path)
    client = build_client()

    response = generate_with_retry(
        client,
        GEMINI_MODEL,
        [
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            PROMPT,
        ],
    )

    raw = _extract_json(response.text)
    kb = KnowledgeBase.model_validate(raw)
    kb_dict = kb.model_dump()

    print(f"[run_knowledge_base] ✅ {len(kb.deep_findings)} findings, {len(kb.key_facts)} facts", flush=True)
    update_job(job_id, knowledge_base=kb_dict)
    return kb_dict
