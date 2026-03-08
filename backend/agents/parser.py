"""
ParserAgent
IN:  session["file_path"]  — local path to the uploaded PDF
OUT: session["manifest"]   — validated Manifest dict

Two backends, switched via PARSER_BACKEND env var:

  PARSER_BACKEND=gemini      (default)
    → PDF sent directly to Gemini 3.1 Pro as a native file part.
      One-shot: model both reads and reasons over the PDF.

  PARSER_BACKEND=documentai
    → Document AI Layout Parser extracts structured text + table layout first.
      That structured text is then passed to Gemini 3.1 Pro to produce the manifest.
      Slower but gives better spatial understanding of tables and multi-column layouts.
      Requires DOCUMENT_AI_PROCESSOR_ID env var.
"""

import json
import os
import re
from typing import AsyncGenerator

from google import genai
from google.genai import types
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from pydantic import ValidationError

from models.manifest import Manifest
from tools.gemini import build_client, GEMINI_MODEL
from tools.job_store import update_job

MANIFEST_PROMPT = """
You are a document analyst. Based on the report content provided, extract the following as JSON.

{
  "title": "<report title>",
  "type": "<corporate | financial | research | other>",
  "total_pages": <int>,
  "key_sections": [
    {
      "id": <int, starting at 1>,
      "heading": "<section heading>",
      "summary": "<2-3 sentence summary of this section>",
      "key_stats": ["<specific number, percentage, or named fact>"],
      "page": <page number where this section starts>
    }
  ],
  "overall_summary": "<1-2 sentence summary of the entire report>",
  "sentiment": "<positive | cautious | negative | neutral>"
}

Rules:
- Include only the most important 3-6 sections — skip boilerplate like table of contents or legal disclaimers
- key_stats must be specific numbers, percentages, or named facts — not vague statements
- Return ONLY valid JSON. No markdown fences, no explanation.
"""




def _extract_json(text: str) -> dict:
    """Strip markdown fences before parsing — models sometimes wrap JSON anyway."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _parse_with_gemini(pdf_bytes: bytes, client: genai.Client) -> dict:
    """Send PDF directly to Gemini as a native file part."""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            MANIFEST_PROMPT,
        ],
    )
    return _extract_json(response.text)


def _parse_with_documentai(pdf_bytes: bytes, client: genai.Client) -> dict:
    """
    Run Document AI Layout Parser first to extract structured text and table layout,
    then pass that structured content to Gemini to produce the manifest JSON.
    """
    from google.cloud import documentai

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    processor_id = os.getenv("DOCUMENT_AI_PROCESSOR_ID")

    if not processor_id:
        raise ValueError("DOCUMENT_AI_PROCESSOR_ID env var is required when PARSER_BACKEND=documentai")

    # Step 1: Document AI extracts structured layout
    docai_client = documentai.DocumentProcessorServiceClient(
        client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
    )
    processor_name = docai_client.processor_path(project_id, location, processor_id)

    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf"),
    )
    result = docai_client.process_document(request=request)
    document = result.document

    # Step 2: Convert Document AI output to structured text Gemini can reason over
    structured_text = _docai_to_structured_text(document)

    # Step 3: Gemini reasons over structured text → manifest JSON
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[structured_text, MANIFEST_PROMPT],
    )
    return _extract_json(response.text)


def _docai_to_structured_text(document) -> str:
    """
    Convert Document AI document object to a clean structured text representation.
    Preserves table structure and reading order for Gemini to reason over.
    """
    lines = []
    full_text = document.text

    for page_num, page in enumerate(document.pages, start=1):
        lines.append(f"\n--- Page {page_num} ---")

        # Tables: render as markdown so Gemini sees column relationships clearly
        for table in page.tables:
            lines.append("\n[TABLE]")
            for row in table.header_rows:
                cells = [_get_text(cell.layout, full_text) for cell in row.cells]
                lines.append("| " + " | ".join(cells) + " |")
                lines.append("|" + "---|" * len(cells))
            for row in table.body_rows:
                cells = [_get_text(cell.layout, full_text) for cell in row.cells]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("[/TABLE]")

        # Paragraphs in reading order
        for block in page.blocks:
            text = _get_text(block.layout, full_text).strip()
            if text:
                lines.append(text)

        # Note visual elements (charts/images) by position for context
        for element in page.visual_elements:
            if "image" in element.type_.lower() or "figure" in element.type_.lower():
                lines.append(f"[VISUAL: {element.type_} on page {page_num}]")

    return "\n".join(lines)


def _get_text(layout, full_text: str) -> str:
    """Extract text from a Document AI layout segment."""
    text = ""
    for segment in layout.text_anchor.text_segments:
        start = int(segment.start_index) if segment.start_index else 0
        end = int(segment.end_index)
        text += full_text[start:end]
    return text.strip()


class ParserAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        job_id = ctx.session.state["job_id"]
        file_path = ctx.session.state["file_path"]
        backend = os.getenv("PARSER_BACKEND", "gemini").lower()

        update_job(job_id, step="parsing")

        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        client = build_client()

        if backend == "documentai":
            raw = _parse_with_documentai(pdf_bytes, client)
        else:
            raw = _parse_with_gemini(pdf_bytes, client)

        try:
            manifest = Manifest.model_validate(raw)
        except ValidationError as e:
            raise ValueError(f"ParserAgent: manifest validation failed:\n{e}") from e

        ctx.session.state["manifest"] = manifest.model_dump()

        # Write to job store now so LiveAgent can load it before the full pipeline finishes
        update_job(job_id, step="scripting", manifest=manifest.model_dump())

        yield Event(
            author=self.name,
            content=f"[{backend}] Parsed '{manifest.title}' — {len(manifest.key_sections)} sections, sentiment: {manifest.sentiment}",
        )


parser_agent = ParserAgent(name="ParserAgent")
