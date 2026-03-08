"""
End-to-end pipeline test: PDF → Parser + KnowledgeBase (parallel) → NarrativeScript

Parser and KnowledgeBase fire concurrently via asyncio.to_thread (Gemini calls
are synchronous/blocking — to_thread offloads them to a thread pool so the event
loop stays free and both run at the same time).

Narration starts immediately after Parser finishes, without waiting for KnowledgeBase.

Usage:
  cd backend
  uv run python tests/test_pipeline.py path/to/report.pdf
"""

import asyncio
import json
import sys

from dotenv import load_dotenv
load_dotenv()

from pydantic import ValidationError

from agents.parser import _parse_with_gemini
from agents.knowledge_base import _extract_json as kb_extract_json, PROMPT as KB_PROMPT
from agents.narrative_script import _extract_json as ns_extract_json, PROMPT_TEMPLATE
from models.manifest import Manifest
from models.knowledge_base import KnowledgeBase
from models.narration_script import NarrationScript
from tools.gemini import build_client, GEMINI_MODEL, GEMINI_FLASH_MODEL, generate_with_retry
from google.genai import types


def _divider(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


async def run_parser(pdf_bytes: bytes, client) -> Manifest:
    _divider("STEP 1a: ParserAgent")
    print("Sending PDF to Gemini (manifest extraction)...")
    # Offload blocking sync call to thread pool so KB can run concurrently
    raw = await asyncio.to_thread(_parse_with_gemini, pdf_bytes, client)
    manifest = Manifest.model_validate(raw)
    print(f"✓ Manifest valid")
    print(f"  Title:     {manifest.title}")
    print(f"  Type:      {manifest.type}  |  Pages: {manifest.total_pages}  |  Sentiment: {manifest.sentiment}")
    print(f"  Why:       {manifest.sentiment_reason}")
    print(f"  Summary:   {manifest.overall_summary}")
    print(f"  Sections:")
    for s in manifest.key_sections:
        print(f"    [{s.id}] {s.heading} (p.{s.page})")
        print(f"         {s.summary}")
        print(f"         Stats: {s.key_stats}")
    return manifest


def _kb_generate(pdf_bytes: bytes, client) -> KnowledgeBase:
    """Blocking KB generation — called via to_thread."""
    response = generate_with_retry(
        client,
        GEMINI_MODEL,
        [
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            KB_PROMPT,
        ],
    )
    raw = kb_extract_json(response.text)
    return KnowledgeBase.model_validate(raw)


async def run_knowledge_base(pdf_bytes: bytes, client) -> KnowledgeBase:
    _divider("STEP 1b: KnowledgeBaseAgent (concurrent with Parser)")
    print("Sending PDF to Gemini (deep knowledge extraction)...")
    # Offload blocking sync call to thread pool
    kb = await asyncio.to_thread(_kb_generate, pdf_bytes, client)
    print(f"✓ Knowledge base valid")
    print(f"  Deep findings:    {len(kb.deep_findings)}")
    print(f"  Key facts:        {len(kb.key_facts)}")
    print(f"  Risks & failures: {len(kb.risks_and_failures)}")
    print(f"  Definitions:      {len(kb.definitions)}")
    print(f"  Expert detail (first 200 chars):")
    print(f"    {kb.expert_detail[:200]}...")
    return kb


def _narration_generate(manifest: Manifest, client) -> NarrationScript:
    """Blocking narration generation — called via to_thread."""
    prompt = PROMPT_TEMPLATE.format(
        manifest_json=json.dumps(manifest.model_dump(), indent=2),
        sentiment=manifest.sentiment,
        sentiment_reason=manifest.sentiment_reason,
    )
    response = generate_with_retry(client, GEMINI_FLASH_MODEL, [prompt])
    raw = ns_extract_json(response.text)
    return NarrationScript.model_validate(raw)


async def run_narrative_script(manifest: Manifest, client) -> NarrationScript:
    _divider("STEP 2: NarrativeScriptAgent")
    print("Sending manifest to Gemini Flash (script generation)...")
    # Offload blocking sync call to thread pool
    script = await asyncio.to_thread(_narration_generate, manifest, client)
    total_words = (
        len(script.hook.split())
        + sum(len(s.narration.split()) for s in script.scenes)
        + len(script.outro.split())
    )
    print(f"✓ NarrationScript valid")
    print(f"  Scenes:      {len(script.scenes)}")
    print(f"  Total words: ~{total_words} (~{total_words // 2}s read aloud)")
    print(f"\n  HOOK:  {script.hook}")
    for scene in script.scenes:
        print(f"\n  Scene {scene.scene_id} → section {scene.section_id} [{scene.tone}]")
        print(f"    Caption:   {scene.caption}")
        print(f"    Narration: {scene.narration}")
    print(f"\n  OUTRO: {script.outro}")
    return script


async def main(pdf_path: str):
    print(f"Pipeline test: {pdf_path}\n")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    client = build_client()

    # KB fires immediately as a background task — runs in its own thread
    kb_task = asyncio.create_task(run_knowledge_base(pdf_bytes, client))

    # Parser runs concurrently with KB (both in thread pool via to_thread)
    manifest = await run_parser(pdf_bytes, client)

    # Narration starts as soon as parser is done — doesn't wait for KB
    script = await run_narrative_script(manifest, client)

    # KB is likely already done; collect it
    kb = await kb_task

    _divider("DONE")
    print(f"  Manifest sections:    {len(manifest.key_sections)}")
    print(f"  Knowledge base facts: {len(kb.key_facts)}")
    print(f"  Script scenes:        {len(script.scenes)}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python tests/test_pipeline.py path/to/report.pdf")
        sys.exit(1)

    try:
        asyncio.run(main(sys.argv[1]))
    except ValidationError as e:
        print(f"\n✗ Validation error:\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
