"""
Quick test for KnowledgeBaseAgent — run directly without starting the server.

Usage:
  cd backend
  uv run python tests/test_knowledge_base.py path/to/report.pdf
"""

import asyncio
import json
import sys

from dotenv import load_dotenv
load_dotenv()

from agents.knowledge_base import _extract_json
from models.knowledge_base import KnowledgeBase
from tools.gemini import build_client, GEMINI_MODEL
from google.genai import types
from agents.knowledge_base import PROMPT
from pydantic import ValidationError


async def test(pdf_path: str):
    print(f"Building knowledge base from: {pdf_path}\n")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    client = build_client()

    print("Sending to Gemini 3.1 Pro...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            PROMPT,
        ],
    )

    raw = _extract_json(response.text)

    print("\n--- Raw JSON from model ---")
    print(json.dumps(raw, indent=2))

    print("\n--- Pydantic validation ---")
    try:
        kb = KnowledgeBase.model_validate(raw)
        print(f"✓ Valid knowledge base")
        print(f"  Deep findings:          {len(kb.deep_findings)}")
        print(f"  Key facts:              {len(kb.key_facts)}")
        print(f"  Risks & failures:       {len(kb.risks_and_failures)}")
        print(f"  Successes & rationale:  {len(kb.successes_and_rationale)}")
        print(f"  Definitions:            {len(kb.definitions)}")
        print(f"\n  Expert detail (first 300 chars):\n  {kb.expert_detail[:300]}...")
    except ValidationError as e:
        print(f"✗ Validation failed:\n{e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python tests/test_knowledge_base.py path/to/report.pdf")
        sys.exit(1)

    asyncio.run(test(sys.argv[1]))
