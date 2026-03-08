"""
Quick test for NarrativeScriptAgent — run directly without starting the server.

Accepts a pre-existing manifest JSON file (from test_parser.py output) so it
can be tested without re-parsing a PDF.

Usage:
  cd backend
  uv run python tests/test_narrative_script.py path/to/manifest.json
"""

import json
import sys

from dotenv import load_dotenv
load_dotenv()

from agents.narrative_script import _extract_json, PROMPT_TEMPLATE
from models.narration_script import NarrationScript
from tools.gemini import build_client, GEMINI_MODEL
from pydantic import ValidationError


def test(manifest_path: str):
    print(f"Loading manifest from: {manifest_path}\n")

    with open(manifest_path) as f:
        manifest = json.load(f)

    sentiment = manifest.get("sentiment", "neutral")
    manifest_json = json.dumps(manifest, indent=2)

    prompt = PROMPT_TEMPLATE.format(
        manifest_json=manifest_json,
        sentiment=sentiment,
    )

    client = build_client()

    print("Sending to Gemini (NarrativeScriptAgent)...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt],
    )

    raw = _extract_json(response.text)

    print("\n--- Raw JSON from model ---")
    print(json.dumps(raw, indent=2))

    print("\n--- Pydantic validation ---")
    try:
        script = NarrationScript.model_validate(raw)
        total_words = (
            len(script.hook.split())
            + sum(len(s.narration.split()) for s in script.scenes)
            + len(script.outro.split())
        )
        print(f"✓ Valid NarrationScript")
        print(f"  Scenes:       {len(script.scenes)}")
        print(f"  Total words:  ~{total_words} (~{total_words // 2}-{total_words // 2 + 5}s)")
        print(f"\n  Hook:  {script.hook}")
        print(f"  Outro: {script.outro}")
        print()
        for scene in script.scenes:
            print(f"  Scene {scene.scene_id} (section {scene.section_id}) [{scene.tone}]")
            print(f"    Caption:   {scene.caption}")
            print(f"    Narration: {scene.narration}")
            print()
    except ValidationError as e:
        print(f"✗ Validation failed:\n{e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python tests/test_narrative_script.py path/to/manifest.json")
        sys.exit(1)

    test(sys.argv[1])
