"""
NarrativeScriptAgent
IN:  session["manifest"]          — validated Manifest dict from ParserAgent
OUT: session["narration_script"]  — NarrationScript dict { hook, scenes, outro }

Reads the full manifest JSON and produces a TikTok-style video script.
One scene per key_section, targeting 30-60 seconds total narration (~75-150 words).
Model: uses GEMINI_MODEL from tools/gemini.py (swap to Flash for faster/cheaper runs)
"""

import json
import re
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from pydantic import ValidationError

from models.narration_script import NarrationScript
from tools.gemini import build_client, GEMINI_MODEL
from tools.job_store import update_job

PROMPT_TEMPLATE = """
You are a TikTok scriptwriter for corporate and research reports. Write punchy, Bloomberg-style narration — fast, conversational, zero jargon.

Here is the report manifest (JSON):
{manifest_json}

Produce a short-form video script as JSON with this exact shape:

{{
  "hook": "<1 punchy opening sentence, max 15 words — grab attention immediately>",
  "scenes": [
    {{
      "scene_id": <int, starting at 1>,
      "section_id": <int — must match a key_section id from the manifest>,
      "narration": "<2-3 fast sentences spoken aloud — conversational, no jargon, punchy>",
      "caption": "<max 8 words — include the most striking key_stat from this section>",
      "tone": "<urgent | optimistic | neutral | dramatic — match section content and overall sentiment>"
    }}
  ],
  "outro": "<1 closing sentence — prompts reflection or action>"
}}

Rules:
- Write exactly one scene per key_section in the manifest (same count, same order, matching section_id)
- Total narration across all scenes + hook + outro should be 75-150 words (30-60 seconds when read aloud)
- Tone per scene should reflect the section content AND the overall document sentiment: {sentiment}
- Caption must include a specific number or stat from that section's key_stats if available
- Write like a Bloomberg short, not a board presentation — punchy verbs, active voice
- Return ONLY valid JSON. No markdown fences, no explanation.
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


class NarrativeScriptAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        job_id = ctx.session.state["job_id"]
        # manifest is a dict produced by ParserAgent and stored in session state
        manifest = ctx.session.state["manifest"]

        # Mark job as scripting so the status endpoint reflects the current step
        update_job(job_id, step="scripting")

        # Serialize the full manifest so Gemini sees all headings, summaries, and key_stats
        manifest_json = json.dumps(manifest, indent=2)
        # Pass sentiment separately so the prompt rule is explicit and easy to tune
        sentiment = manifest.get("sentiment", "neutral")

        prompt = PROMPT_TEMPLATE.format(
            manifest_json=manifest_json,
            sentiment=sentiment,
        )

        client = build_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt],
        )

        # Strip any accidental markdown fences before JSON parsing
        raw = _extract_json(response.text)

        try:
            script = NarrationScript.model_validate(raw)
        except ValidationError as e:
            raise ValueError(f"NarrativeScriptAgent: validation failed:\n{e}") from e

        # Write to session so VideoScriptAgent and TTSAgent can consume it downstream
        ctx.session.state["narration_script"] = script.model_dump()
        # Advance job step to "tts" and persist the script so LiveAgent can access it
        update_job(job_id, step="tts", narration_script=script.model_dump())

        # Rough word count used only for the log message — not used downstream
        total_words = (
            len(script.hook.split())
            + sum(len(s.narration.split()) for s in script.scenes)
            + len(script.outro.split())
        )

        yield Event(
            author=self.name,
            content=f"Script written: {len(script.scenes)} scenes, ~{total_words} words (~{total_words // 2}-{total_words // 2 + 5}s)",
        )


narrative_script_agent = NarrativeScriptAgent(name="NarrativeScriptAgent")
