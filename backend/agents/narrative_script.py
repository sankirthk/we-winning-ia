"""
NarrativeScriptAgent
IN:  session["manifest"]          — validated Manifest dict from ParserAgent
OUT: session["narration_script"]  — NarrationScript dict { hook, scenes, outro }

Reads the full manifest JSON and produces a TikTok-style video script.
One scene per key_section, targeting 30-60 seconds total narration (~75-150 words).
Model: Gemini 3 Flash (publishers/google/models/gemini-3-flash-preview)
"""

import json
import re
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from pydantic import ValidationError

from models.narration_script import NarrationScript
from tools.gemini import build_client, GEMINI_FLASH_MODEL
from tools.job_store import update_job

PROMPT_TEMPLATE = """
You are a tech content creator — think Fireship, MKBHD, or Andrej Karpathy's explainer style.
Someone handed you this report. You read it. Now you're breaking it down for your audience in a short-form video.

You are NOT the author. You're reacting to and explaining someone else's work.
Your audience is smart — engineers, founders, analysts. Don't talk down to them.
Be direct, be precise, and let the interesting findings speak for themselves.

Here is the report manifest (JSON):
{manifest_json}

Produce a short-form video script as JSON with this exact shape:

{{
  "hook": "<1 sentence, max 15 words — lead with the most surprising or counterintuitive finding. No hype, just the fact.>",
  "scenes": [
    {{
      "scene_id": <int, starting at 1>,
      "section_id": <int — must match a key_section id from the manifest>,
      "narration": "<2-3 sentences spoken aloud. Explain what they did, what they found, and why it's interesting or surprising. Use specific numbers. Talk like you're explaining to a smart colleague, not reading a press release.>",
      "caption": "<max 8 words — the single most striking stat or finding from this section>",
      "tone": "<urgent | optimistic | neutral | dramatic | cautious — match the actual content, not just sentiment>"
    }}
  ],
  "outro": "<1 sentence — what this means for the field or what question it leaves open. No fluff.>"
}}

Rules:
- Write exactly one scene per key_section in the manifest (same count, same order, matching section_id)
- Total narration across all scenes + hook + outro: 75-150 words (30-60 seconds read aloud)
- Overall document sentiment is {sentiment} — reason: {sentiment_reason}
- Every narration must reference at least one specific number or fact from that section
- No Gen Z slang, no filler phrases ("game-changing", "deep dive", "unpack"), no hype
- Captions are for on-screen overlay — keep them scannable, not a sentence
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
        sentiment = manifest.get("sentiment", "neutral")
        sentiment_reason = manifest.get("sentiment_reason", "")

        prompt = PROMPT_TEMPLATE.format(
            manifest_json=manifest_json,
            sentiment=sentiment,
            sentiment_reason=sentiment_reason,
        )

        client = build_client()
        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
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
