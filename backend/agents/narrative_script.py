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
from tools.gemini import build_client, GEMINI_FLASH_MODEL, generate_with_retry
from tools.job_store import update_job

PROMPT_TEMPLATE = """
You are writing narration for a short-form video that condenses a report
into 30-60 seconds. Think Veritasium, Wendover Productions, or a Bloomberg
QuickTake — intelligent, fast-paced, but never dumbed down. The audience
is smart. Respect that.

Report manifest:
{manifest_json}

Produce a script as JSON with this exact shape:
{{
  "hook": "<1 sentence — open with the most surprising or counterintuitive finding. No questions. State it as fact.>",
  "scenes": [
    {{
      "scene_id": <int starting at 1>,
      "section_id": <int matching a key_section id>,
      "narration": "<2-3 sentences. Explain the finding clearly and quickly. Use the actual numbers. Connect cause to effect. Write how a smart person explains something to another smart person — no filler, no hype, no exclamation points.>",
      "caption": "<The single most important stat from this section. Just the number and context. Max 8 words.>",
      "tone": "<urgent | optimistic | neutral | dramatic | cautious>"
    }}
  ],
  "outro": "<1 sentence — what this means going forward. Forward-looking, grounded, no fluff.>"
}}

Rules:
- One scene per key_section, same order, matching section_id
- Total words across hook + all narration + outro: 75-110 words maximum
- Use the actual numbers and stats from the manifest — never vague
- No rhetorical questions. No "but here's the thing". No "let that sink in".
- No exclamation points. No emojis in narration.
- Active voice. Short sentences. Concrete nouns.
- Tone per scene must match both the section content and overall sentiment: {sentiment}
- Each narration must have a connective arc: state the finding, then explain the tension or contradiction, then land the implication. Never list facts side by side without connecting them causally.
- Captions must capture the counterintuitive twist of the scene, not just restate the stat. A good caption makes someone stop scrolling.
- Return ONLY valid JSON. No markdown fences. No explanation.

Bad narration: "Pipelining impact is non-linear and kernel-dependent. Aggressive prefetching improved attention throughput by 8.7 percent but slowed compute-bound kernels by 13 percent due to instruction fetch overhead."
Good narration: "Pipelining helps and hurts simultaneously. The same aggressive prefetching that boosted attention throughput 8.7% actively stalled compute-bound kernels — instruction fetch overhead ate the gains."

Bad caption: "97% bandwidth collapse in fused kernels"
Good caption: "More fusion ≠ faster inference"
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
        )

        client = build_client()
        response = generate_with_retry(client, GEMINI_FLASH_MODEL, [prompt])

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
