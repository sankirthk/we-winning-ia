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
from google.genai import types
from pydantic import ValidationError

from models.narration_script import NarrationScript
from tools.gemini import build_client, GEMINI_FLASH_MODEL, generate_with_retry
from tools.job_store import update_job

PROMPT_TEMPLATE = """
You are narrating a short video where you personally explain a report to a friend
who is smart but hasn't read it. You've read it. You found it interesting.
Now you're telling them what's in it — conversationally, like you're sitting
across from them.

Not a presenter. Not a newsreader. A person talking to another person.

Report manifest:
{manifest_json}

Overall sentiment: {sentiment}
Why: {sentiment_reason}

Output JSON with this exact shape:
{{
  "hook": "<1 sentence — the most surprising thing you found in this report. Start with 'So' or 'Turns out' or just state it directly. Sound like you just looked up from reading and said this out loud.>",
  "scenes": [
    {{
      "scene_id": <int starting at 1>,
      "section_id": <int matching a key_section id>,
      "narration": "<3-4 conversational sentences. You are explaining this section to your friend. Use 'they', 'the team', 'the authors', 'the company' — whoever wrote the report. Say what they found, why it was surprising, and what it means. Use the real numbers but say them naturally — '30 times slower' not '30x latency increase'. Connect ideas with 'but', 'so', 'which means', 'the problem is', 'what's wild is'. Sound human.>",
      "caption": "<The twist or punchline of this scene. Max 6 words. Something that would make someone stop scrolling.>",
      "tone": "<urgent | optimistic | neutral | dramatic | cautious>"
    }}
  ],
  "outro": "<1 sentence — said directly to the viewer. What should they take away from this? Start with 'The takeaway is' or 'Bottom line:' or just say it plainly.>"
}}

Rules:
- One scene per key_section, same order, matching section_id
- Total words across hook + all narration + outro: 90-120 words
- Use the real numbers — but say them how a person would say them out loud
- No bullet points in your head. No listy sentences. Flowing speech only.
- No exclamation points. No rhetorical questions. No "let that sink in".
- Never say "this report", "the document", "the data shows" — you are telling someone what happened, not summarizing a document
- Each scene must flow: here's what they found → here's why it's surprising or contradictory → here's what it means
- Return ONLY valid JSON. No markdown fences. No explanation.

Bad narration: "Pipelining outcomes are kernel-dependent. Prefetching boosted attention 8.7% but slowed MLPs 13% as instruction fetch overhead stalled compute."
Good narration: "So pipelining turned out to be a double-edged sword. Adding prefetch stages made the attention kernels about 9% faster — but the same technique slowed down the MLP layers by 13%, because all those extra pipeline instructions were competing for the same compute units. It helped one thing and broke another."

Bad caption: "Prefetching stalls compute-bound kernels"
Good caption: "Same trick, opposite results"
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
            content=types.Content(role="model", parts=[types.Part(text=f"Script written: {len(script.scenes)} scenes, ~{total_words} words (~{total_words // 2}-{total_words // 2 + 5}s)")]),
        )


narrative_script_agent = NarrativeScriptAgent(name="NarrativeScriptAgent")
