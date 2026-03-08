# VideoScriptAgent
# IN:  session["manifest"]
# OUT: session["video_script"] — { scenes: [{ scene_id, type, avatar, duration_seconds, dialogue, prompt, caption }] }
# Model: Gemini 3.1 Pro (GEMINI_MODEL)
#
# Single agent that writes the spoken dialogue AND directs the visual scenes.
# NarrativeScriptAgent and TTSAgent are no longer in the pipeline.
# Produces alternating presenter / b-roll scenes.
# Presenter scenes carry an avatar field ("male" | "female") — VeoAgent loads
# the pre-generated reference image from AVATAR_MALE_PATH / AVATAR_FEMALE_PATH.
# Dialogue is passed as audio_prompt so Veo 3.1 bakes in lip-synced audio natively.

import json
import os
import re
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from tools.gemini import build_client, GEMINI_MODEL, generate_with_retry
from tools.job_store import update_job

VIDEO_SCRIPT_PROMPT = """
You are a writer and director creating a short-form explainer video about
a report. You write the spoken dialogue AND direct the visuals.

You've read the report. You're explaining it to a smart friend who hasn't.
Conversational, precise, no jargon. Use the real numbers. Sound human.

Report:
Title: {title}
Type: {report_type}
Overall summary: {overall_summary}
Sentiment: {sentiment} — {sentiment_reason}

Key sections:
{sections}

Output a JSON array of scenes. Alternate between presenter and b-roll.
Start with presenter. Cut to b-roll for emphasis. Return to presenter.

[
  {{
    "scene_id": <int starting at 1>,
    "type": "presenter" | "broll",
    "avatar": "male" | "female" | null,
    "duration_seconds": 4 | 6 | 8,
    "dialogue": "<presenter only — words spoken aloud. Conversational, uses real numbers, sounds human. null for b-roll.>",
    "prompt": "<presenter: camera framing + lighting + tone. b-roll: cinematic visual metaphor — no charts, no graphs, no text, no people.>",
    "caption": "<max 6 words — the twist or punchline of this scene>"
  }}
]

Avatar rules:
- Odd-numbered presenter scenes → "male"
- Even-numbered presenter scenes → "female"
- b-roll scenes → null
- The avatar reference image is pre-generated — your prompt describes framing and lighting only,
  not what the presenter looks like

Presenter prompt rules:
- Always include: "direct to camera, shallow depth of field, soft studio lighting,
  modern office bokeh background, vertical 9:16 format, photorealistic 4K, natural gestures"
- Vary the framing: "medium close-up", "chest-up shot", "slight side angle" — never repeat consecutively
- Match lighting tone: cautious = cool blue, optimistic = warm golden, dramatic = high contrast
- Do NOT describe the presenter's appearance — that comes from the reference image

B-roll prompt rules:
- Visual METAPHOR not literal — no charts, graphs, floating numbers, or text overlays
- Specify camera movement: slow push-in, aerial pullback, tracking shot
- No people
- Always end with: "vertical 9:16 format, no text, photorealistic 4K, cinematic motion"

Timing rules:
- Duration must be exactly 4, 6, or 8 seconds (Veo hardware constraint)
- Presenter scenes: 6 or 8 seconds
- B-roll scenes: 4 or 6 seconds
- Total duration: 30-60 seconds across all scenes

Dialogue rules:
- Write how a person actually talks — "thirty times slower" not "30x latency regression"
- Each presenter scene covers one key section or the hook/outro
- No exclamation points. No rhetorical questions. Concrete nouns, active voice.

Example output:
[
  {{
    "scene_id": 1,
    "type": "presenter",
    "avatar": "male",
    "duration_seconds": 8,
    "dialogue": "Turns out, fusing GPU kernels doesn't always make them faster. In fact, it can make them thirty times slower.",
    "prompt": "Medium close-up, direct to camera, cool blue studio lighting, modern office bokeh background, slight left angle, vertical 9:16 format, photorealistic 4K, natural gestures",
    "caption": "More fusion ≠ faster"
  }},
  {{
    "scene_id": 2,
    "type": "broll",
    "avatar": null,
    "duration_seconds": 6,
    "dialogue": null,
    "prompt": "Slow tracking shot through a dark server farm corridor, rows of blinking lights suddenly dimming one by one, cold blue ambient light, fog at floor level, vertical 9:16 format, no text, no people, photorealistic 4K, cinematic motion",
    "caption": "30x slowdown"
  }},
  {{
    "scene_id": 3,
    "type": "presenter",
    "avatar": "female",
    "duration_seconds": 6,
    "dialogue": "The problem is that once you merge too many operations, the GPU can't schedule them efficiently — and bandwidth drops off a cliff.",
    "prompt": "Chest-up shot, direct to camera, warm golden studio lighting, modern office bokeh background, slight right angle, vertical 9:16 format, photorealistic 4K, natural gestures",
    "caption": "Bandwidth fell 97%"
  }}
]

Return ONLY the JSON array. No markdown. No explanation.
"""


def _extract_json(text: str) -> list:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _format_sections(manifest: dict) -> str:
    lines = []
    for s in manifest.get("key_sections", []):
        lines.append(f"[{s['id']}] {s['heading']}")
        lines.append(f"  Summary: {s['summary']}")
        if s.get("key_stats"):
            lines.append(f"  Stats: {'; '.join(s['key_stats'])}")
    return "\n".join(lines)


class VideoScriptAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        job_id = ctx.session.state["job_id"]
        manifest = ctx.session.state["manifest"]

        update_job(job_id, step="video_script")

        prompt = VIDEO_SCRIPT_PROMPT.format(
            title=manifest["title"],
            report_type=manifest["type"],
            overall_summary=manifest["overall_summary"],
            sentiment=manifest.get("sentiment", "neutral"),
            sentiment_reason=manifest.get("sentiment_reason", ""),
            sections=_format_sections(manifest),
        )

        client = build_client()
        response = generate_with_retry(client, GEMINI_MODEL, [prompt])

        scenes = _extract_json(response.text)

        # Prefer session state (set by a future AvatarAgent), fall back to env
        video_script = {
            "scenes": scenes,
            "avatar_male_path": ctx.session.state.get("avatar_male_path", os.getenv("AVATAR_MALE_PATH", "")),
            "avatar_female_path": ctx.session.state.get("avatar_female_path", os.getenv("AVATAR_FEMALE_PATH", "")),
        }
        ctx.session.state["video_script"] = video_script
        update_job(job_id, step="veo", video_script=video_script)

        presenter_count = sum(1 for s in scenes if s["type"] == "presenter")
        broll_count = sum(1 for s in scenes if s["type"] == "broll")
        total_duration = sum(s["duration_seconds"] for s in scenes)

        yield Event(
            author=self.name,
            content=f"VideoScript: {len(scenes)} scenes ({presenter_count} presenter, {broll_count} b-roll), ~{total_duration}s",
        )


video_script_agent = VideoScriptAgent(name="VideoScriptAgent")
