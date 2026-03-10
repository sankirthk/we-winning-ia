# VideoScriptAgent
# IN:  session["manifest"]
# OUT: session["video_script"] — { scenes: [{ scene_id, type, avatar, duration_seconds, dialogue, background, prompt, caption }] }
# Model: Gemini 3.1 Pro (GEMINI_MODEL)
#
# Single presenter on screen the entire video. Background behind them changes
# each scene to visually reinforce what's being said — news anchor with a
# dynamic backdrop. No b-roll, no scene switching, no audio overlay.

import json
import os
import re
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from tools.gemini import build_client, GEMINI_MODEL, generate_with_retry
from tools.storage import save_cache
from tools.job_store import update_job

_TONE_DIALOGUE_RULES = {
    "formal": (
        "Dialogue tone: FORMAL. No contractions, no slang. Precise, professional language — "
        "as if briefing a C-suite audience. Use exact figures. Active but measured voice. "
        "Example: 'Revenue declined 8.3 percent year-over-year, driven by North American enterprise contraction.'"
    ),
    "explanatory": (
        "Dialogue tone: EXPLANATORY. Clear, educational — break down the 'why', not just the 'what'. "
        "Use analogies where helpful. Guide the viewer through reasoning step by step. "
        "Example: 'Revenue fell because their biggest customers, large US enterprises, pulled back — "
        "and that segment is their main growth engine.'"
    ),
    "casual": (
        "Dialogue tone: CASUAL / INFORMAL. Conversational, contractions welcome, relatable framing. "
        "Talk like a smart friend explaining this over coffee. Plain language, punchy delivery. "
        "Example: 'Basically, their biggest customers stopped spending — and that's where most of their money comes from, so everything took a hit.'"
    ),
}

VIDEO_SCRIPT_PROMPT = """
You are directing a short-form explainer video with a single AI presenter
who speaks directly to camera the entire time. The presenter stays on screen
throughout — but the background behind them changes each scene to visually
reinforce what's being said.

Think: news anchor with a dynamic backdrop. The person is the constant.
The background is the visual storytelling.

Report:
Title: {title}
Type: {report_type}
Overall summary: {overall_summary}
Sentiment: {sentiment} — {sentiment_reason}

Key sections:
{sections}

Output a JSON array of scenes — one per key section plus a hook and outro.

[
  {{
    "scene_id": <int starting at 1>,
    "type": "presenter",
    "avatar": "male" | "female",
    "duration_seconds": 8,
    "dialogue": "<words spoken aloud. {tone_instruction} Max 20 words per scene — Veo 8-second limit.>",
    "background": "<what appears behind the presenter — a visual metaphor for this scene's content. Specific, cinematic, no text, no floating numbers. e.g. 'glowing server racks fading to dark', 'aerial city skyline at golden hour', 'abstract deep blue particle field slowly shifting'>",
    "prompt": "<full Veo prompt combining presenter framing + background. Always include: 'direct to camera, shallow depth of field, soft studio lighting, vertical 9:16 format, photorealistic 4K, natural gestures'. Vary framing each scene.>",
    "caption": "<max 6 words — the punchline of this scene>"
  }}
]

Avatar rules:
- Alternate male/female each scene: scene 1 → male, scene 2 → female, etc.
- Do NOT describe the presenter's appearance in the prompt — reference image handles that

Prompt rules:
- Vary framing each scene: "medium close-up", "chest-up shot", "slight side angle" — never repeat consecutively
- Background must be cinematic and metaphorical — not literal data viz
- Match background tone to sentiment: cautious = dark/cool, optimistic = bright/warm, dramatic = high contrast
- No text, no charts, no floating numbers anywhere in the frame

Dialogue rules:
- {tone_instruction}
- Max 20 words per scene — any more gets cut off by Veo's 8-second limit
- Real numbers, active voice
- Flows naturally from one scene to the next as one continuous narration

Total duration: 16 seconds exactly (2 scenes × 8s). Always output exactly 2 scenes — a hook and a key-insight scene. Never more, never less.

Example:
[
  {{
    "scene_id": 1,
    "type": "presenter",
    "avatar": "male",
    "duration_seconds": 8,
    "dialogue": "Turns out, fusing GPU kernels doesn't always make them faster. It can make them thirty times slower.",
    "background": "Glowing server racks in a dark corridor, lights slowly dimming one by one",
    "prompt": "Medium close-up, direct to camera, cool blue studio lighting, background: glowing server racks in dark corridor with lights slowly dimming, shallow depth of field, vertical 9:16 format, photorealistic 4K, natural gestures",
    "caption": "More fusion ≠ faster"
  }},
  {{
    "scene_id": 2,
    "type": "presenter",
    "avatar": "female",
    "duration_seconds": 8,
    "dialogue": "Bandwidth dropped 97 percent — because the fusion disrupted how the GPU accesses memory.",
    "background": "Abstract deep blue particle field slowly contracting and fading to black",
    "prompt": "Chest-up shot, direct to camera, cool blue studio lighting, background: abstract deep blue particle field contracting to black, shallow depth of field, vertical 9:16 format, photorealistic 4K, natural gestures",
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

        tone = ctx.session.state.get("tone", "explanatory")
        tone_instruction = _TONE_DIALOGUE_RULES.get(tone, _TONE_DIALOGUE_RULES["explanatory"])

        update_job(job_id, step="video_script")
        print(f"\n[VideoScriptAgent] ▶ Starting — title={manifest.get('title')!r}  tone={tone!r}", flush=True)

        prompt = VIDEO_SCRIPT_PROMPT.format(
            title=manifest["title"],
            report_type=manifest["type"],
            overall_summary=manifest["overall_summary"],
            sentiment=manifest.get("sentiment", "neutral"),
            sentiment_reason=manifest.get("sentiment_reason", ""),
            sections=_format_sections(manifest),
            tone_instruction=tone_instruction,
        )

        client = build_client()
        print(f"[VideoScriptAgent]   Calling Gemini for scene directions...", flush=True)
        response = generate_with_retry(client, GEMINI_MODEL, [prompt])

        print(f"[VideoScriptAgent]   Gemini responded, parsing scenes...", flush=True)
        scenes = _extract_json(response.text)

        # Hard cap — exactly 2 scenes regardless of what the model returns
        if len(scenes) > 2:
            print(f"[VideoScriptAgent]   Trimming {len(scenes)} scenes → 2", flush=True)
            scenes = scenes[:2]

        # Prefer session state (set by a future AvatarAgent), fall back to env
        video_script = {
            "scenes": scenes,
            "avatar_male_path": ctx.session.state.get("avatar_male_path", os.getenv("AVATAR_MALE_PATH", "")),
            "avatar_female_path": ctx.session.state.get("avatar_female_path", os.getenv("AVATAR_FEMALE_PATH", "")),
        }
        ctx.session.state["video_script"] = video_script
        update_job(job_id, step="veo", video_script=video_script)

        # Cache so future runs with the same PDF + tone skip this step
        if ctx.session.state.get("pdf_hash"):
            save_cache(ctx.session.state["pdf_hash"], f"video_script_{tone}", video_script)
            print(f"[VideoScriptAgent]   Cached video_script_{tone} for pdf_hash={ctx.session.state['pdf_hash'][:8]}...", flush=True)

        total_duration = sum(s["duration_seconds"] for s in scenes)
        print(f"[VideoScriptAgent] ✅ Done — {len(scenes)} scenes, ~{total_duration}s total", flush=True)
        for s in scenes:
            print(f"  scene {s['scene_id']:2d}  avatar={s.get('avatar'):<6} duration={s['duration_seconds']}s  caption={s.get('caption')!r}", flush=True)

        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=f"VideoScript: {len(scenes)} presenter scenes, ~{total_duration}s")]),
        )


video_script_agent = VideoScriptAgent(name="VideoScriptAgent")
