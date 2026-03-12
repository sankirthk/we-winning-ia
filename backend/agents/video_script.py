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
from google.genai import types

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

Output a JSON array of presenter-only scenes.
Do NOT generate b-roll scenes.

[
  {{
    "scene_id": <int starting at 1>,
    "type": "presenter",
    "avatar": "male" | "female",
    "duration_seconds": 4 | 6 | 8,
    "dialogue": "<words spoken aloud. Conversational, uses real numbers, sounds human.>",
    "prompt": "<camera framing + lighting + tone + visual style that looks like report-derived visuals, not abstract cinematic metaphor>",
    "caption": "<max 6 words — the twist or punchline of this scene>"
  }}
]

Avatar rules:
- Odd-numbered presenter scenes → "male"
- Even-numbered presenter scenes → "female"
- The avatar reference image is pre-generated — your prompt describes framing and lighting only,
  not what the presenter looks like

Presenter visual rules:
- Always include: "direct to camera, shallow depth of field, soft studio lighting,
  modern office bokeh background, widescreen 16:9 format, photorealistic 4K, natural gestures"
- Vary the framing: "medium close-up", "chest-up shot", "slight side angle" — never repeat consecutively
- Match lighting tone: cautious = cool blue, optimistic = warm golden, dramatic = high contrast
- Do NOT describe the presenter's appearance — that comes from the reference image
- Use visuals grounded in report content: mention section topics, tables, figures, charts, or page context as backdrop cues
- Avoid abstract metaphor shots and unrelated cinematic inserts

Timing rules:
- Duration must be exactly 4, 6, or 8 seconds (Veo hardware constraint)
- Presenter scenes: 6 or 8 seconds
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
    "prompt": "Medium close-up, direct to camera, cool blue studio lighting, modern office bokeh background, slight left angle, widescreen 16:9 format, photorealistic 4K, natural gestures",
    "caption": "More fusion ≠ faster"
  }},
  {{
    "scene_id": 2,
    "type": "presenter",
    "avatar": "female",
    "duration_seconds": 6,
    "dialogue": "In the report's performance section, they show that after aggressive fusion, memory throughput collapses and latency spikes.",
    "prompt": "Chest-up shot, direct to camera, cool blue studio lighting, modern office bokeh background, section visual cues from report performance tables and charts, widescreen 16:9 format, photorealistic 4K, natural gestures",
    "caption": "30x slowdown"
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


def _cap_scene_duration(raw_duration: int, scene_type: str) -> int:
    _ = raw_duration
    _ = scene_type
    # Veo reference_to_video currently supports 8s duration.
    return 8


def _max_words_for_duration(duration_seconds: int) -> int:
    # ~2.3 spoken words/sec keeps delivery natural and avoids truncation
    return 20 if duration_seconds >= 8 else 14


def _normalize_dialogue_to_duration(text: str, duration_seconds: int) -> str:
    words = (text or "").strip().split()
    max_words = _max_words_for_duration(duration_seconds)
    if len(words) > max_words:
        words = words[:max_words]
    out = " ".join(words).strip()
    if out and out[-1] not in ".!?":
        out += "."
    return out


def _enforce_max_total_duration(
    scenes: list[dict], max_seconds: int = 60
) -> list[dict]:
    kept: list[dict] = []
    total = 0

    for scene in scenes:
        scene_type = "presenter"
        capped_duration = _cap_scene_duration(
            int(scene.get("duration_seconds", 4)),
            scene_type,
        )

        if total + capped_duration > max_seconds:
            break

        scene["duration_seconds"] = capped_duration
        scene["type"] = "presenter"
        if not scene.get("dialogue"):
            scene["dialogue"] = scene.get("caption") or "Key finding from the report."
        scene["dialogue"] = _normalize_dialogue_to_duration(
            scene["dialogue"],
            capped_duration,
        )
        kept.append(scene)
        total += capped_duration

    if not kept and scenes:
        first = scenes[0]
        first["type"] = "presenter"
        first["duration_seconds"] = 6
        if not first.get("dialogue"):
            first["dialogue"] = first.get("caption") or "Key finding from the report."
        first["dialogue"] = _normalize_dialogue_to_duration(first["dialogue"], 6)
        kept.append(first)

    for idx, scene in enumerate(kept, start=1):
        scene["scene_id"] = idx
        scene["type"] = "presenter"
        scene["avatar"] = "male" if idx % 2 == 1 else "female"

    if kept:
        last = kept[-1]
        outro = "That is the key takeaway from this report."
        last_dialogue = (last.get("dialogue") or "").strip()
        if "key takeaway" not in last_dialogue.lower():
            merged = f"{last_dialogue} {outro}".strip()
            last["dialogue"] = _normalize_dialogue_to_duration(
                merged,
                int(last["duration_seconds"]),
            )

    return kept


class VideoScriptAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        job_id = ctx.session.state["job_id"]
        manifest = ctx.session.state["manifest"]

        update_job(job_id, step="video_script")
        print(f"\n[VideoScriptAgent] ▶ Starting — title={manifest.get('title')!r}", flush=True)

        prompt = VIDEO_SCRIPT_PROMPT.format(
            title=manifest["title"],
            report_type=manifest["type"],
            overall_summary=manifest["overall_summary"],
            sentiment=manifest.get("sentiment", "neutral"),
            sentiment_reason=manifest.get("sentiment_reason", ""),
            sections=_format_sections(manifest),
        )

        client = build_client()
        print(f"[VideoScriptAgent]   Calling Gemini for scene directions...", flush=True)
        response = generate_with_retry(client, GEMINI_MODEL, [prompt])

        print(f"[VideoScriptAgent]   Gemini responded, parsing scenes...", flush=True)
        scenes = _extract_json(response.text)
        scenes = _enforce_max_total_duration(scenes, max_seconds=60)

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
        print(f"[VideoScriptAgent] ✅ Done — {len(scenes)} scenes ({presenter_count} presenter, {broll_count} b-roll), ~{total_duration}s total", flush=True)
        for s in scenes:
            print(f"  scene {s['scene_id']:2d}  type={s['type']:<9} duration={s['duration_seconds']}s  caption={s.get('caption')!r}", flush=True)

        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=f"VideoScript: {len(scenes)} scenes ({presenter_count} presenter, {broll_count} b-roll), ~{total_duration}s")]),
        )


video_script_agent = VideoScriptAgent(name="VideoScriptAgent")
