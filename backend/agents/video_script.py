# VideoScriptAgent
# IN:  session["narration_script"] + session["tts_result"]
# OUT: session["video_script"] — { veo_prompts: [{ scene_id, start, end, prompt, style }] }
# Model: Gemini 2.0 Flash

import json
import re
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from tools.gemini import build_client, GEMINI_MODEL
from tools.job_store import update_job

STYLE_OPTIONS = [
    "cinematic corporate",
    "modern data visualization",
    "clean minimalist",
    "dynamic infographic",
    "professional documentary",
]

VIDEO_SCRIPT_PROMPT = """
You are a creative director generating video scene prompts for a TikTok-style explainer video.

Given a narration script and scene timing data, generate one Veo video prompt per scene.

Each prompt should visually represent the content being spoken during that scene.

Return ONLY a valid JSON array. No markdown, no explanation:

[
  {{
    "scene_id": "<scene_id>",
    "start": <start_s as float>,
    "end": <end_s as float>,
    "prompt": "<detailed visual description for Veo, 1-3 sentences, cinematic and specific>",
    "style": "<one of: cinematic corporate | modern data visualization | clean minimalist | dynamic infographic | professional documentary>"
  }}
]

Rules:
- The prompt must visually match what is being SAID during that time window
- Be specific: describe camera angle, subject, movement, lighting
- Keep prompts under 50 words each
- Choose style based on content type (data → data visualization, narrative → cinematic, etc.)
- Return ONLY the JSON array

Narration Script:
{narration_script}

Scene Timestamps:
{scene_timestamps}
"""


def _extract_json(text: str) -> list:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


class VideoScriptAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        job_id = ctx.session.state["job_id"]
        narration_script = ctx.session.state["narration_script"]
        tts_result = ctx.session.state["tts_result"]

        update_job(job_id, step="video_script")

        scene_timestamps = tts_result["scene_timestamps"]

        prompt = VIDEO_SCRIPT_PROMPT.format(
            narration_script=json.dumps(narration_script, indent=2),
            scene_timestamps=json.dumps(scene_timestamps, indent=2),
        )

        client = build_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt],
        )

        veo_prompts = _extract_json(response.text)

        video_script = {"veo_prompts": veo_prompts}
        ctx.session.state["video_script"] = video_script
        update_job(job_id, step="veo")

        yield Event(
            author=self.name,
            content=f"VideoScript done: {len(veo_prompts)} Veo prompts generated",
        )


video_script_agent = VideoScriptAgent(name="VideoScriptAgent")