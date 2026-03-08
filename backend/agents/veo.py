# VeoAgent
# IN:  session["video_script"] — { scenes, avatar_male_path, avatar_female_path }
# OUT: session["veo_clips"]   — [{ scene_id, clip_path, duration_seconds, type, caption }]
# Model: Veo 3.0 (publishers/google/models/veo-3.0-generate-preview)
#
# Presenter scenes: reference avatar image + generateAudio=True (Veo bakes in
#                   lip-synced audio from the dialogue natively)
# B-roll scenes:    prompt only, no reference image, no audio

import asyncio
import time
from pathlib import Path
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from tools.gemini import build_client, VEO_MODEL
from tools.storage import save_upload
from tools.job_store import update_job

POLL_INTERVAL = 15  # seconds between operation status checks


def _load_avatar(path: str) -> types.Image | None:
    """Load avatar image bytes from local path. Returns None if path missing."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"  [VeoAgent] Warning: avatar not found at {path}")
        return None
    return types.Image(image_bytes=p.read_bytes(), mime_type="image/jpeg")


def _generate_clip(client, scene: dict, avatar_image: types.Image | None) -> bytes:
    """
    Blocking Veo generation + polling — called via asyncio.to_thread.
    Presenter scenes use the avatar as a reference image and enable native audio.
    B-roll scenes are silent with no reference image.
    """
    is_presenter = scene["type"] == "presenter"

    config = types.GenerateVideosConfig(
        aspect_ratio="9:16",
        duration_seconds=scene["duration_seconds"],
        number_of_videos=1,
        # Native audio generation — Veo lip-syncs the dialogue for presenter scenes
        generate_audio=is_presenter,
        # Reference image anchors the presenter's appearance across scenes
        reference_images=(
            [types.VideoGenerationReferenceImage(
                image=avatar_image,
                reference_type=types.VideoGenerationReferenceType.ASSET,
            )]
            if is_presenter and avatar_image
            else None
        ),
    )

    # For presenter scenes, append dialogue to the prompt so Veo knows what to say
    prompt = scene["prompt"]
    if is_presenter and scene.get("dialogue"):
        prompt = f'{prompt}\n\nSpoken dialogue: "{scene["dialogue"]}"'

    operation = client.models.generate_videos(
        model=VEO_MODEL,
        prompt=prompt,
        config=config,
    )

    # Poll until the operation completes
    while not operation.done:
        time.sleep(POLL_INTERVAL)
        operation = client.operations.get(operation)

    if operation.error:
        raise RuntimeError(f"Veo generation failed for scene {scene['scene_id']}: {operation.error}")

    # Download the generated video bytes
    generated_video = operation.result.generated_videos[0]
    video_bytes = client.files.download(file=generated_video.video)
    return bytes(video_bytes)


class VeoAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        job_id = ctx.session.state["job_id"]
        video_script = ctx.session.state["video_script"]

        update_job(job_id, step="veo")

        scenes = video_script["scenes"]
        client = build_client()

        # Load avatar images once — reused across all presenter scenes
        avatar_male = _load_avatar(video_script.get("avatar_male_path", ""))
        avatar_female = _load_avatar(video_script.get("avatar_female_path", ""))

        clips = []
        for scene in scenes:
            scene_id = scene["scene_id"]
            is_presenter = scene["type"] == "presenter"

            avatar_img = None
            if is_presenter:
                avatar_img = avatar_male if scene.get("avatar") == "male" else avatar_female

            print(f"  [VeoAgent] Generating scene {scene_id} ({scene['type']}, {scene['duration_seconds']}s)...")

            # Offload blocking generation + polling to thread pool
            video_bytes = await asyncio.to_thread(_generate_clip, client, scene, avatar_img)

            clip_path = save_upload(job_id, f"clip_{scene_id:02d}.mp4", video_bytes)
            clips.append({
                "scene_id": scene_id,
                "clip_path": clip_path,
                "duration_seconds": scene["duration_seconds"],
                "type": scene["type"],
                "caption": scene.get("caption", ""),
            })

            print(f"  [VeoAgent] Scene {scene_id} saved → {clip_path}")

        ctx.session.state["veo_clips"] = clips
        update_job(job_id, step="stitching", veo_clips=clips)

        total_duration = sum(c["duration_seconds"] for c in clips)
        yield Event(
            author=self.name,
            content=f"Veo done: {len(clips)} clips, ~{total_duration}s total",
        )


veo_agent = VeoAgent(name="VeoAgent")
