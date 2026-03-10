# VeoAgent
# IN:  session["video_script"] — { scenes, avatar_male_path, avatar_female_path }
# OUT: session["veo_clips"]   — [{ scene_id, clip_path, duration_seconds, caption }]
# Model: Veo 3.1 Fast (reference_to_video) for all scenes
#
# Every scene is a presenter scene — one continuous presenter on screen,
# background changes per scene. Avatar reference image anchors appearance.
# Dialogue is appended to the prompt so Veo bakes in lip-synced audio.
# reference_to_video requires exactly 8s — shorter durations are clamped.

import asyncio
import os
import time
from pathlib import Path
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from tools.gemini import build_veo_client, VEO_MODEL, VEO_FAST_MODEL
from tools.storage import save_hash_bytes, build_gcs_client
from tools.job_store import update_job

POLL_INTERVAL = 15  # seconds between operation status checks

# Max simultaneous Veo operations. Veo 3 preview quota is ~2 concurrent requests.
# Raise to 3-4 if your project has higher quota; lower to 1 to avoid 429s.
MAX_CONCURRENT_VEO = int(os.getenv("MAX_CONCURRENT_VEO", "2"))

# Retry config for transient Veo errors (code 8 = resource exhausted / high load)
VEO_MAX_RETRIES = int(os.getenv("VEO_MAX_RETRIES", "4"))
VEO_RETRY_BASE_DELAY = float(os.getenv("VEO_RETRY_BASE_DELAY", "30"))  # seconds
_VEO_RETRYABLE_CODES = {8}  # gRPC RESOURCE_EXHAUSTED


def _load_avatar(path: str) -> types.Image | None:
    """Load avatar image bytes from local path or GCS URI. Returns None if missing."""
    if not path:
        return None
    if path.startswith("gs://"):
        try:
            # gs://bucket/object/path
            without_scheme = path[5:]
            bucket_name, _, blob_name = without_scheme.partition("/")
            client = build_gcs_client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            image_bytes = blob.download_as_bytes()
            return types.Image(image_bytes=image_bytes, mime_type="image/jpeg")
        except Exception as e:
            print(f"  [VeoAgent] Warning: could not download avatar from {path}: {e}")
            return None
    p = Path(path)
    if not p.exists():
        print(f"  [VeoAgent] Warning: avatar not found at {path}")
        return None
    return types.Image(image_bytes=p.read_bytes(), mime_type="image/jpeg")


def _generate_clip(client, scene: dict, avatar_image: types.Image | None, job_id: str) -> bytes:
    """
    Blocking Veo generation + polling — called via asyncio.to_thread.
    All scenes are presenter scenes: avatar reference image + dialogue audio.
    reference_to_video only supports 8s — clamp anything shorter.
    """
    # Build prompt: prepend reference-character instruction, append dialogue
    prompt = scene["prompt"]
    if scene.get("dialogue"):
        prompt = f'{prompt}\n\nSpoken dialogue: "{scene["dialogue"]}"'

    duration = scene["duration_seconds"]
    # reference_to_video only supports 8s clips — clamp if shorter
    if duration < 8:
        print(f"    [Veo] Clamping duration {duration}s → 8s (reference_to_video minimum)", flush=True)
        duration = 8

    reference_images = None
    model = VEO_MODEL
    if avatar_image is not None:
        reference_images = [
            types.VideoGenerationReferenceImage(
                image=avatar_image,
                reference_type=types.VideoGenerationReferenceType.ASSET,
            )
        ]
        model = VEO_FAST_MODEL
        prompt = f"The 3D animated character from the reference image presents to camera. {prompt}"
        print(f"    [Veo] Using avatar reference image (ASSET) → {VEO_FAST_MODEL}", flush=True)
    else:
        print(f"    [Veo] No avatar image — falling back to {VEO_MODEL} (no reference)", flush=True)

    config = types.GenerateVideosConfig(
        aspect_ratio="9:16",
        duration_seconds=duration,
        number_of_videos=1,
        reference_images=reference_images,
    )

    print(f"    [Veo] model={model}  aspect=9:16  duration={duration}s", flush=True)
    print(f"    [Veo] Prompt: {prompt[:120]!r}", flush=True)

    last_error = None
    for attempt in range(VEO_MAX_RETRIES + 1):
        if attempt > 0:
            delay = VEO_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"    [Veo] Retry {attempt}/{VEO_MAX_RETRIES} for scene {scene['scene_id']} — waiting {delay:.0f}s...", flush=True)
            time.sleep(delay)

        print(f"    [Veo] Submitting (attempt {attempt + 1})...", flush=True)
        operation = client.models.generate_videos(
            model=model,
            prompt=prompt,
            config=config,
        )
        print(f"    [Veo] Operation started: {getattr(operation, 'name', '?')}", flush=True)

        # Poll until the operation completes
        poll_count = 0
        while not operation.done:
            time.sleep(POLL_INTERVAL)
            poll_count += 1
            operation = client.operations.get(operation)
            print(f"    [Veo] Polling ({poll_count * POLL_INTERVAL}s elapsed) done={operation.done}", flush=True)

        if operation.error:
            err = operation.error
            err_code = err.get("code") if isinstance(err, dict) else getattr(err, "code", None)
            print(f"    [Veo] ❌ Generation error (code={err_code}): {err}", flush=True)
            if err_code in _VEO_RETRYABLE_CODES and attempt < VEO_MAX_RETRIES:
                last_error = err
                continue  # retry
            raise RuntimeError(f"Veo generation failed for scene {scene['scene_id']}: {err}")

        last_error = None
        break  # success

    if last_error is not None:
        raise RuntimeError(f"Veo generation failed after {VEO_MAX_RETRIES} retries for scene {scene['scene_id']}: {last_error}")

    # Download the generated video bytes.
    generated_video = operation.result.generated_videos[0]
    video = generated_video.video

    video_uri = getattr(video, "uri", None) if video else None
    video_bytes_inline = getattr(video, "video_bytes", None) if video else None
    print(f"    [Veo] uri={video_uri!r}  video_bytes={'len=' + str(len(video_bytes_inline)) if video_bytes_inline else None}", flush=True)

    if video_uri:
        gcs = build_gcs_client()
        without_prefix = video_uri.removeprefix("gs://")
        bucket_name, blob_path = without_prefix.split("/", 1)
        return gcs.bucket(bucket_name).blob(blob_path).download_as_bytes()

    if video_bytes_inline:
        return bytes(video_bytes_inline)

    raise RuntimeError(f"Veo returned no URI and no video_bytes for scene {scene['scene_id']}. Raw video obj: {video}")


class VeoAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        job_id = ctx.session.state["job_id"]
        pdf_hash = ctx.session.state["pdf_hash"]
        video_script = ctx.session.state["video_script"]

        update_job(job_id, step="veo")
        scenes = video_script["scenes"]  # pipeline pre-filters to missing scenes only
        existing_clips: list[dict] = ctx.session.state.get("existing_clips", [])

        print(f"\n[VeoAgent] ▶ Starting — {len(scenes)} scenes to generate, {len(existing_clips)} already cached", flush=True)
        print(f"[VeoAgent]   avatar_male_path={video_script.get('avatar_male_path')!r}", flush=True)
        print(f"[VeoAgent]   avatar_female_path={video_script.get('avatar_female_path')!r}", flush=True)

        client = build_veo_client()

        # Load avatar images once — reused across all presenter scenes
        avatar_male = _load_avatar(video_script.get("avatar_male_path", ""))
        avatar_female = _load_avatar(video_script.get("avatar_female_path", ""))
        print(f"[VeoAgent]   avatar_male loaded={avatar_male is not None}  avatar_female loaded={avatar_female is not None}", flush=True)

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_VEO)
        print(f"[VeoAgent]   Parallelism: {MAX_CONCURRENT_VEO} concurrent Veo requests", flush=True)

        async def generate_scene(scene: dict) -> dict:
            scene_id = scene["scene_id"]
            avatar_img = avatar_male if scene.get("avatar") == "male" else avatar_female

            print(f"[VeoAgent]   scene {scene_id}/{len(scenes)}: avatar={scene.get('avatar')!r}  duration={scene['duration_seconds']}s", flush=True)
            print(f"[VeoAgent]   prompt: {scene['prompt'][:100]}...", flush=True)

            async with semaphore:
                print(f"[VeoAgent]   scene {scene_id} → slot acquired, submitting to Veo", flush=True)
                video_bytes = await asyncio.to_thread(_generate_clip, client, scene, avatar_img, job_id)

            print(f"[VeoAgent]   scene {scene_id} ✅ {len(video_bytes):,} bytes", flush=True)
            clip_path = save_hash_bytes(pdf_hash, f"clips/clip_{scene_id:02d}.mp4", video_bytes)
            return {
                "scene_id": scene_id,
                "clip_path": clip_path,
                "duration_seconds": scene["duration_seconds"],
                "caption": scene.get("caption", ""),
            }

        # Fire all missing scenes concurrently (capped by semaphore)
        new_clips = await asyncio.gather(*[generate_scene(s) for s in scenes])

        # Merge with pre-existing clips and sort by scene_id
        all_clips = list(existing_clips) + list(new_clips)
        clips = sorted(all_clips, key=lambda c: c["scene_id"])

        ctx.session.state["veo_clips"] = clips
        update_job(job_id, step="stitching", veo_clips=clips)

        total_duration = sum(c["duration_seconds"] for c in clips)
        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=f"Veo done: {len(clips)} clips, ~{total_duration}s total")]),
        )


veo_agent = VeoAgent(name="VeoAgent")
