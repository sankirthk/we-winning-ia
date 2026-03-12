import json
import asyncio
import mimetypes
import os
import time
from pathlib import Path
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.cloud import storage as gcs
from google.genai import types

from tools.gemini import VEO_MODEL, build_veo_client
from tools.job_store import update_job
from tools.storage import ensure_local_dir, save_text_local


def _text_event(author: str, text: str) -> Event:
    return Event(
        author=author,
        content=types.Content(
            role="model",
            parts=[types.Part(text=text)],
        ),
    )


class VeoAgent(BaseAgent):
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        job_id = ctx.session.state["job_id"]
        video_script = ctx.session.state.get("video_script")

        if not video_script:
            raise ValueError("VeoAgent: missing video_script in session state.")

        update_job(job_id, step="rendering")

        scenes = video_script.get("scenes", [])
        ensure_local_dir("veo")
        max_concurrency = max(1, int(os.getenv("VEO_MAX_CONCURRENCY", "2")))
        semaphore = asyncio.Semaphore(max_concurrency)
        avatar_male_path = video_script.get("avatar_male_path") or os.getenv("AVATAR_MALE_PATH", "")
        avatar_female_path = video_script.get("avatar_female_path") or os.getenv("AVATAR_FEMALE_PATH", "")

        print(
            f"[VeoAgent] ▶ Starting — {len(scenes)} scenes "
            f"(max_concurrency={max_concurrency})"
        )

        async def _generate_scene(scene: dict) -> dict:
            scene_id = scene["scene_id"]
            prompt_text = scene.get("prompt", f"Scene {scene_id}")
            duration = scene.get("duration_seconds", 4)

            prompt_file = save_text_local(
                "veo",
                f"{job_id}_scene_{scene_id}.txt",
                prompt_text,
            )

            async with semaphore:
                print(
                    f"[VeoAgent]   scene={scene_id} generating video...",
                    flush=True,
                )
                clip_uri = await asyncio.to_thread(
                    _generate_and_save_scene_video,
                    job_id,
                    scene_id,
                    prompt_text,
                    duration,
                    scene.get("type"),
                    scene.get("dialogue"),
                    avatar_male_path if scene.get("avatar") == "male" else avatar_female_path,
                )

            clip_obj = {
                "scene_id": scene_id,
                "clip_uri": clip_uri,
                "prompt_uri": prompt_file,
                "duration_seconds": duration,
                "type": scene.get("type"),
                "caption": scene.get("caption"),
            }
            print(
                f"[VeoAgent]   scene={scene_id} type={clip_obj['type']} "
                f"duration={duration}s file={clip_uri}"
            )
            return clip_obj

        tasks = [asyncio.create_task(_generate_scene(scene)) for scene in scenes]
        veo_clips = await asyncio.gather(*tasks)
        veo_clips.sort(key=lambda clip: clip["scene_id"])

        ctx.session.state["veo_clips"] = veo_clips

        # Persist to job store too, so status/debug endpoints can inspect it
        update_job(
            job_id,
            step="stitching",
            veo_clips=veo_clips,
        )

        # Also save a JSON manifest locally for debugging
        save_text_local(
            "veo",
            f"{job_id}_veo_clips.json",
            json.dumps(veo_clips, indent=2),
        )

        print(f"[VeoAgent] ✅ Done — {len(veo_clips)} video clips")

        yield _text_event(
            self.name,
            f"Generated {len(veo_clips)} scene video clips.",
        )


veo_agent = VeoAgent(name="VeoAgent")


def _generate_and_save_scene_video(
    job_id: str,
    scene_id: int,
    prompt: str,
    duration_seconds: int,
    scene_type: str | None,
    dialogue: str | None,
    avatar_path: str | None,
) -> str:
    return _generate_and_save_scene_video_with_mode(
        job_id=job_id,
        scene_id=scene_id,
        prompt=prompt,
        duration_seconds=duration_seconds,
        scene_type=scene_type,
        dialogue=dialogue,
        avatar_path=avatar_path,
        use_reference_image=True,
    )


def _generate_and_save_scene_video_with_mode(
    job_id: str,
    scene_id: int,
    prompt: str,
    duration_seconds: int,
    scene_type: str | None,
    dialogue: str | None,
    avatar_path: str | None,
    use_reference_image: bool,
) -> str:
    timeout_seconds = int(os.getenv("VEO_SCENE_TIMEOUT_SECONDS", "900"))
    poll_interval_seconds = int(os.getenv("VEO_POLL_INTERVAL_SECONDS", "3"))
    progress_log_interval = int(os.getenv("VEO_PROGRESS_LOG_INTERVAL_SECONDS", "30"))

    client = build_veo_client()
    is_presenter = scene_type == "presenter"
    default_aspect_ratio = os.getenv("VEO_ASPECT_RATIO", "16:9")
    aspect_ratio = "16:9" if use_reference_image else default_aspect_ratio
    effective_duration_seconds = 8 if use_reference_image else duration_seconds
    prompt_for_veo = prompt
    if is_presenter and dialogue:
        prompt_for_veo = f'{prompt}\n\nSpoken dialogue: "{dialogue}"'

    config = types.GenerateVideosConfig(
        number_of_videos=1,
        duration_seconds=effective_duration_seconds,
        aspect_ratio=aspect_ratio,
        generate_audio=bool(is_presenter and dialogue),
        output_gcs_uri=os.getenv("VEO_OUTPUT_GCS_URI"),
    )
    avatar_image = (
        _load_avatar_image(avatar_path)
        if is_presenter and use_reference_image
        else None
    )
    if is_presenter and use_reference_image and avatar_image is None:
        raise ValueError(
            f"Missing/invalid avatar image for presenter scene {scene_id}. "
            f"avatar_path={avatar_path!r}"
        )
    if avatar_image is not None:
        config.reference_images = [
            types.VideoGenerationReferenceImage(
                image=avatar_image,
                reference_type=types.VideoGenerationReferenceType.ASSET,
            )
        ]

    operation = client.models.generate_videos(
        model=VEO_MODEL,
        prompt=prompt_for_veo,
        config=config,
    )

    started = time.time()
    last_progress_log = started
    while not operation.done:
        now = time.time()
        if now - started > timeout_seconds:
            raise TimeoutError(
                f"Veo timed out for scene {scene_id} after {timeout_seconds}s."
            )
        if now - last_progress_log >= progress_log_interval:
            print(
                f"[VeoAgent]   scene={scene_id} still generating... "
                f"{int(now - started)}s elapsed",
                flush=True,
            )
            last_progress_log = now
        time.sleep(poll_interval_seconds)
        operation = client.operations.get(operation)

    if operation.error:
        error_text = str(operation.error)
        allow_no_ref_fallback = (
            os.getenv("VEO_ALLOW_NO_REFERENCE_FALLBACK", "false").lower() == "true"
        )
        if (
            use_reference_image
            and allow_no_ref_fallback
            and "not supported by this model" in error_text.lower()
        ):
            print(
                f"[VeoAgent]   scene={scene_id} model rejected reference image; "
                "retrying without reference image...",
                flush=True,
            )
            return _generate_and_save_scene_video_with_mode(
                job_id=job_id,
                scene_id=scene_id,
                prompt=prompt,
                duration_seconds=duration_seconds,
                scene_type=scene_type,
                dialogue=dialogue,
                avatar_path=avatar_path,
                use_reference_image=False,
            )
        raise RuntimeError(
            f"Veo generation failed for scene {scene_id}: {operation.error}"
        )

    result = operation.result
    if not result or not result.generated_videos:
        rai_count = getattr(result, "rai_media_filtered_count", None) if result else None
        rai_reasons = getattr(result, "rai_media_filtered_reasons", None) if result else None
        op_name = getattr(operation, "name", None)
        op_done = getattr(operation, "done", None)
        op_metadata = getattr(operation, "metadata", None)
        print(
            f"[VeoAgent]   scene={scene_id} empty output. "
            f"op_name={op_name} done={op_done} "
            f"rai_media_filtered_count={rai_count} "
            f"rai_media_filtered_reasons={rai_reasons} "
            f"metadata={op_metadata}",
            flush=True,
        )
        raise RuntimeError(f"Veo returned no generated videos for scene {scene_id}.")

    video = result.generated_videos[0].video
    if video is None:
        raise RuntimeError(f"Veo returned an empty video object for scene {scene_id}.")

    filename = f"{job_id}_scene_{scene_id}.mp4"
    out_path = ensure_local_dir("veo") / filename

    if video.video_bytes:
        out_path.write_bytes(video.video_bytes)
        return str(out_path)

    if video.uri:
        if video.uri.startswith("gs://"):
            _download_gcs_uri(video.uri, out_path)
            return str(out_path)
        raise RuntimeError(
            f"Unsupported Veo video URI for scene {scene_id}: {video.uri}"
        )

    raise RuntimeError(f"Veo returned no bytes/URI for scene {scene_id}.")


def _download_gcs_uri(gcs_uri: str, target_path: Path) -> None:
    # gs://bucket/path/to/file.mp4
    uri_body = gcs_uri.removeprefix("gs://")
    bucket_name, blob_name = uri_body.split("/", 1)
    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(str(target_path))


def _load_avatar_image(avatar_path: str | None) -> types.Image | None:
    if not avatar_path:
        return None

    mime, _ = mimetypes.guess_type(avatar_path)
    mime_type = mime or "image/png"

    if avatar_path.startswith("gs://"):
        uri_body = avatar_path.removeprefix("gs://")
        bucket_name, blob_name = uri_body.split("/", 1)
        client = gcs.Client()
        data = client.bucket(bucket_name).blob(blob_name).download_as_bytes()
        return types.Image(image_bytes=data, mime_type=mime_type)

    path = Path(avatar_path)
    if not path.is_absolute():
        # Resolve relative paths from backend root, not process cwd.
        path = (Path(__file__).resolve().parent.parent / path).resolve()
    if not path.exists():
        print(f"[VeoAgent]   avatar path not found: {path}", flush=True)
        return None
    return types.Image(image_bytes=path.read_bytes(), mime_type=mime_type)
