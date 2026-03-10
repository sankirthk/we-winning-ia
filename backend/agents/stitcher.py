# StitcherAgent
# IN:  session["veo_clips"] — [{ scene_id, clip_path, duration_seconds, type, caption }]
# OUT: session["final_video_uri"] — signed GCS URL (prod) or local path (dev)
# Tool: ffmpeg
#
# Steps:
#   1. Download each clip to a temp directory (handles both local paths and gs:// URIs)
#   2. Burn caption onto each clip via ffmpeg drawtext
#   3. Concatenate all captioned clips into a single MP4
#   4. Upload final video to GCS and return a signed URL

import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from tools.storage import save_hash_bytes, get_signed_url, DEV_MODE, build_gcs_client
from tools.job_store import update_job


def _download_clip(clip_path: str, dest: Path) -> None:
    """Copy from local path or download from GCS to dest."""
    if clip_path.startswith("gs://"):
        client = build_gcs_client()
        without_prefix = clip_path.removeprefix("gs://")
        bucket_name, blob_path = without_prefix.split("/", 1)
        client.bucket(bucket_name).blob(blob_path).download_to_filename(str(dest))
    else:
        shutil.copy2(clip_path, dest)


def _escape_drawtext(text: str) -> str:
    """Escape special characters for ffmpeg drawtext filter."""
    return (
        text
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _burn_caption(input_path: Path, output_path: Path, caption: str) -> None:
    """
    Burn caption text onto a clip using ffmpeg drawtext.
    Positioned at bottom center with a semi-transparent black box.
    If no caption, just copies the file.
    """
    if not caption:
        shutil.copy2(input_path, output_path)
        return

    escaped = _escape_drawtext(caption)
    vf = (
        f"drawtext=text='{escaped}'"
        ":fontsize=52"
        ":fontcolor=white"
        ":x=(w-text_w)/2"
        ":y=h-200"
        ":box=1"
        ":boxcolor=black@0.55"
        ":boxborderw=18"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "copy",   # preserve Veo-generated audio untouched
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg caption burn failed for {input_path.name}:\n{result.stderr}")


def _concat_clips(captioned_paths: list[Path], output_path: Path) -> None:
    """Concatenate captioned clips using ffmpeg concat demuxer."""
    concat_file = output_path.parent / "concat.txt"
    with open(concat_file, "w") as f:
        for p in captioned_paths:
            f.write(f"file '{p.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",   # clips are already h264 from caption burn step
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed:\n{result.stderr}")


def _stitch(clips: list[dict], pdf_hash: str) -> str:
    """
    Full stitch pipeline — runs in a thread via asyncio.to_thread.
    Returns the final video URI (saved under cache/{pdf_hash}/final.mp4).
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Step 1: download all clips
        raw_paths = []
        for clip in clips:
            dest = tmp_dir / f"raw_{clip['scene_id']:02d}.mp4"
            _download_clip(clip["clip_path"], dest)
            raw_paths.append((clip, dest))
            print(f"  [StitcherAgent] Downloaded scene {clip['scene_id']}")

        # Step 2: burn captions onto each clip
        captioned_paths = []
        for clip, raw_path in raw_paths:
            captioned = tmp_dir / f"captioned_{clip['scene_id']:02d}.mp4"
            _burn_caption(raw_path, captioned, clip.get("caption", ""))
            captioned_paths.append(captioned)
            print(f"  [StitcherAgent] Caption burned for scene {clip['scene_id']}")

        # Step 3: concatenate
        final_path = tmp_dir / "final.mp4"
        _concat_clips(captioned_paths, final_path)
        print(f"  [StitcherAgent] Clips concatenated → {final_path}")

        # Step 4: save to content-addressed cache path
        final_bytes = final_path.read_bytes()
        uri = save_hash_bytes(pdf_hash, "final.mp4", final_bytes)
        print(f"  [StitcherAgent] Saved → {uri}")

        return uri


class StitcherAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        job_id = ctx.session.state["job_id"]
        pdf_hash = ctx.session.state["pdf_hash"]
        clips = ctx.session.state["veo_clips"]

        update_job(job_id, step="stitching")
        print(f"\n[StitcherAgent] ▶ Starting — {len(clips)} clips to stitch", flush=True)
        for c in clips:
            print(f"  clip {c['scene_id']:2d}: {c['clip_path']}", flush=True)

        # Run ffmpeg work in thread pool — subprocess calls block the event loop
        final_uri = await asyncio.to_thread(_stitch, clips, pdf_hash)

        # In prod, return a signed URL so the frontend can stream directly from GCS
        if not DEV_MODE and final_uri.startswith("gs://"):
            final_uri = get_signed_url(final_uri)

        print(f"[StitcherAgent] ✅ Done — final_uri: {final_uri}", flush=True)
        ctx.session.state["final_video_uri"] = final_uri
        update_job(job_id, step="complete", final_video_uri=final_uri)

        total_duration = sum(c["duration_seconds"] for c in clips)
        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=f"Stitch complete: {len(clips)} clips, ~{total_duration}s → {final_uri}")]),
        )


stitcher_agent = StitcherAgent(name="StitcherAgent")
