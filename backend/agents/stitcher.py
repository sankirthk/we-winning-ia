# StitcherAgent
# IN:  session["veo_clips"] — [{ scene_id, clip_path, duration_seconds, type, caption }]
# OUT: session["final_video_uri"] — signed GCS URL (prod) or local path (dev)
# Tool: ffmpeg
#
# Steps:
#   1. Download each clip to a temp directory (handles both local paths and gs:// URIs)
#   2. Concatenate with -c copy (no re-encode — stream copy is near-instant)
#   4. Upload final video to GCS and return a signed URL
#
# Captions are passed through in the job metadata for the frontend to overlay,
# not burned in — burning required libx264 re-encoding which took 15+ min on
# Cloud Run's limited CPU.

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


def _concat_clips(clip_paths: list[Path], output_path: Path) -> None:
    """Concatenate clips using ffmpeg concat demuxer with stream copy — no re-encode."""
    concat_file = output_path.parent / "concat.txt"
    with open(concat_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed:\n{result.stderr}")


def _stitch(clips: list[dict], pdf_hash: str, tone: str = "explanatory") -> str:
    """
    Download clips, concat with stream copy (no re-encode), upload.
    Returns the final video URI (saved under cache/{pdf_hash}/final_{tone}.mp4).
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Step 1: download all clips
        clip_paths = []
        for clip in clips:
            dest = tmp_dir / f"clip_{clip['scene_id']:02d}.mp4"
            print(f"  [StitcherAgent] Downloading scene {clip['scene_id']}...", flush=True)
            _download_clip(clip["clip_path"], dest)
            clip_paths.append(dest)
            print(f"  [StitcherAgent] Downloaded scene {clip['scene_id']} ({dest.stat().st_size:,} bytes)", flush=True)

        # Step 2: concat with -c copy (seconds, not minutes)
        final_path = tmp_dir / "final.mp4"
        _concat_clips(clip_paths, final_path)
        print(f"  [StitcherAgent] Concatenated → {final_path} ({final_path.stat().st_size:,} bytes)", flush=True)

        # Step 3: save to content-addressed cache path (tone-scoped)
        final_bytes = final_path.read_bytes()
        uri = save_hash_bytes(pdf_hash, f"final_{tone}.mp4", final_bytes)
        print(f"  [StitcherAgent] Saved → {uri}", flush=True)

        return uri


class StitcherAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        job_id = ctx.session.state["job_id"]
        pdf_hash = ctx.session.state["pdf_hash"]
        tone = ctx.session.state.get("tone", "explanatory")
        clips = ctx.session.state["veo_clips"]

        update_job(job_id, step="stitching")
        print(f"\n[StitcherAgent] ▶ Starting — {len(clips)} clips  tone={tone!r}", flush=True)
        for c in clips:
            print(f"  clip {c['scene_id']:2d}: {c['clip_path']}", flush=True)

        final_uri = await asyncio.to_thread(_stitch, clips, pdf_hash, tone)

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
