import json
import subprocess
from pathlib import Path
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from tools.job_store import update_job
from tools.storage import ensure_local_dir, file_url_for_local_path, save_text_local


def _text_event(author: str, text: str) -> Event:
    return Event(
        author=author,
        content=types.Content(
            role="model",
            parts=[types.Part(text=text)],
        ),
    )


class StitcherAgent(BaseAgent):
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        job_id = ctx.session.state["job_id"]
        veo_clips = ctx.session.state.get("veo_clips", [])

        update_job(job_id, step="stitching")
        ensure_local_dir("final")

        print(f"[StitcherAgent] ▶ Starting — veo_clips={len(veo_clips)}")

        if not veo_clips:
            update_job(
                job_id,
                status="error",
                step="failed",
                error="No generated clips available for stitching.",
            )
            raise ValueError("StitcherAgent: no generated clips available for stitching.")

        # Debug artifact only: save what would be stitched
        stitch_plan_path = save_text_local(
            "final",
            f"{job_id}_stitch_plan.json",
            json.dumps(veo_clips, indent=2),
        )

        print(f"[StitcherAgent]   wrote stitch plan: {stitch_plan_path}")

        final_video_path = _stitch_clips_to_mp4(job_id, veo_clips)
        final_video_uri = file_url_for_local_path(final_video_path)
        ctx.session.state["final_video_uri"] = final_video_uri

        update_job(
            job_id,
            status="done",
            step="complete",
            video_url=final_video_uri,
            final_video_path=final_video_path,
            stitch_plan_path=stitch_plan_path,
        )

        print(f"[StitcherAgent] ✅ Done — final MP4: {final_video_uri}")

        yield _text_event(
            self.name,
            f"Stitched {len(veo_clips)} clips into final MP4.",
        )


stitcher_agent = StitcherAgent(name="StitcherAgent")


def _stitch_clips_to_mp4(job_id: str, veo_clips: list[dict]) -> str:
    final_dir = ensure_local_dir("final")
    output_path = final_dir / f"{job_id}_final.mp4"
    concat_list_path = final_dir / f"{job_id}_concat.txt"

    clip_paths: list[Path] = []
    for clip in sorted(veo_clips, key=lambda c: c.get("scene_id", 0)):
        clip_path = clip.get("clip_uri") or clip.get("clip_path")
        if not clip_path:
            raise ValueError(f"StitcherAgent: missing clip path for scene {clip.get('scene_id')}.")
        path = Path(clip_path)
        if not path.exists():
            raise FileNotFoundError(f"StitcherAgent: clip not found: {clip_path}")
        clip_paths.append(path)

    concat_lines = []
    for path in clip_paths:
        escaped = str(path).replace("'", "'\\''")
        concat_lines.append(f"file '{escaped}'")
    concat_list_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    cmd_copy = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
        "-c",
        "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd_copy, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[StitcherAgent]   stitched via stream copy: {output_path}")
        return str(output_path)

    copy_error = result.stderr
    print("[StitcherAgent]   stream-copy concat failed, retrying with re-encode...")
    cmd_reencode = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        str(output_path),
    ]
    result = subprocess.run(cmd_reencode, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "StitcherAgent: ffmpeg stitching failed.\n"
            f"copy_error={copy_error[-1200:]}\n"
            f"reencode_error={result.stderr[-1200:]}"
        )

    print(f"[StitcherAgent]   stitched via re-encode: {output_path}")
    return str(output_path)
