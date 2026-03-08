"""
Full end-to-end video generation test.
Runs the complete pipeline: Parser → VideoScript → Veo → Stitcher

Calls run_pipeline() directly — no server needed.
Progress is printed as each agent completes.

Usage:
  cd backend
  uv run python tests/test_video_pipeline.py path/to/report.pdf

Prerequisites:
  1. Run scripts/generate_avatars.py and set AVATAR_MALE_PATH / AVATAR_FEMALE_PATH in .env
  2. GCS bucket must exist (run scripts/setup_gcs.py if not)
  3. DEV_MODE=true for local output, DEV_MODE=false for GCS
"""

import asyncio
import sys
import uuid

from dotenv import load_dotenv
load_dotenv()

from pipeline import run_pipeline
from tools.job_store import create_job, get_job


def _divider(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


async def main(pdf_path: str):
    job_id = f"test-{uuid.uuid4().hex[:8]}"

    _divider(f"Starting pipeline — job {job_id}")
    print(f"  PDF: {pdf_path}")

    create_job(job_id)
    await run_pipeline(job_id=job_id, file_path=pdf_path)

    # Print final job state
    job = get_job(job_id)

    if job["status"] == "error":
        _divider("FAILED")
        print(f"  Error: {job.get('error')}")
        print(f"\n  Traceback:\n{job.get('traceback', '')}")
        sys.exit(1)

    _divider("DONE")

    manifest = job.get("manifest", {})
    print(f"  Title:       {manifest.get('title', 'n/a')}")
    print(f"  Sentiment:   {manifest.get('sentiment', 'n/a')}")

    veo_clips = job.get("veo_clips", [])
    print(f"  Clips:       {len(veo_clips)}")
    for clip in veo_clips:
        print(f"    Scene {clip['scene_id']} ({clip['type']}, {clip['duration_seconds']}s) → {clip['clip_path']}")

    print(f"\n  Final video: {job.get('final_video_uri', 'n/a')}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python tests/test_video_pipeline.py path/to/report.pdf")
        sys.exit(1)

    asyncio.run(main(sys.argv[1]))
