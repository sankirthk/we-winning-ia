"""
GET /api/status/{job_id}

Returns: { status, step, video_url }
"""

from fastapi import APIRouter, HTTPException

from tools.job_store import get_job

router = APIRouter()


@router.get("/status/{job_id}")
async def status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    context_ready = bool(job.get("manifest") or job.get("knowledge_base"))
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "step": job["step"],
        "video_url": job["video_url"],
        "error": job.get("error"),
        "context_ready": context_ready,
    }
