"""
POST /api/generate

Accepts a PDF upload, saves it, creates a job, and fires the pipeline
as a FastAPI background task.

Returns: { job_id: str, status: "processing" }
"""

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile

from tools.auth import require_token
from tools.limiter import limiter
from tools.job_store import create_job
from tools.rate_limit import check_global_generate_limit
from tools.storage import save_upload

VALID_TONES = {"formal", "explanatory", "casual"}

router = APIRouter()


@router.post("/generate")
@limiter.limit("3/day")
async def generate(
    request: Request,
    file: UploadFile = File(...),
    tone: str = Form("explanatory"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _: None = Depends(require_token),
):
    allowed, remaining = check_global_generate_limit()
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Daily generation limit reached. Try again tomorrow.",
        )
    print(f"[generate] Global daily budget: {remaining} generations remaining today", flush=True)
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    tone = tone if tone in VALID_TONES else "explanatory"

    job_id = str(uuid.uuid4())
    data = await file.read()

    page_count = len(__import__("re").findall(rb"/Type\s*/Page[^s]", data))
    if page_count > 20:
        raise HTTPException(status_code=400, detail=f"PDF has {page_count} pages — limit is 20.")

    print(f"\n{'='*60}", flush=True)
    print(f"[generate] New job: {job_id}  tone={tone!r}", flush=True)
    print(f"[generate] File: {file.filename!r}  size: {len(data):,} bytes  pages: {page_count}", flush=True)

    file_path = save_upload(job_id, file.filename or "upload.pdf", data)
    print(f"[generate] Saved to: {file_path}", flush=True)
    create_job(job_id)

    from pipeline import run_pipeline
    background_tasks.add_task(run_pipeline, job_id, file_path, data, tone)
    print(f"[generate] Pipeline queued as background task", flush=True)

    return {"job_id": job_id, "status": "processing", "tone": tone}
