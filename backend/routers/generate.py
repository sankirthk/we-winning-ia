"""
POST /api/generate

Accepts a PDF upload, saves it, creates a job, and fires the pipeline
as a FastAPI background task.

Returns: { job_id: str, status: "processing" }
"""

import uuid

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from tools.job_store import create_job
from tools.storage import save_upload

router = APIRouter()


@router.post("/generate")
async def generate(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    job_id = str(uuid.uuid4())
    data = await file.read()

    file_path = save_upload(job_id, file.filename or "upload.pdf", data)
    create_job(job_id)

    # Import here to avoid circular imports at module load time
    from pipeline import run_pipeline
    background_tasks.add_task(run_pipeline, job_id, file_path)

    return {"job_id": job_id, "status": "processing"}
