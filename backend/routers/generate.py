"""
POST /api/generate

Accepts a PDF upload, saves it locally, creates a job,
and launches the processing pipeline as a FastAPI background task.

Returns:
{
    "job_id": str,
    "status": "processing"
}
"""

import uuid

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from tools.job_store import create_job
from tools.storage import save_upload

router = APIRouter()


@router.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    # Create job
    job_id = str(uuid.uuid4())
    create_job(job_id)

    # Read uploaded file
    data = await file.read()

    # Save locally
    file_path = save_upload(
        job_id=job_id,
        filename=file.filename or "upload.pdf",
        data=data,
    )

    # Import here to avoid circular imports
    from pipeline import run_pipeline

    # Launch pipeline asynchronously
    background_tasks.add_task(
        run_pipeline,
        job_id,
        file_path,
    )

    return {
        "job_id": job_id,
        "status": "processing",
    }