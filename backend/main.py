import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from routers import generate, live, status
from tools.job_store import create_job, update_job
from fastapi.staticfiles import StaticFiles

TEST_JOB_ID = "test-live-job"

create_job(TEST_JOB_ID)
update_job(
    TEST_JOB_ID,
    manifest={
        "title": "Kernel Fusion Paper",
        "overall_summary": "Naive fusion caused 30x slowdown.",
    },
    knowledge_base={
        "document_title": "Kernel Fusion Paper",
        "deep_findings": ["Naive fusion collapsed memory bandwidth"],
    },
    status="done",
)

print("TEST JOB_ID:", TEST_JOB_ID)
print("GOOGLE_CLOUD_PROJECT:", os.getenv("GOOGLE_CLOUD_PROJECT"))
print("GOOGLE_CLOUD_LOCATION:", os.getenv("GOOGLE_CLOUD_LOCATION"))
print("GOOGLE_GENAI_USE_VERTEXAI:", os.getenv("GOOGLE_GENAI_USE_VERTEXAI"))
print("LIVE_MODEL:", os.getenv("LIVE_MODEL"))
print("GOOGLE_APPLICATION_CREDENTIALS:", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

app = FastAPI(title="NeverRTFM API", version="0.1.0")

app.mount("/storage", StaticFiles(directory="storage"), name="storage")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate.router, prefix="/api")
app.include_router(status.router, prefix="/api")
app.include_router(live.router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/debug/live-env")
async def debug_live_env():
    return {
        "project": os.getenv("GOOGLE_CLOUD_PROJECT"),
        "location": os.getenv("GOOGLE_CLOUD_LOCATION"),
        "use_vertex": os.getenv("GOOGLE_GENAI_USE_VERTEXAI"),
        "live_model": os.getenv("LIVE_MODEL"),
        "creds_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    }