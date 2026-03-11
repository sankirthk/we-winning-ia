from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import os

load_dotenv(override=True)

from tools.limiter import limiter
from routers import auth, generate, status, live, worker

app = FastAPI(title="DocuReel API", version="0.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "https://docureel.vercel.app",
        os.getenv("FRONTEND_URL", "https://docureel.vercel.app"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")
app.include_router(generate.router, prefix="/api")
app.include_router(status.router, prefix="/api")
app.include_router(live.router, prefix="/api")
app.include_router(worker.router)  # no prefix — /internal/run-pipeline

@app.get("/health")
async def health():
    return {"status": "ok"}
