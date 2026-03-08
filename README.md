# NeverRTFM

> Upload any report. Get a 30–60 second TikTok-style video. Ask questions live.

---

## Prerequisites

- Python 3.13+
- [uv](https://astral.sh/uv) — fast Python package manager
- [Docker](https://docs.docker.com/get-docker/) + Docker Compose (optional)

---

## GCP Setup

### 1. Create a project

Go to [console.cloud.google.com](https://console.cloud.google.com) and create a project, or use an existing one. Note the **Project ID** (not the name).

### 2. Enable APIs

In the GCP Console → **APIs & Services → Enable APIs**, enable:

| API | Used by |
|-----|---------|
| Vertex AI API | Gemini 3.1 Pro (parser, script), Veo 2 (video) |
| Cloud Text-to-Speech API | TTSAgent |
| Cloud Storage API | File storage in prod |
| Cloud Document AI API | Optional — only if `PARSER_BACKEND=documentai` |

### 3. Get credentials

**Option A — Vertex AI Express API Key (recommended for local dev, no gcloud needed)**

1. Go to [console.cloud.google.com/apis/credentials](https://console.cloud.google.com/apis/credentials)
2. Click **Create Credentials → API Key**
3. Restrict it to the APIs above
4. Copy the key → set as `GOOGLE_VERTEX_API_KEY` in `backend/.env`

**Option B — Application Default Credentials (for Cloud Run / service accounts)**

```bash
# Install gcloud CLI: https://cloud.google.com/sdk/docs/install
gcloud auth application-default login
```

This writes credentials to `~/.config/gcloud/application_default_credentials.json`. No env var needed — the SDK picks it up automatically. For Cloud Run, attach a service account to the instance instead.

---

## Environment Variables

Copy and fill in `backend/.env`:

```env
# ── Google Cloud ──────────────────────────────────────────
GOOGLE_CLOUD_PROJECT=your-project-id        # GCP Project ID (not the display name)
GOOGLE_CLOUD_LOCATION=us-central1           # Vertex AI region

# Auth — choose ONE:
GOOGLE_VERTEX_API_KEY=your-api-key          # Option A: API key (local dev)
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json  # Option B: service account

# ── Storage ───────────────────────────────────────────────
DEV_MODE=true          # true = files saved locally under backend/local_storage/
GCS_BUCKET=nevertrtfm  # only used when DEV_MODE=false

# ── Parser backend ────────────────────────────────────────
PARSER_BACKEND=gemini  # "gemini" (default) or "documentai"
DOCUMENT_AI_PROCESSOR_ID=your-processor-id  # only needed if PARSER_BACKEND=documentai

# ── CORS ──────────────────────────────────────────────────
FRONTEND_URL=http://localhost:3000
```

> **Never commit `.env`** — it's gitignored. Only `.env.example` should be committed.

---

## Local Dev Setup

```bash
git clone <repo-url>
cd we-winning-ia

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
cd backend
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e .

# Run
uvicorn main:app --reload --port 8080
```

API: `http://localhost:8080`
Swagger: `http://localhost:8080/docs`

---

## Running with Docker

```bash
# From repo root
docker compose up --build

# With hot reload
docker compose watch
```

---

## Project Structure

```
backend/
├── main.py           — FastAPI app, CORS, router registration
├── pipeline.py       — ADK SequentialAgent root, run_pipeline()
├── agents/
│   ├── parser.py           — PDF → manifest (Gemini 3.1 Pro)
│   ├── narrative_script.py — manifest → script (Gemini 3.1 Pro)
│   ├── tts.py              — script → audio + timestamps (Cloud TTS Chirp HD)
│   ├── video_script.py     — script + timestamps → Veo prompts (Gemini 3.1 Pro)
│   ├── veo.py              — prompts → video clips (Veo 2)
│   └── stitcher.py         — clips + audio → final MP4 (ffmpeg)
├── models/
│   └── manifest.py   — Pydantic models for validated agent outputs
├── routers/
│   ├── generate.py   — POST /api/generate
│   ├── status.py     — GET  /api/status/{job_id}
│   └── live.py       — WS   /api/live/{job_id}
└── tools/
    ├── gemini.py     — shared Gemini client + model ID
    ├── job_store.py  — in-memory job state (swap Firestore for prod)
    └── storage.py    — local filesystem (DEV_MODE=true) or GCS toggle
```

---

## API

```
POST /api/generate
  Body: multipart/form-data { file: <PDF> }
  Returns: { job_id, status: "processing" }

GET /api/status/{job_id}
  Returns: { status: "processing"|"done"|"error", step, video_url }

WS /api/live/{job_id}
  Send: PCM 16-bit audio bytes
  Recv: { type: "transcript", text }
        → "resuming now" in text means frontend should resume video
```

---

## Team Assignments

| Who | Files |
|-----|-------|
| Person 1 | `agents/parser.py`, `agents/narrative_script.py`, `pipeline.py`, GCP infra |
| Person 2 | `agents/tts.py`, `agents/video_script.py` |
| Person 3 | `agents/veo.py`, `agents/stitcher.py`, `tools/storage.py` |
| Person 4 | `routers/live.py`, frontend |

---

## GCP Deployment

```bash
# Backend → Cloud Run
cd backend
gcloud run deploy nevertrtfm-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=your-project-id,DEV_MODE=false,GCS_BUCKET=nevertrtfm

# Frontend → Firebase
cd frontend
npm run build
firebase deploy
```
