# WS /api/live/{job_id} — Person 4
# LiveAgent — always-on WebSocket for real-time voice Q&A mid-playback
# Receives: PCM 16-bit audio bytes
# Sends:    { type: "transcript", text: str }
# "resuming now" in text → frontend resumes video
# Model: Gemini Live API

from fastapi import APIRouter

router = APIRouter()

# TODO: Person 4 implements WebSocket endpoint + Gemini Live API wiring
