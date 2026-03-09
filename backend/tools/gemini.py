"""
Shared Gemini client builder and generate helper.
All agents that call Gemini should import from here.

Auth priority:
  1. GOOGLE_VERTEX_API_KEY — Vertex AI Express Mode (local dev, no gcloud needed)
  2. Application Default Credentials — service account / gcloud login (Cloud Run, prod)
"""

import os
import time
import random

from google import genai
from google.genai import types

GEMINI_MODEL = "publishers/google/models/gemini-2.5-pro-preview-03-25"
GEMINI_FLASH_MODEL = "publishers/google/models/gemini-2.0-flash-001"
VEO_MODEL = "publishers/google/models/veo-3.0-generate-preview"
# Veo 3.1 fast — used for presenter scenes that require a reference image (avatar).
# reference_images is not supported by veo-3.0-generate-preview.
VEO_FAST_MODEL = "publishers/google/models/veo-3.1-fast-generate-001"

# Retry config for 429 RESOURCE_EXHAUSTED
_MAX_RETRIES = 5
_BASE_DELAY = 5   # seconds — first retry waits ~5s
_MAX_DELAY = 60   # cap backoff at 60s


def build_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_VERTEX_API_KEY")
    if api_key:
        return genai.Client(vertexai=True, api_key=api_key)
    return genai.Client(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )


def build_live_client() -> genai.Client:
    """
    Live API requires project + location — same as Veo.
    Express Mode (API key) doesn't support the Live API on Vertex AI.
    """
    return genai.Client(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )


def build_veo_client() -> genai.Client:
    """
    Veo's PredictLongRunning endpoint requires a project + location — Express
    Mode (API key only) does not carry a project reference and returns
    RESOURCE_PROJECT_INVALID.  Always use ADC / explicit project credentials.
    """
    return genai.Client(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )


def generate_with_retry(client: genai.Client, model: str, contents) -> types.GenerateContentResponse:
    """
    Call client.models.generate_content with exponential backoff on 429s.
    All agents should use this instead of calling generate_content directly.
    """
    delay = _BASE_DELAY
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except Exception as e:
            is_429 = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if is_429 and attempt < _MAX_RETRIES:
                # Jitter prevents both concurrent callers retrying at the same instant
                jitter = random.uniform(0, delay * 0.3)
                wait = delay + jitter
                print(f"  [gemini] 429 rate limit (attempt {attempt}/{_MAX_RETRIES}), retrying in {wait:.1f}s...")
                time.sleep(wait)
                delay = min(delay * 2, _MAX_DELAY)
            else:
                raise
