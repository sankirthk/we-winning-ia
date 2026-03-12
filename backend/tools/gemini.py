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

GEMINI_MODEL = "publishers/google/models/gemini-2.5-pro"
GEMINI_FLASH_MODEL = "publishers/google/models/gemini-3-flash-preview"
VEO_MODEL = os.getenv(
    "VEO_MODEL",
    "publishers/google/models/veo-3.1-generate-preview",
)

_MAX_RETRIES = 5
_BASE_DELAY = 5
_MAX_DELAY = 60


def build_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_VERTEX_API_KEY")
    if api_key:
        return genai.Client(vertexai=True, api_key=api_key)

    return genai.Client(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )


def build_veo_client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )


def generate_with_retry(
    client: genai.Client, model: str, contents
) -> types.GenerateContentResponse:
    delay = _BASE_DELAY

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except Exception as e:
            is_429 = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)

            if is_429 and attempt < _MAX_RETRIES:
                jitter = random.uniform(0, delay * 0.3)
                wait = delay + jitter
                print(
                    f"[gemini] 429 rate limit (attempt {attempt}/{_MAX_RETRIES}), "
                    f"retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
                delay = min(delay * 2, _MAX_DELAY)
            else:
                raise
