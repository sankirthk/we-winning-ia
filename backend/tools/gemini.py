"""
Shared Gemini client builder.
All agents that call Gemini should import from here.

Auth priority:
  1. GOOGLE_VERTEX_API_KEY — Vertex AI Express Mode (local dev, no gcloud needed)
  2. Application Default Credentials — service account / gcloud login (Cloud Run, prod)
"""

import os

from google import genai

GEMINI_MODEL = "publishers/google/models/gemini-3.1-pro-preview"


def build_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_VERTEX_API_KEY")
    if api_key:
        return genai.Client(vertexai=True, api_key=api_key)
    return genai.Client(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )
