"""
Pydantic models for the narration script produced by NarrativeScriptAgent.
Consumed by TTSAgent (audio synthesis) and VideoScriptAgent (Veo prompt generation).
"""

from typing import Literal

from pydantic import BaseModel, Field


class Scene(BaseModel):
    scene_id: int
    section_id: int  # maps to Manifest.key_sections[].id
    narration: str   # spoken aloud by TTS
    caption: str     # short on-screen overlay text (max 8 words)
    tone: Literal["urgent", "optimistic", "neutral", "dramatic"]


class NarrationScript(BaseModel):
    hook: str
    scenes: list[Scene] = Field(min_length=1)
    outro: str
