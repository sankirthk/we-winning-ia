from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from tools.job_store import update_job


def _text_event(author: str, text: str) -> Event:
    return Event(
        author=author,
        content=types.Content(
            role="model",
            parts=[types.Part(text=text)],
        ),
    )


class TTSAgent(BaseAgent):
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        job_id = ctx.session.state["job_id"]

        update_job(job_id, step="tts")

        narration_script = ctx.session.state.get("narration_script")
        if not narration_script:
            raise ValueError("TTSAgent: missing narration_script in session state.")

        # Temporary stub output so pipeline can continue
        tts_result = {
            "audio_uri": "stub://tts-audio.wav",
            "duration_sec": 30.0,
            "voice": "Kore",
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 30.0,
                    "text": narration_script,
                }
            ],
        }

        ctx.session.state["tts_result"] = tts_result

        update_job(job_id, step="video-script")

        yield _text_event(
            self.name,
            "Generated TTS result placeholder and stored it in session state.",
        )


tts_agent = TTSAgent(name="TTSAgent")