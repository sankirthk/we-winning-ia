# TTSAgent
# IN:  session["narration_script"]
# OUT: session["tts_result"] — { audio_path, duration_seconds, word_timestamps, scene_timestamps }

import os
import wave
import io
from typing import AsyncGenerator

from google.cloud import texttospeech
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from tools.job_store import update_job

VOICE_NAME = os.getenv("TTS_VOICE", "en-US-Neural2-D")
LANGUAGE_CODE = "en-US"


def _get_chunks(narration_script: dict) -> list[dict]:
    chunks = []
    if narration_script.get("hook"):
        chunks.append({"scene_id": "hook", "text": narration_script["hook"]})
    for scene in narration_script.get("scenes", []):
        chunks.append({"scene_id": scene.get("scene_id", len(chunks)), "text": scene["narration"]})
    if narration_script.get("outro"):
        chunks.append({"scene_id": "outro", "text": narration_script["outro"]})
    return chunks


def _text_to_ssml(text: str) -> tuple[str, list[str]]:
    words = text.split()
    parts = [f'<mark name="w{i}"/>{word}' for i, word in enumerate(words)]
    ssml = f'<speak>{" ".join(parts)}</speak>'
    return ssml, words


def _synthesize_chunk(client, text: str) -> tuple[bytes, float, list[dict]]:
    ssml, words = _text_to_ssml(text)

    request = texttospeech.SynthesizeSpeechRequest(
        input=texttospeech.SynthesisInput(ssml=ssml),
        voice=texttospeech.VoiceSelectionParams(
            language_code=LANGUAGE_CODE,
            name=VOICE_NAME,
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000,
        ),
        enable_time_pointing=[
            texttospeech.SynthesizeSpeechRequest.TimepointType.SSML_MARK
        ],
    )

    response = client.synthesize_speech(request=request)
    audio_bytes = response.audio_content

    # Parse WAV to get duration
    with wave.open(io.BytesIO(audio_bytes)) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)

    # Build word timestamps from timepoints
    word_timestamps = []
    for tp in response.timepoints:
        idx = int(tp.mark_name[1:])  # "w0" -> 0
        if idx < len(words):
            word_timestamps.append({
                "word": words[idx],
                "start_s": round(tp.time_seconds, 3),
            })

    return audio_bytes, duration, word_timestamps


def _combine_wav_bytes(chunks_audio: list[bytes]) -> bytes:
    """Concatenate multiple LINEAR16 WAV files into one."""
    combined = io.BytesIO()
    all_frames = b""
    params = None

    for audio_bytes in chunks_audio:
        with wave.open(io.BytesIO(audio_bytes)) as wf:
            if params is None:
                params = wf.getparams()
            all_frames += wf.readframes(wf.getnframes())

    with wave.open(combined, "wb") as out:
        out.setparams(params)
        out.writeframes(all_frames)

    return combined.getvalue()


class TTSAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        job_id = ctx.session.state["job_id"]
        narration_script = ctx.session.state["narration_script"]

        update_job(job_id, step="tts")

        chunks = _get_chunks(narration_script)
        tts_client = texttospeech.TextToSpeechClient()

        all_audio = []
        scene_timestamps = []
        word_timestamps = []
        offset = 0.0

        for chunk in chunks:
            audio_bytes, duration, wtimes = _synthesize_chunk(tts_client, chunk["text"])
            all_audio.append(audio_bytes)

            # Word timestamps with global offset applied
            for wt in wtimes:
                word_timestamps.append({
                    "word": wt["word"],
                    "start_s": round(wt["start_s"] + offset, 3),
                })

            scene_timestamps.append({
                "scene_id": chunk["scene_id"],
                "start_s": round(offset, 3),
                "end_s": round(offset + duration, 3),
            })

            offset += duration

        # Save combined audio
        combined_audio = _combine_wav_bytes(all_audio)
        audio_dir = f"local_storage/{job_id}"
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = f"{audio_dir}/narration.wav"
        with open(audio_path, "wb") as f:
            f.write(combined_audio)

        tts_result = {
            "audio_path": audio_path,
            "duration_seconds": round(offset, 2),
            "word_timestamps": word_timestamps,
            "scene_timestamps": scene_timestamps,
        }

        ctx.session.state["tts_result"] = tts_result
        update_job(job_id, step="video_script")

        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=f"TTS done: {round(offset, 1)}s audio, {len(word_timestamps)} words, {len(scene_timestamps)} scenes")]),
        )


tts_agent = TTSAgent(name="TTSAgent")