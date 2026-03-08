import asyncio
import base64
import json
import os
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from google.adk.agents import Agent
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types

from tools.job_store import get_job

router = APIRouter()

APP_NAME = "neverrtfm-live"
LIVE_MODEL = os.getenv("LIVE_MODEL", "gemini-live-2.5-flash-native-audio")
LIVE_VOICE = os.getenv("LIVE_VOICE", "Kore")

session_service = InMemorySessionService()


def get_report_context(
    section: str | None = None,
    tool_context: ToolContext | None = None,
) -> str:
    """
    Return grounded report context for the live agent.
    Supports both newer and older manifest layouts.
    """
    if tool_context is None:
        return json.dumps({"error": "Tool context unavailable."}, ensure_ascii=False)

    state = tool_context.state or {}
    manifest = state.get("manifest") or {}
    knowledge_base = state.get("knowledge_base") or {}
    current_scene = state.get("current_scene") or ""

    if not manifest and not knowledge_base:
        return json.dumps(
            {"error": "No report context available for this session."},
            ensure_ascii=False,
        )

    def _matches(value: Any, query: str) -> bool:
        return isinstance(value, str) and query in value.lower()

    def _get_sections() -> list[dict[str, Any]]:
        raw_sections = manifest.get("key_sections") or manifest.get("sections") or []
        if not isinstance(raw_sections, list):
            return []

        cleaned: list[dict[str, Any]] = []
        for sec in raw_sections:
            if not isinstance(sec, dict):
                continue
            cleaned.append(
                {
                    "id": sec.get("id"),
                    "title": sec.get("heading") or sec.get("title"),
                    "summary": sec.get("summary"),
                    "key_points": sec.get("key_stats") or sec.get("key_points") or [],
                    "page": sec.get("page"),
                }
            )
        return cleaned

    def _filter_str_list(values: Any, query: str, limit: int) -> list[str]:
        if not isinstance(values, list):
            return []
        return [v for v in values if isinstance(v, str) and query in v.lower()][:limit]

    def _filter_definitions(definitions: Any, query: str) -> dict[str, Any]:
        if not isinstance(definitions, dict):
            return {}
        return {
            k: v
            for k, v in definitions.items()
            if (isinstance(k, str) and query in k.lower())
            or (isinstance(v, str) and query in v.lower())
        }

    def _full_payload() -> dict[str, Any]:
        return {
            "current_scene": current_scene,
            "document_title": manifest.get("title") or knowledge_base.get("document_title"),
            "document_type": manifest.get("type"),
            "total_pages": manifest.get("total_pages"),
            "sentiment": manifest.get("sentiment"),
            "overall_summary": (
                manifest.get("overall_summary")
                or manifest.get("global_summary")
                or manifest.get("summary")
            ),
            "sections": _get_sections()[:8],
            "knowledge_base": {
                "document_title": knowledge_base.get("document_title"),
                "deep_findings": (knowledge_base.get("deep_findings") or [])[:12],
                "key_facts": (knowledge_base.get("key_facts") or [])[:20],
                "risks_and_failures": (knowledge_base.get("risks_and_failures") or [])[:12],
                "successes_and_rationale": (
                    knowledge_base.get("successes_and_rationale") or []
                )[:12],
                "definitions": knowledge_base.get("definitions", {}),
                "expert_detail": knowledge_base.get("expert_detail", ""),
            },
        }

    if not section or not section.strip():
        return json.dumps(_full_payload(), ensure_ascii=False)

    query = section.strip().lower()
    sections = _get_sections()

    matched_sections: list[dict[str, Any]] = []
    for sec in sections:
        if _matches(sec.get("title"), query):
            matched_sections.append(sec)
            continue

        key_points = sec.get("key_points") or []
        if any(isinstance(k, str) and query in k.lower() for k in key_points):
            matched_sections.append(sec)
            continue

        if _matches(sec.get("summary"), query):
            matched_sections.append(sec)

    payload = {
        "current_scene": current_scene,
        "section": section,
        "document_title": manifest.get("title") or knowledge_base.get("document_title"),
        "overall_summary": (
            manifest.get("overall_summary")
            or manifest.get("global_summary")
            or manifest.get("summary")
        ),
        "matched_sections": matched_sections[:6],
        "knowledge_base_matches": {
            "deep_findings": _filter_str_list(
                knowledge_base.get("deep_findings"), query, limit=8
            ),
            "key_facts": _filter_str_list(
                knowledge_base.get("key_facts"), query, limit=10
            ),
            "risks_and_failures": _filter_str_list(
                knowledge_base.get("risks_and_failures"), query, limit=8
            ),
            "successes_and_rationale": _filter_str_list(
                knowledge_base.get("successes_and_rationale"), query, limit=8
            ),
            "definitions": _filter_definitions(
                knowledge_base.get("definitions"), query
            ),
            "expert_detail": knowledge_base.get("expert_detail", ""),
        },
    }

    return json.dumps(payload, ensure_ascii=False)


live_agent = Agent(
    name="neverrtfm_live_agent",
    model=LIVE_MODEL,
    tools=[get_report_context],
    instruction=(
        "You are the live voice agent for NeverRTFM.\n"
        "Always call get_report_context before answering report-related questions.\n"
        "Use matched_sections and knowledge_base_matches when section-specific context is returned.\n"
        "Use the full manifest and knowledge_base only when broad context is needed.\n"
        "Answer briefly, clearly, and only from the uploaded report.\n"
        "If the report does not support the answer, say so.\n"
    ),
)

runner = Runner(
    app_name=APP_NAME,
    agent=live_agent,
    session_service=session_service,
)


async def _ensure_session(job_id: str):
    user_id = f"job:{job_id}"
    session_id = f"live:{job_id}"

    job = get_job(job_id) or {}
    manifest = job.get("manifest") or {}
    knowledge_base = job.get("knowledge_base") or {}

    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    if session is None:
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
            state={
                "job_id": job_id,
                "manifest": manifest,
                "knowledge_base": knowledge_base,
                "current_scene": "",
            },
        )
    else:
        session.state["manifest"] = manifest
        session.state["knowledge_base"] = knowledge_base

    return user_id, session_id, session

def _build_run_config() -> RunConfig:
    return RunConfig(
        streaming_mode=StreamingMode.BIDI,
        response_modalities=[types.Modality.AUDIO],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        session_resumption=types.SessionResumptionConfig(),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=LIVE_VOICE
                )
            ),
            language_code="en-US",
        ),
    )

@router.websocket("/live/{job_id}")
async def live_ws(websocket: WebSocket, job_id: str):
    print(f"[live_ws] incoming connection for job_id={job_id}")
    await websocket.accept()
    print(f"[live_ws] websocket accepted for job_id={job_id}")

    live_request_queue = LiveRequestQueue()
    paused_for_user = False
    sent_resume = False
    websocket_closed = False

    async def safe_send_json(payload: dict[str, Any]) -> None:
        nonlocal websocket_closed
        if websocket_closed:
            return
        try:
            await websocket.send_json(payload)
        except Exception as e:
            websocket_closed = True
            print(f"[websocket] send failed: {e!r}")

    async def safe_close(code: int = 1000) -> None:
        nonlocal websocket_closed
        if websocket_closed:
            return
        try:
            await websocket.close(code=code)
        except Exception:
            pass
        websocket_closed = True

    try:
        job = get_job(job_id)
        print(f"[live_ws] job lookup: {job}")

        if not job:
            await safe_send_json(
                {"type": "error", "message": f"Unknown job_id: {job_id}"}
            )
            await safe_close(code=4404)
            return

        if job.get("status") == "error":
            await safe_send_json(
                {
                    "type": "error",
                    "message": job.get("error", "Pipeline failed before live session."),
                }
            )
            await safe_close(code=4410)
            return

        if not job.get("manifest") and not job.get("knowledge_base"):
            await safe_send_json(
                {
                    "type": "error",
                    "message": "Report context not ready yet for live Q&A.",
                }
            )
            await safe_close(code=4409)
            return

        user_id, session_id, _session = await _ensure_session(job_id)
        print(f"[live_ws] session ready user_id={user_id} session_id={session_id}")

        run_config = _build_run_config()

        async def upstream_task():
            nonlocal paused_for_user, sent_resume, websocket_closed
            print("[upstream] started")

            try:
                while True:
                    message = await websocket.receive()
                    print(f"[upstream] received message keys={list(message.keys())}")

                    if message.get("type") == "websocket.disconnect":
                        print("[upstream] disconnect message received")
                        break

                    if message.get("bytes") is not None:
                        audio_bytes = message["bytes"]
                        print(f"[upstream] got audio bytes len={len(audio_bytes)}")

                        if not paused_for_user:
                            paused_for_user = True
                            sent_resume = False
                            await safe_send_json({"type": "pause_video"})

                        audio_blob = types.Blob(
                            mime_type="audio/pcm;rate=16000",
                            data=audio_bytes,
                        )
                        live_request_queue.send_realtime(audio_blob)
                        continue

                    raw_text = message.get("text")
                    if raw_text is None:
                        continue

                    print(f"[upstream] raw_text={raw_text}")

                    try:
                        payload = json.loads(raw_text)
                    except json.JSONDecodeError:
                        payload = {"type": "text", "text": raw_text}

                    msg_type = payload.get("type")
                    print(f"[upstream] msg_type={msg_type}")

                    if msg_type == "text":
                        text = (payload.get("text") or "").strip()
                        print(f"[upstream] text payload={text}")

                        if text:
                            if not paused_for_user:
                                paused_for_user = True
                                sent_resume = False
                                await safe_send_json({"type": "pause_video"})

                            live_request_queue.send_content(
                                types.Content(
                                    role="user",
                                    parts=[types.Part(text=text)],
                                )
                            )
                            print("[upstream] sent text content to live_request_queue")

                    elif msg_type == "set_scene":
                        scene_text = (payload.get("scene_text") or "").strip()
                        print(f"[upstream] set_scene={scene_text}")

                        session = await session_service.get_session(
                            app_name=APP_NAME,
                            user_id=user_id,
                            session_id=session_id,
                        )
                        if session:
                            session.state["current_scene"] = scene_text

                        await safe_send_json(
                            {
                                "type": "scene_updated",
                                "scene_text": scene_text,
                            }
                        )

                    elif msg_type == "end_turn":
                        print("[upstream] end_turn")
                        live_request_queue.send_content(
                            types.Content(
                                role="user",
                                parts=[types.Part(text="")],
                            )
                        )

                    elif msg_type == "ping":
                        print("[upstream] ping")
                        await safe_send_json({"type": "pong"})

                    else:
                        print(f"[upstream] unknown msg_type={msg_type}")

            except WebSocketDisconnect:
                print("[upstream] websocket disconnected")
            except Exception as e:
                print(f"[upstream] ERROR: {e!r}")
                await safe_send_json({"type": "error", "message": repr(e)})
            finally:
                print("[upstream] closing live_request_queue")
                live_request_queue.close()

        async def downstream_task():
            nonlocal paused_for_user, sent_resume
            print("[downstream] started")
            print("[downstream] entering runner.run_live")

            try:
                async for event in runner.run_live(
                    user_id=user_id,
                    session_id=session_id,
                    live_request_queue=live_request_queue,
                    run_config=run_config,
                ):
                    print(f"[downstream] got event: {event}")

                    if getattr(event, "input_transcription", None):
                        t = event.input_transcription
                        if t and t.text and t.text.strip():
                            await safe_send_json(
                                {
                                    "type": "transcript",
                                    "speaker": "user",
                                    "text": t.text,
                                    "final": bool(t.finished),
                                }
                            )

                    if getattr(event, "output_transcription", None):
                        t = event.output_transcription
                        if t and t.text and t.text.strip():
                            await safe_send_json(
                                {
                                    "type": "transcript",
                                    "speaker": "agent",
                                    "text": t.text,
                                    "final": bool(t.finished),
                                }
                            )

                            if t.finished and paused_for_user and not sent_resume:
                                sent_resume = True
                                paused_for_user = False
                                await safe_send_json({"type": "resume_video"})

                    if getattr(event, "content", None) and getattr(event.content, "parts", None):
                        for part in event.content.parts:
                            inline_data = getattr(part, "inline_data", None)
                            if (
                                inline_data
                                and inline_data.mime_type
                                and inline_data.mime_type.startswith("audio/pcm")
                            ):
                                await safe_send_json(
                                    {
                                        "type": "audio",
                                        "mime_type": inline_data.mime_type,
                                        "data_b64": base64.b64encode(
                                            inline_data.data
                                        ).decode("utf-8"),
                                    }
                                )

                            text_part = getattr(part, "text", None)
                            if text_part:
                                await safe_send_json(
                                    {
                                        "type": "message",
                                        "text": text_part,
                                    }
                                )

                    if getattr(event, "error_code", None) or getattr(event, "error_message", None):
                        err_msg = getattr(event, "error_message", "Live agent error")
                        print(f"[downstream] event error: {err_msg}")
                        await safe_send_json(
                            {
                                "type": "error",
                                "message": err_msg,
                            }
                        )

                print("[downstream] run_live loop ended normally")

            except Exception as e:
                print(f"[downstream] ERROR: {e!r}")
                await safe_send_json({"type": "error", "message": repr(e)})
            finally:
                print("[downstream] closing live_request_queue")
                live_request_queue.close()

        await asyncio.gather(upstream_task(), downstream_task())

    except WebSocketDisconnect:
        print("[live_ws] websocket disconnect bubbled up")
    except Exception as e:
        print(f"[live_ws] ERROR: {e!r}")
        await safe_send_json({"type": "error", "message": repr(e)})
    finally:
        print("[live_ws] closing websocket")
        await safe_close()