# WS /api/live/{job_id} — LiveAgent
# Bidirectional: receives PCM 16-bit audio from browser, sends text transcripts back.
# "resuming now" in response text → frontend resumes video (no backend logic needed).

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from google.genai import types

from tools.gemini import build_live_client
from tools.job_store import get_job

router = APIRouter()

LIVE_MODEL = "publishers/google/models/gemini-2.0-flash-live-preview-04-09"


def _build_system_prompt(job: dict) -> str:
    manifest = job.get("manifest") or {}
    knowledge_base = job.get("knowledge_base") or {}

    lines = [
        f"You are a helpful assistant answering questions about a report titled: \"{manifest.get('title', 'this report')}\".",
        "",
        f"OVERALL SUMMARY: {manifest.get('overall_summary', '')}",
        f"SENTIMENT: {manifest.get('sentiment', '')} — {manifest.get('sentiment_reason', '')}",
        "",
        "KEY SECTIONS:",
    ]

    for section in manifest.get("key_sections", []):
        lines.append(f"\n[{section.get('heading', '')}]")
        lines.append(f"Summary: {section.get('summary', '')}")
        for stat in section.get("key_stats", []):
            lines.append(f"  • {stat}")

    if knowledge_base:
        lines.append("\nDETAILED KNOWLEDGE BASE:")

        for finding in knowledge_base.get("deep_findings", []):
            lines.append(f"  - {finding}")

        for fact in knowledge_base.get("key_facts", []):
            lines.append(f"  - {fact}")

        for risk in knowledge_base.get("risks_and_failures", []):
            lines.append(f"  ⚠ {risk}")

        for success in knowledge_base.get("successes_and_rationale", []):
            lines.append(f"  ✓ {success}")

        definitions = knowledge_base.get("definitions", {})
        if isinstance(definitions, dict):
            for term, definition in definitions.items():
                lines.append(f"  [{term}]: {definition}")

        expert = knowledge_base.get("expert_detail", "")
        if expert:
            lines.append(f"\nEXPERT DETAIL: {expert}")

    lines.append(
        "\nIMPORTANT: Answer questions conversationally, like explaining to a friend. "
        "Be concise (2-4 sentences max). "
        "Always end every answer with the exact phrase \"resuming now\" so the video can resume."
    )

    return "\n".join(lines)


@router.websocket("/live/{job_id}")
async def live(websocket: WebSocket, job_id: str):
    job = get_job(job_id)
    if job is None or job["status"] != "done":
        await websocket.close(code=1008)
        return

    await websocket.accept()
    print(f"[live] WebSocket accepted for job {job_id}", flush=True)

    system_prompt = _build_system_prompt(job)
    client = build_live_client()

    live_config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(
            role="user",
            parts=[types.Part(text=system_prompt)],
        ),
    )

    print(f"[live] Connecting to Gemini Live ({LIVE_MODEL})...", flush=True)
    try:
        async with client.aio.live.connect(
            model=LIVE_MODEL,
            config=live_config,
        ) as session:
            print(f"[live] Gemini Live session established for job {job_id}", flush=True)

            closed = asyncio.Event()

            async def receive_from_client():
                """Read PCM audio bytes from browser → forward to Gemini."""
                audio_chunk_count = 0
                try:
                    while not closed.is_set():
                        data = await websocket.receive()
                        if data.get("type") == "websocket.disconnect":
                            print("[live] Client disconnected (receive)", flush=True)
                            break
                        if data.get("bytes"):
                            audio_chunk_count += 1
                            if audio_chunk_count <= 3 or audio_chunk_count % 50 == 0:
                                print(f"[live] 🎤 Audio chunk #{audio_chunk_count} ({len(data['bytes'])} bytes)", flush=True)
                            await session.send_realtime_input(
                                media=types.Blob(data=data["bytes"], mime_type="audio/pcm;rate=16000")
                            )
                        elif data.get("text"):
                            import json
                            try:
                                msg = json.loads(data["text"])
                                if msg.get("type") == "client_interrupt":
                                    print("[live] 🛑 Interrupt from client", flush=True)
                                    await session.send(
                                        input=types.LiveClientContent(
                                            turns=[
                                                types.Content(role="user", parts=[]),
                                            ],
                                            turn_complete=True,
                                        )
                                    )
                                elif msg.get("type") == "client_turn_done":
                                    print("[live] 🏁 User stopped mic", flush=True)
                                    await session.send(
                                        input=types.LiveClientContent(turn_complete=True)
                                    )
                            except Exception as parse_err:
                                print(f"[live] WS text parse error: {parse_err}", flush=True)
                except WebSocketDisconnect:
                    print("[live] receive_from_client: WebSocketDisconnect", flush=True)
                except Exception as e:
                    print(f"[live] receive_from_client error: {e}", flush=True)
                finally:
                    closed.set()

            async def send_to_client():
                """Stream audio chunks to browser; signal turn_complete when done."""
                msg_count = 0
                turn_count = 0
                try:
                    # session.receive() yields messages for ONE turn only.
                    # We must re-call it after each turn to keep the session alive.
                    while not closed.is_set():
                        turn_count += 1
                        print(f"[live] 📡 Waiting for Gemini turn #{turn_count}...", flush=True)
                        async for msg in session.receive():
                            if closed.is_set():
                                break

                            msg_count += 1

                            # Shorthand: msg.data is PCM audio bytes when modality=AUDIO
                            if msg.data:
                                await websocket.send_bytes(msg.data)
                            elif msg.server_content:
                                sc = msg.server_content
                                if sc.model_turn:
                                    for part in sc.model_turn.parts or []:
                                        if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                                            await websocket.send_bytes(bytes(part.inline_data.data))
                                if sc.turn_complete:
                                    print(f"[live] turn_complete (msg #{msg_count}, turn #{turn_count}) → signalling client", flush=True)
                                    await websocket.send_json({"type": "turn_complete"})
                            else:
                                print(f"[live] msg #{msg_count}: no data/server_content", flush=True)
                except WebSocketDisconnect:
                    print("[live] send_to_client: WebSocketDisconnect", flush=True)
                except Exception as e:
                    print(f"[live] send_to_client error: {type(e).__name__}: {e}", flush=True)
                finally:
                    closed.set()
                    print(f"[live] send_to_client exited after {msg_count} msgs, {turn_count} turns", flush=True)

            receive_task = asyncio.create_task(receive_from_client())
            send_task = asyncio.create_task(send_to_client())

            # Wait for BOTH tasks — don't kill one when the other finishes.
            # The closed event coordinates shutdown.
            await closed.wait()

            # Give the other task a moment to finish gracefully
            await asyncio.sleep(0.5)

            for task in [receive_task, send_task]:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            print(f"[live] Session fully closed for job {job_id}", flush=True)

    except WebSocketDisconnect:
        print(f"[live] WebSocketDisconnect (outer) for job {job_id}", flush=True)
    except Exception as e:
        print(f"[live] session error for job {job_id}: {e}")
        try:
            await websocket.send_json({"type": "error", "text": str(e)})
            await websocket.close()
        except Exception:
            pass
