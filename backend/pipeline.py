import hashlib
import traceback

from google.adk.agents import SequentialAgent, ParallelAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.parser import ParserAgent
from agents.knowledge_base import KnowledgeBaseAgent
from agents.video_script import VideoScriptAgent
from agents.veo import VeoAgent
from agents.stitcher import StitcherAgent
from tools.job_store import update_job, get_job
from tools.storage import (
    load_cache,
    hash_file_exists, get_hash_uri, get_signed_url,
    DEV_MODE,
)

APP_NAME = "nevertrtfm"


def _ingestion_agent():
    return ParallelAgent(
        name="Ingestion",
        sub_agents=[
            ParserAgent(name="ParserAgent"),
            KnowledgeBaseAgent(name="KnowledgeBaseAgent"),
        ],
    )


def _build_pipeline(agents: list) -> SequentialAgent:
    return SequentialAgent(name="NeverRTFM", sub_agents=agents)


VALID_TONES = {"formal", "explanatory", "casual"}


async def run_pipeline(job_id: str, file_path: str, pdf_bytes: bytes, tone: str = "explanatory"):
    """
    Smart resumption pipeline. Checks what already exists for this PDF
    (by content hash) and only runs what's missing:

      final.mp4 exists?          → return immediately
      all clips exist?           → stitch only
      some clips missing?        → Veo for missing scenes only, then stitch
      video_script missing?      → VideoScript → Veo (all) → Stitch
      manifest/kb missing?       → Ingestion → VideoScript → Veo → Stitch
    """
    try:
        tone = tone if tone in VALID_TONES else "explanatory"
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
        print(f"\n[pipeline] ▶ job={job_id}  pdf_hash={pdf_hash[:16]}...  tone={tone!r}", flush=True)
        update_job(job_id, status="processing", step="checking_cache")

        # ── 1. Final video already exists? ────────────────────────────────────
        if hash_file_exists(pdf_hash, f"final_{tone}.mp4"):
            final_uri = get_hash_uri(pdf_hash, f"final_{tone}.mp4")
            if not DEV_MODE and final_uri.startswith("gs://"):
                final_uri = get_signed_url(final_uri)
            print(f"[pipeline] ⚡ Final video cached — done instantly: {final_uri}", flush=True)
            update_job(
                job_id, status="done", step="complete",
                video_url=final_uri, final_video_uri=final_uri,
                manifest=load_cache(pdf_hash, "manifest"),
                knowledge_base=load_cache(pdf_hash, "knowledge_base"),
            )
            return

        # ── 2. Load cached intermediate results ───────────────────────────────
        cached_manifest     = load_cache(pdf_hash, "manifest")
        cached_kb           = load_cache(pdf_hash, "knowledge_base")
        cached_video_script = load_cache(pdf_hash, f"video_script_{tone}")

        has_ingestion = cached_manifest is not None and cached_kb is not None
        has_script    = cached_video_script is not None

        # ── 3. Determine which clips already exist ────────────────────────────
        existing_clips: list[dict] = []
        missing_scene_ids: set[int] = set()

        if has_script:
            for scene in cached_video_script.get("scenes", []):
                sid = scene["scene_id"]
                rel = f"clips_{tone}/clip_{sid:02d}.mp4"
                if hash_file_exists(pdf_hash, rel):
                    existing_clips.append({
                        "scene_id": sid,
                        "clip_path": get_hash_uri(pdf_hash, rel),
                        "duration_seconds": scene["duration_seconds"],
                        "caption": scene.get("caption", ""),
                    })
                    print(f"[pipeline]   clip {sid:02d} ✅ cached", flush=True)
                else:
                    missing_scene_ids.add(sid)
                    print(f"[pipeline]   clip {sid:02d} ❌ missing — will generate", flush=True)

        # ── 4. Decide which agents to run ──────────────────────────────────────
        agents_to_run = []

        if not has_ingestion:
            agents_to_run.append(_ingestion_agent())
            print(f"[pipeline]   → Ingestion (Parser ∥ KnowledgeBase)", flush=True)

        if not has_script:
            agents_to_run.append(VideoScriptAgent(name="VideoScriptAgent"))
            print(f"[pipeline]   → VideoScript", flush=True)

        need_veo = (not has_script) or bool(missing_scene_ids)
        if need_veo:
            agents_to_run.append(VeoAgent(name="VeoAgent"))
            label = f"{len(missing_scene_ids)} missing scenes" if has_script else "all scenes"
            print(f"[pipeline]   → Veo ({label})", flush=True)

        agents_to_run.append(StitcherAgent(name="StitcherAgent"))
        print(f"[pipeline]   → Stitcher", flush=True)

        # ── 5. Build session state ─────────────────────────────────────────────
        initial_state: dict = {
            "job_id":         job_id,
            "file_path":      file_path,
            "pdf_hash":       pdf_hash,
            "tone":           tone,
            "existing_clips": existing_clips,
        }

        if cached_manifest:
            initial_state["manifest"] = cached_manifest
        if cached_kb:
            initial_state["knowledge_base"] = cached_kb

        if has_script:
            script_copy = dict(cached_video_script)
            if missing_scene_ids:
                # Only pass missing scenes to VeoAgent
                script_copy["scenes"] = [
                    s for s in cached_video_script["scenes"]
                    if s["scene_id"] in missing_scene_ids
                ]
            else:
                # All clips exist — VeoAgent not in pipeline; pre-populate veo_clips
                script_copy["scenes"] = []
                initial_state["veo_clips"] = existing_clips
            initial_state["video_script"] = script_copy

        # ── 6. Run pipeline ────────────────────────────────────────────────────
        first_step = (
            "parsing"   if not has_ingestion else
            "scripting" if not has_script    else
            "veo"       if need_veo          else
            "stitching"
        )
        update_job(job_id, step=first_step)

        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=job_id,
            session_id=job_id,
            state=initial_state,
        )

        agent = _build_pipeline(agents_to_run)
        runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

        async for event in runner.run_async(
            user_id=job_id,
            session_id=job_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=f"Process the PDF at: {file_path}")],
            ),
        ):
            if event.content:
                print(f"  [{event.author}] {event.content}")

        # ── 7. Finalise ────────────────────────────────────────────────────────
        final_session = await session_service.get_session(
            app_name=APP_NAME, user_id=job_id, session_id=job_id,
        )
        state = final_session.state

        final_uri = state.get("final_video_uri") or (get_job(job_id) or {}).get("final_video_uri")

        if final_uri and not DEV_MODE and final_uri.startswith("gs://"):
            final_uri = get_signed_url(final_uri)

        print(f"\n[pipeline] ✅ Complete — final_uri: {final_uri}", flush=True)
        update_job(
            job_id,
            status="done",
            step="complete",
            video_url=final_uri,
            manifest=state.get("manifest") or cached_manifest,
            knowledge_base=state.get("knowledge_base") or cached_kb,
            veo_clips=state.get("veo_clips"),
            final_video_uri=final_uri,
        )

    except Exception as e:
        print(f"\n[pipeline] ❌ Pipeline FAILED for job {job_id}", flush=True)
        print(f"[pipeline]   error: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        update_job(
            job_id,
            status="error",
            error=str(e),
            traceback=traceback.format_exc(),
        )
