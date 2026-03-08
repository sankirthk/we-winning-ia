import traceback

from google.adk.agents import SequentialAgent, ParallelAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.parser import parser_agent
from agents.knowledge_base import knowledge_base_agent
from agents.video_script import video_script_agent
from agents.veo import veo_agent
from agents.stitcher import stitcher_agent
from tools.job_store import update_job

APP_NAME = "nevertrtfm"

# ParserAgent and KnowledgeBaseAgent both read session["file_path"] and run in parallel.
# ParserAgent        → session["manifest"]        (used by VideoScriptAgent + LiveAgent)
# KnowledgeBaseAgent → session["knowledge_base"]  (injected into LiveAgent system context only)
ingestion = ParallelAgent(
    name="Ingestion",
    sub_agents=[parser_agent, knowledge_base_agent],
)

pipeline = SequentialAgent(
    name="NeverRTFM",
    sub_agents=[
        ingestion,          # PDF → manifest + knowledge_base (parallel)
        video_script_agent, # manifest → video_script (writes dialogue + directs presenter/b-roll)
        veo_agent,          # video_script → veo_clips (audio baked in via Veo 3.0)
        stitcher_agent,     # concatenate clips + burn captions → final_video_uri
    ],
)


async def run_pipeline(job_id: str, file_path: str):
    """Entry point called by the FastAPI background task."""
    try:
        update_job(job_id, status="processing", step="parsing")

        session_service = InMemorySessionService()

        # Create a session with initial state — agents read/write via ctx.session.state
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=job_id,
            session_id=job_id,
            state={
                "job_id": job_id,
                "file_path": file_path,
            },
        )

        runner = Runner(
            agent=pipeline,
            app_name=APP_NAME,
            session_service=session_service,
        )

        # Drain the event stream — agents update job_store internally as they run
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

        # Read final state from session
        final_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=job_id,
            session_id=job_id,
        )
        state = final_session.state

        update_job(
            job_id,
            status="done",
            step="complete",
            video_url=state.get("final_video_uri"),
            manifest=state.get("manifest"),
            knowledge_base=state.get("knowledge_base"),
            veo_clips=state.get("veo_clips"),
            final_video_uri=state.get("final_video_uri"),
        )

    except Exception as e:
        update_job(
            job_id,
            status="error",
            error=str(e),
            traceback=traceback.format_exc(),
        )
