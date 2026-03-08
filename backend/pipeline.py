from google.adk.agents import SequentialAgent, ParallelAgent

from agents.parser import parser_agent
from agents.knowledge_base import knowledge_base_agent
from agents.video_script import video_script_agent
from agents.veo import veo_agent
from agents.stitcher import stitcher_agent
from tools.job_store import update_job
import traceback

# ParserAgent and KnowledgeBaseAgent both read session["file_path"] and run in parallel.
# ParserAgent  → session["manifest"]       (used by VideoScriptAgent + LiveAgent)
# KnowledgeBaseAgent → session["knowledge_base"] (injected into LiveAgent system context only)
ingestion = ParallelAgent(
    name="Ingestion",
    sub_agents=[parser_agent, knowledge_base_agent],
)

pipeline = SequentialAgent(
    name="NeverRTFM",
    sub_agents=[
        ingestion,          # PDF → manifest + knowledge_base (parallel)
        video_script_agent, # manifest → video_script (writes dialogue + directs presenter/b-roll)
        veo_agent,          # video_script → veo_clips (audio baked in via Veo 3.1)
        stitcher_agent,     # concatenate clips + burn captions → final_video_uri
    ],
)


async def run_pipeline(job_id: str, file_path: str):
    """Entry point called by the FastAPI background task."""
    try:
        update_job(job_id, status="processing", step="parsing")

        session_state = {
            "job_id": job_id,
            "file_path": file_path,
        }

        result = await pipeline.run_async(
            input=f"Process the PDF at: {file_path}",
            state=session_state,
        )

        update_job(
            job_id,
            status="done",
            step="complete",
            video_url=session_state.get("final_video_uri"),
            manifest=session_state.get("manifest"),
            knowledge_base=session_state.get("knowledge_base"),
        )

    except Exception as e:
        update_job(
            job_id,
            status="error",
            error=str(e),
            traceback=traceback.format_exc(),
        )
