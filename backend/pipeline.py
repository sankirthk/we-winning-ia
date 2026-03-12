import traceback

from google.adk.agents import SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.parser import parser_agent
from agents.video_script import video_script_agent
from agents.veo import veo_agent
from agents.stitcher import stitcher_agent
from tools.job_store import update_job

APP_NAME = "neverrtfm-pipeline"

pipeline = SequentialAgent(
    name="NeverRTFM",
    sub_agents=[
        parser_agent,
        video_script_agent,
        veo_agent,
        stitcher_agent,
    ],
)

session_service = InMemorySessionService()

runner = Runner(
    app_name=APP_NAME,
    agent=pipeline,
    session_service=session_service,
)


async def run_pipeline(job_id: str, file_path: str):
    user_id = f"job:{job_id}"
    session_id = f"pipeline:{job_id}"

    try:
        update_job(job_id, status="processing", step="parsing")

        await session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
            state={
                "job_id": job_id,
                "file_path": file_path,
            },
        )

        new_message = types.Content(
            role="user",
            parts=[types.Part(text=f"Process the PDF at: {file_path}")],
        )

        async for _event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message,
        ):
            pass

        final_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        final_state = final_session.state if final_session else {}

        update_job(
            job_id,
            status="done",
            step="complete",
            video_url=final_state.get("final_video_uri"),
            manifest=final_state.get("manifest"),
            video_script=final_state.get("video_script"),
            knowledge_base=final_state.get("knowledge_base"),
        )

    except Exception as e:
        update_job(
            job_id,
            status="error",
            step="failed",
            error=str(e),
            traceback=traceback.format_exc(),
        )