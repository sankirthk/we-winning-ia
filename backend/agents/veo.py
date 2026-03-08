# VeoAgent — Person 3
# IN:  session["video_script"]
# OUT: session["veo_clips"]  { clips: [{ scene_id, gcs_uri, duration }] }
# Model: Veo 2 via Vertex AI

from google.adk.agents import BaseAgent

class VeoAgent(BaseAgent):
    pass

veo_agent = VeoAgent(name="VeoAgent")
