# StitcherAgent — Person 3
# IN:  session["veo_clips"] + session["tts_result"]
# OUT: session["final_video_uri"]  signed GCS URL to finished 1080x1920 MP4
# Tool: ffmpeg

from google.adk.agents import BaseAgent

class StitcherAgent(BaseAgent):
    pass

stitcher_agent = StitcherAgent(name="StitcherAgent")
