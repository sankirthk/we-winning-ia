# VideoScriptAgent — Person 2
# IN:  session["narration_script"] + session["tts_result"]
# OUT: session["video_script"]  { veo_prompts: [{ scene_id, start, end, prompt, style }] }
# Model: Gemini 2.0 Flash

from google.adk.agents import BaseAgent

class VideoScriptAgent(BaseAgent):
    pass

video_script_agent = VideoScriptAgent(name="VideoScriptAgent")
