# TTSAgent
# IN:  session["narration_script"]  — script with hook, scenes, outro
# OUT: session["tts_result"]        — { audio_gcs_uri, duration_seconds, word_timestamps, scene_timestamps }
# Model: Google Cloud TTS (Chirp HD)
# Note: word_timestamps drive caption generation in StitcherAgent (pure function, no LLM)

from google.adk.agents import BaseAgent

class TTSAgent(BaseAgent):
    pass

tts_agent = TTSAgent(name="TTSAgent")
