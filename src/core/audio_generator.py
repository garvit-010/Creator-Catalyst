import asyncio
import edge_tts
import tempfile
import os

# Voice mapping for better UI experience
AVAILABLE_VOICES = {
    "👨‍💼 Professional (Male)": "en-US-ChristopherNeural",
    "👩‍💼 Professional (Female)": "en-US-JennyNeural",
    "👱 Casual (Male)": "en-US-EricNeural",
    "👩 Casual (Female)": "en-US-AnaNeural"
}

async def _generate_audio_async(text, voice, output_path):
    """
    Async helper to communicate with the edge-tts service.
    """
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def generate_audio_file(text, voice_friendly_name="👨‍💼 Professional (Male)"):
    """
    Generates an audio file from text using edge-tts.
    
    Args:
        text (str): The text content to convert to speech.
        voice_friendly_name (str): The UI-friendly name of the voice to use.
        
    Returns:
        str: The file path of the generated audio (mp3).
    """
    if not text:
        return None
        
    try:
        # Map friendly name to internal ID, default to Christopher if not found
        voice_id = AVAILABLE_VOICES.get(voice_friendly_name, "en-US-ChristopherNeural")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            output_path = f.name
            
        # Run the async generation in a synchronous context
        asyncio.run(_generate_audio_async(text, voice_id, output_path))
        
        return output_path
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None