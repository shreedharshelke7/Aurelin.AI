import os
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

client = ElevenLabs(
    api_key=os.getenv("TTS_API_KEY")
)

VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"


def generate_audio(narration: str, request_id: str):
    output_dir = Path("media_output") / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = output_dir / f"{request_id}.mp3"

    audio = client.text_to_speech.convert(
        text=narration,
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    with open(audio_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    return str(audio_path)