
from tts_manager import generate_audio

try:
    path = generate_audio(
        "The Pythagoras theorem states that in a right angled triangle, a squared plus b squared equals c squared.",
        "test_audio"
    )

    print("TTS successful")
    print("Audio saved at:", path)

except Exception as e:
    print("TTS failed")
    print(e)