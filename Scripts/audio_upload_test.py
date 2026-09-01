from storage_manager import upload_audio

AUDIO_PATH = r"media_output\audio\test_audio.mp3"
REQUEST_ID = "test_audio_upload"

try:
    remote_path, signed_url = upload_audio(
        AUDIO_PATH,
        REQUEST_ID
    )

    print("Audio upload successful")
    print("Remote path:", remote_path)
    print("Signed URL:", signed_url)

except Exception as e:
    print("Audio upload failed")
    print(e)