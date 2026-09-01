from pathlib import Path
from supabase_client import supabase

BUCKET_NAME = "aurelin-media"


def upload_video(video_path: str, request_id: str):
    local_file = Path(video_path)

    if not local_file.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    remote_path = f"videos/{request_id}/{local_file.name}"

    with open(local_file, "rb") as file:
        supabase.storage.from_(BUCKET_NAME).upload(
            path=remote_path,
            file=file,
            file_options={
                "content-type": "video/mp4",
                "upsert": "true"
            }
        )

    signed_response = (
        supabase.storage
        .from_(BUCKET_NAME)
        .create_signed_url(remote_path, 3600)
    )

    signed_url = (
        signed_response.get("signedURL")
        or signed_response.get("signedUrl")
    )

    return remote_path, signed_url


def upload_audio(audio_path: str, request_id: str):
    local_file = Path(audio_path)

    if not local_file.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    remote_path = f"audio/{request_id}/{local_file.name}"

    with open(local_file, "rb") as file:
        supabase.storage.from_(BUCKET_NAME).upload(
            path=remote_path,
            file=file,
            file_options={
                "content-type": "audio/mpeg",
                "upsert": "true"
            }
        )

    signed_response = (
        supabase.storage
        .from_(BUCKET_NAME)
        .create_signed_url(remote_path, 3600)
    )

    signed_url = (
        signed_response.get("signedURL")
        or signed_response.get("signedUrl")
    )

    return remote_path, signed_url