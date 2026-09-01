import subprocess
from pathlib import Path


def get_duration(file_path: str):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )

    return float(result.stdout.strip())


def merge_video_audio(video_path: str, audio_path: str, request_id: str):
    output_dir = Path("media_output") / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{request_id}.mp4"

    video_duration = get_duration(video_path)
    audio_duration = get_duration(audio_path)

    print("Video duration:", video_duration)
    print("Audio duration:", audio_duration)

    padding = max(0, audio_duration - video_duration + 1)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-i", audio_path,
        "-filter_complex",
        f"[0:v]tpad=stop_mode=clone:stop_duration={padding}[v]",
        "-map", "[v]",
        "-map", "1:a",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-t", str(audio_duration),
        str(output_path)
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return str(output_path)