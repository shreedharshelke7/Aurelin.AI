import subprocess
import shutil
import os
import re

def run_manim(file_path, scene_name="AurelinScene"):
    cmd = [
        "python", "-m", "manim",
        "-pql",
        "--media_dir", "media_output",
        file_path,
        scene_name
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Manim stopped : \n", result.stderr)
        return False, result.stderr, None

    if os.path.exists("media"):
        shutil.rmtree("media")

    script_name = os.path.splitext(os.path.basename(file_path))[0]
    video_path = os.path.join("media_output", "videos", script_name, "480p15", f"{scene_name}.mp4")

    return True, None, video_path
