import subprocess
import shutil
import os

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
        return False, result.stderr

    if os.path.exists("media"):
        shutil.rmtree("media")

    return True, None