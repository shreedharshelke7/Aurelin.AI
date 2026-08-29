def save_LLM2_output(code):
    import os
    os.makedirs("media_output", exist_ok=True)
    file_path = "media_output/generated_scene.py"
    with open(file_path,"w",encoding="utf-8") as f:
        f.write(code)
        return file_path
