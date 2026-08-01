import sys
import os
import subprocess
import re

Allowed_concepts = ["liquidity_grab","ob","fvg","mmxm","bb","pb"]


def srt_to_timestamped_text(srt_path, output_txt_path, window_seconds=45):
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    segments = []

    for block in blocks:
        lines = block.split("\n")
        if len(lines) < 3:
            continue
        timestamp_line = lines[1]
        text = " ".join(lines[2:]).strip()

        match = re.match(r"(\d{2}):(\d{2}):(\d{2}),\d{3}", timestamp_line)
        if match:
            hh, mm, ss = match.groups()
            total_seconds = int(hh) * 3600 + int(mm) * 60 + int(ss)
            segments.append((total_seconds, text))

    lines_out = []
    window_start = None
    buffer_text = []

    for seconds, text in segments:
      if window_start is None:
          window_start = seconds

      if seconds - window_start >= window_seconds:
        mm, ss = divmod(window_start, 60)
        lines_out.append(f"[{mm:02d}:{ss:02d}] {' '.join(buffer_text)}")
        buffer_text = []
        window_start = seconds
      buffer_text.append(text)

    if buffer_text:
      mm, ss = divmod(window_start,60)
      lines_out.append(f"[{mm:02d}:{ss:02d}] {' '.join(buffer_text)}")

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))

    return output_txt_path


def get_concept_name(input_source,args):
    if "--name" in  args:
        idx = args.index("--name")
        name = args[idx+1]
    else:
        filename = os.path.splitext(os.path.basename(input_source))[0].lower()
        match=None
        for concept in Allowed_concepts:
            if concept in filename:
                match=concept
                break
        if match is None:
            print(f"Error : No Known concept found in filename : {filename} . Allowed : {Allowed_concepts}")
            sys.exit(1)
        name = match
    if name not in Allowed_concepts:
        print(f"name : {name} not avaliable use from [ {Allowed_concepts} ]")
        sys.exit(1)
    return name

def extract_audio(input_source,is_link,output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if is_link:
        subprocess.run([
            "yt-dlp", "-x", "--audio-format", "mp3",
            "-o", output_path, input_source
        ], check=True)
    else:
        subprocess.run(["ffmpeg", "-i", input_source,"-q:a", "0", "-map", "a", output_path, "-y"], check=True)
    
def get_transcript(audio_source,concept_name):
    output_dir = "Data/srt"
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run([
        "whisper",audio_source,"--model","base",
        "--output_format","srt","--output_dir",output_dir
    ],check=True)
    srt_path = os.path.join(output_dir, f"{concept_name}.srt")
    return srt_path


def is_youtube_link(input_str):
    return input_str.startswith("http") and ("youtube.com" in input_str or "youtu.be" in input_str)

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_lecture.py <video_path_or_link> [--name <concept_name>]")
        sys.exit(1)

    input_source = sys.argv[1]
    args = sys.argv[1:]
    print(f"Input received: {input_source}")
    print(f"Is YouTube link? {is_youtube_link(input_source)}")
    concept_name = get_concept_name(input_source,args)
    print(f"concept name : {concept_name}")

    audio_output_path = f"Data/audio/{concept_name}.mp3"
    transcript_path = f"Data/transcripts/{concept_name}.txt"

    extract_audio(input_source,is_youtube_link(input_source),audio_output_path)
    print(f"Audio saved to {audio_output_path}")

    srt_path = get_transcript(audio_output_path,concept_name)
    srt_to_timestamped_text(srt_path,transcript_path)
    print(f"Transcript saved to : {transcript_path}")

    
    
if __name__ == "__main__":
    main()