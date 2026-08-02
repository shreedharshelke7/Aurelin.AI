import json
import re
import os
import sys

def get_concept_name(input_path):
    concept = os.path.splitext(os.path.basename(input_path))[0].lower()
    return concept



def chunk_transcript(input_path, output_path):
    concept_name = get_concept_name(input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    pattern = r"(\[\d{2}:\d{2}\])"
    parts = re.split(pattern, text)
    if os.path.exists(output_path):
        with open(output_path,"r",encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        chunks = []

    chunk_id = len(chunks)+1

    for i in range(1, len(parts), 2):
        timestamp = parts[i].strip()
        content = parts[i + 1].strip()
        if content:
            chunks.append({
                "id": chunk_id,
                "concept": concept_name,
                "timestamp": timestamp,
                "text": content
            })
            chunk_id += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Created {len(chunks)} chunks. Saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chunk.py <input_transcript_path>")
        sys.exit(1)

    chunk_transcript(sys.argv[1], "data/chunks.json")