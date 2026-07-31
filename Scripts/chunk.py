import json
import re
import os

def chunk_transcript(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    pattern = r"(\[\d{2}:\d{2}\])"
    parts = re.split(pattern, text)
    chunks = []
    chunk_id = 1
    for i in range(1, len(parts), 2):
        timestamp = parts[i].strip()
        content = parts[i + 1].strip()
        if content:
            chunks.append({
                "id": chunk_id,
                "timestamp": timestamp,
                "text": content
            })
            chunk_id += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Created {len(chunks)} chunks. Saved to {output_path}")

if __name__ == "__main__":
    chunk_transcript("Data/transcripts/OB.txt", "data/chunks.json")