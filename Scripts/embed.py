import json 
import os 
from google import genai
from dotenv import load_dotenv
from google.genai import types

load_dotenv()
client = genai.Client(api_key= os.getenv("gemini_api_key"))

def embed_chunks(input_path,output_path):
    with open(input_path ,"r",encoding="utf-8") as f:
        chunks = json.load(f)
    embedded_chunks=[]
    for chunk in chunks:
        result = client.models.embed_content(model="gemini-embedding-001",contents=chunk["text"],
                                             config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"))
        if(result.embeddings is None):
            continue
        chunk["embedding"] = result.embeddings[0].values
        embedded_chunks.append(chunk)

    with open(output_path,"w",encoding="utf-8") as f:
        json.dump(embedded_chunks,f,indent=2)

    print(f"{len(embedded_chunks)} Saved to {output_path}")

embed_chunks("Data/chunks.json","Data/embeddings.json")