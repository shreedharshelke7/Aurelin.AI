import json 
import os 
from google import genai
from dotenv import load_dotenv
from google.genai import types
from time import sleep
load_dotenv()
client = genai.Client(api_key= os.getenv("gemini_api_key"))

def embed_chunks(input_path,output_path):

    with open(input_path ,"r",encoding="utf-8") as f:
        chunks = json.load(f)

    if os.path.exists(output_path):
        with open(output_path,"r",encoding="utf-8") as e:
            embed_chunks = json.load(e)
            last_chunk_id = embed_chunks[-1]['id']
    else:
        embed_chunks=[]
        last_chunk_id=0


    RPM = 0
    for chunk in chunks:
        if chunk['id'] > last_chunk_id:
            if(RPM==99):
                sleep(61)
                RPM = 0 
            result = client.models.embed_content(model="gemini-embedding-001",contents=chunk["text"],
                                                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"))
            if(result.embeddings is None):
                continue
            chunk["embedding"] = result.embeddings[0].values
            embed_chunks.append(chunk)
            RPM+=1



    with open(output_path,"w",encoding="utf-8") as f:
        json.dump(embed_chunks,f,indent=2)
    print(f"{len(embed_chunks)} Saved to {output_path} \nold chunks count : {last_chunk_id} -> new chunks added : {embed_chunks[-1]['id']-last_chunk_id}")

if __name__ == "__main__":
    embed_chunks("Data/chunks.json","Data/embeddings.json")
