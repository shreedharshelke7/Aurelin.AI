import os 
from google import genai
from dotenv import load_dotenv
from google.genai import types

load_dotenv()
client = genai.Client(api_key= os.getenv("embed_api_key"))

def embed_chunks(query):
        result = client.models.embed_content(model="gemini-embedding-001",contents=query,
                                        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"))
        if result.embeddings is not None:
            return result.embeddings[0].values
        return None
