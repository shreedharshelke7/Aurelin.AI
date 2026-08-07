import json
import os
from google import genai
from dotenv import load_dotenv
import numpy as np 
from google.genai import types
from groq import Groq
from tavily import TavilyClient

load_dotenv()
client = genai.Client(api_key=os.getenv("gemini_api_key"))
groq_client = Groq(api_key=os.getenv("groq_api_key"))
tavily_client = TavilyClient(api_key=os.getenv("tavily_api_key"))




def generate_answer_and_scene(question, context):
    prompt = f"""You are a trading tutor generating both a spoken explanation and an animated chart scene.
        Lecture content: {context}
        Question: {question}
        Generate a JSON object with this EXACT structure, and output ONLY valid JSON, nothing else:
        {{
        "narration": "A clear spoken explanation answering the question, in your own words, based on the lecture content.",
        "candles": [
            {{"id": 1, "open": <number>, "close": <number>, "wickHigh": <number>, "wickLow": <number>}},
            ... (15-25 candles total, forming a realistic price pattern relevant to the concept)
        ],
        "steps": [
            {{"actions": [{{"type": "reveal_candles", "range": [1, 5]}}], "narration_snippet": "short text for this step"}},
            ... (use action types: reveal_candles, reveal_candle, reveal_candle_with_pulse, draw_rectangle, reveal_candles_expand_up)
        ]
        }}
        Rules:
        - wickHigh must be >= max(open, close). wickLow must be <= min(open, close).
        - Each candle's open should roughly continue from the previous candle's close.
        - draw_rectangle needs a "from_candle" field (candle id).
        - Output ONLY the JSON object, no explanation, no markdown formatting.
        """
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content



def web_search_and_embed(question):
    response = tavily_client.search(query=question, max_results=1)
    if not response["results"]:
        return None, None

    web_text = response["results"][0]["content"]

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=web_text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    web_embedding=None
    if result.embeddings is not None:
        web_embedding = result.embeddings[0].values

    return web_text, web_embedding


def load_chunks(path):
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)

def query_text(text):
    result = client.models.embed_content(model="gemini-embedding-001",contents=text,
                                         config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"))
    if result.embeddings is None:return None
    return result.embeddings[0].values

def cosine_similarity(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a,b)/(np.linalg.norm(a)* np.linalg.norm(b))

def find_best_chunk(question,chunks):
    query_vector = query_text(question)

    best_chunk=None
    best_score=-1
    for chunk in chunks:
        score = cosine_similarity(query_vector,chunk["embedding"])
        if score > best_score:
            best_score=score
            best_chunk = chunk
    return best_chunk,best_score





if __name__=="__main__":
    similarity_threshold = 0.65
    chunks= load_chunks("Data/embeddings.json")
    query = input("Ask Question : ")
    best_chunk, best_chunk_score = find_best_chunk(query, chunks)
    web_text,web_embedding = web_search_and_embed(query)
    if best_chunk is not None:
        web_chunk_score = cosine_similarity(web_embedding,best_chunk['embedding'])
        print(f"Matched chunk: [{best_chunk['timestamp']}] {best_chunk['text']}")
        print(f"best chunk score : {best_chunk_score}")
        print(f"Web Search : {web_text}")
        print(f"web search chunk score : {web_chunk_score}")
        if best_chunk_score >= similarity_threshold:
            if web_chunk_score >= similarity_threshold:
                context = f"Transcript info:\n{best_chunk['text']}\n\nAdditional web info:\n{web_text}"
                print(generate_answer_and_scene(query,context))    
            else:
                context = best_chunk['text']
                print(generate_answer_and_scene(query,context))
        else:  
            context=web_text
            print(generate_answer_and_scene(query,context))
