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

def generate_answer(question,context,grounded):
    if grounded:
        prompt = f"""You are a trading tutor. Below is lecture content and possibly additional web reference info, followed by a student's question.
                Lecture content: {context}
                Question: {question}
                Instructions:
                        - Answer the question directly, as if explaining it yourself in conversation.
                        - Synthesize the information fully in your own words — do not copy sentences, headers, formatting, or structure from the source text.
                        - Never mention specific creator names, channel names, brands, or where any information came from.
                        - Keep the answer as one unified, natural explanation — not separate sections.
                        - If the content is truly unrelated to the question, respond exactly: "This isn't covered in the lecture content I have."
                Answer:"""
    else:
        prompt = f"""You are a trading tutor. No matching lecture content was found for this question.
        Answer using your general trading knowledge, and mention that this isn't from the lecture material.
        Question: {question}Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def get_order_block_scene():
    scene = {
        "concept": "order_block",
        "candles": [
            {"id": 1,  "open": 70, "close": 66, "wickHigh": 71, "wickLow": 65},
            {"id": 2,  "open": 66, "close": 62, "wickHigh": 67, "wickLow": 60},
            {"id": 3,  "open": 62, "close": 59, "wickHigh": 63, "wickLow": 57},
            {"id": 4,  "open": 59, "close": 61, "wickHigh": 62, "wickLow": 57},
            {"id": 5,  "open": 61, "close": 57, "wickHigh": 62, "wickLow": 55},
            {"id": 6,  "open": 57, "close": 54, "wickHigh": 58, "wickLow": 52},
            {"id": 7,  "open": 54, "close": 50, "wickHigh": 55, "wickLow": 48},
            {"id": 8,  "open": 50, "close": 55, "wickHigh": 56, "wickLow": 49},
            {"id": 9,  "open": 55, "close": 60, "wickHigh": 61, "wickLow": 54},
            {"id": 10, "open": 60, "close": 64, "wickHigh": 65, "wickLow": 59},
            {"id": 11, "open": 64, "close": 68, "wickHigh": 69, "wickLow": 63},
            {"id": 12, "open": 68, "close": 72, "wickHigh": 73, "wickLow": 67},
            {"id": 13, "open": 72, "close": 75, "wickHigh": 76, "wickLow": 70},
            {"id": 14, "open": 75, "close": 71, "wickHigh": 76, "wickLow": 69},
            {"id": 15, "open": 71, "close": 66, "wickHigh": 72, "wickLow": 64},
            {"id": 16, "open": 66, "close": 60, "wickHigh": 67, "wickLow": 58},
            {"id": 17, "open": 60, "close": 54, "wickHigh": 61, "wickLow": 52},
            {"id": 18, "open": 54, "close": 58, "wickHigh": 59, "wickLow": 49},
            {"id": 19, "open": 58, "close": 62, "wickHigh": 63, "wickLow": 57},
            {"id": 20, "open": 62, "close": 66, "wickHigh": 67, "wickLow": 61},
            {"id": 21, "open": 66, "close": 70, "wickHigh": 71, "wickLow": 65},
            {"id": 22, "open": 70, "close": 74, "wickHigh": 75, "wickLow": 69}
        ],
        "steps": [
            {"actions": [{"type": "reveal_candles", "range": [1, 6]}]},
            {"actions": [{"type": "reveal_candle_with_pulse", "candle_id": 7}]},
            {"actions": [{"type": "reveal_candles", "range": [8, 13]}]},
            {"actions": [{"type": "draw_rectangle", "from_candle": 7}]},
            {"actions": [{"type": "reveal_candles", "range": [14, 17]}]},
            {"actions": [{"type": "reveal_candle_with_pulse", "candle_id": 18, "highlight_zone": True}]},
            {"actions": [{"type": "reveal_candles_expand_up", "range": [19, 22]}]}
        ]
    }
    return scene

def get_scene_script(chunk, score, threshold):
    if score >= threshold and chunk.get("concept") == "ob":
        return get_order_block_scene()
    return None

def save_scene_script(scene):
    output_path = "scene_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=2)


if __name__ == "__main__":
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
                print(generate_answer(query,context,grounded=True))    
            else:
                context = best_chunk['text']
                print(generate_answer(query,context,grounded=True))
        else:  
            context=web_text
            print(generate_answer(query,context,grounded=False))
    scene = get_scene_script(best_chunk, best_chunk_score, similarity_threshold)
    print(scene)
    if scene:
        save_scene_script(scene)
        