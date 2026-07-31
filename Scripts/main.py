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

def generate_answer(question, chunk_text,grounded):
    if grounded:
        prompt = f"""You are a trading tutor. Below is a piece of lecture content and a student's question.
                Lecture content: {chunk_text}
                Question: {question}
                Instructions:
                        - Answer the question directly, as if explaining it yourself in conversation.
                        - Read the lecture content carefully. If it explains or relates to the question, answer using that content in your own words.
                        - Only if the lecture content is truly unrelated to the question, respond exactly: "This isn't covered in the lecture content I have."
                        - Do not add outside knowledge beyond what's in the lecture content.
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


def get_liquidity_grab_scene():
    return {
        "concept": "liquidity_grab",
        "old_high_level": 62,
        "candles": [
            { "open": 40, "close": 45, "wickHigh": 47, "wickLow": 39 },
            { "open": 46, "close": 49, "wickHigh": 50, "wickLow": 45 },
            { "open": 50, "close": 53, "wickHigh": 54, "wickLow": 47 },
            { "open": 53, "close": 50, "wickHigh": 55, "wickLow": 47 },
            { "open": 51, "close": 53, "wickHigh": 55, "wickLow": 49 },
            { "open": 53, "close": 55, "wickHigh": 56, "wickLow": 49 },
            { "open": 55, "close": 57, "wickHigh": 58, "wickLow": 54 },
            { "open": 56, "close": 54, "wickHigh": 55, "wickLow": 55 },
            { "open": 54, "close": 59, "wickHigh": 59, "wickLow": 53 },
            { "open": 59, "close": 61, "wickHigh": 62, "wickLow": 58 },
            { "open": 61, "close": 57, "wickHigh": 61, "wickLow": 58 },
            { "open": 57, "close": 55, "wickHigh": 58, "wickLow": 54 },
            { "open": 55, "close": 56, "wickHigh": 54, "wickLow": 57 },
            { "open": 56, "close": 54, "wickHigh": 57, "wickLow": 53 },
            { "open": 54, "close": 50, "wickHigh": 55, "wickLow": 49 },
            { "open": 50, "close": 48, "wickHigh": 51, "wickLow": 47 },
            { "open": 48, "close": 45, "wickHigh": 49, "wickLow": 44 },
            { "open": 45, "close": 49, "wickHigh": 50, "wickLow": 44 },
            { "open": 49, "close": 52, "wickHigh": 48, "wickLow": 54 },
            { "open": 51, "close": 50, "wickHigh": 52, "wickLow": 49 },
            { "open": 50, "close": 54, "wickHigh": 55, "wickLow": 49 },
            { "open": 54, "close": 56, "wickHigh": 57, "wickLow": 53 },
            { "open": 56, "close": 58, "wickHigh": 59, "wickLow": 55 },
            { "open": 58, "close": 59, "wickHigh": 60, "wickLow": 57 },
            { "open": 59, "close": 60, "wickHigh": 61, "wickLow": 58 },
            { "open": 60, "close": 57, "wickHigh": 70, "wickLow": 56 },
            { "open": 57, "close": 55, "wickHigh": 58, "wickLow": 52 },
            { "open": 55, "close": 50, "wickHigh": 56, "wickLow": 48 },
            { "open": 50, "close": 44, "wickHigh": 51, "wickLow": 42 },
            { "open": 44, "close": 38, "wickHigh": 45, "wickLow": 36 },
            { "open": 38, "close": 30, "wickHigh": 39, "wickLow": 28 },
            { "open": 30, "close": 24, "wickHigh": 31, "wickLow": 22 }
        ],
        "grab_candle_index": 25,
        "direction": "sellside_grab",
        "label": "Sell-side Liquidity Grab — Wick Spikes Through Old High, Then Reverses Down",
        "steps": [
            {
                "step_id": 1,
                "title": "range_forms",
                "elements": {
                    "candles_visible_upto": 15,
                    "old_high_line": False,
                    "grab_marker": False
                }
            },
            {
                "step_id": 2,
                "title": "old_high_marked",
                "elements": {
                    "candles_visible_upto": 15,
                    "old_high_line": True,
                    "grab_marker": False
                }
            },
            {
                "step_id": 3,
                "title": "liquidity_grab",
                "elements": {
                    "candles_visible_upto": 25,
                    "old_high_line": True,
                    "grab_marker": True
                },
                "highlight": "grab_candle_index"
            },
            {
                "step_id": 4,
                "title": "reversal",
                "elements": {
                    "candles_visible_upto": 31,
                    "old_high_line": True,
                    "grab_marker": True
                },
                "highlight": "grab_candle_index"
            }
        ]
    }


def get_scene_script(chunk, score, threshold):
    if score >= threshold and chunk.get("concept") == "liquidity_grab":
        return get_liquidity_grab_scene()
    return None

def save_scene_script(scene):
    output_path = "Data/scene_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=2)


if __name__ == "__main__":
    similarity_threshold = 0.70
    chunks= load_chunks("Data/embeddings.json")
    query = input("Ask Question : ")
    best_chunk, best_chunk_score = find_best_chunk(query, chunks)
    web_text,web_embedding = web_search_and_embed(query)

    if best_chunk is not None:
        print(f"Matched chunk: [{best_chunk['timestamp']}] {best_chunk['text']}")
        print(f"best chunk score : {best_chunk_score}")
        if best_chunk_score >= similarity_threshold:
            if web_embedding:
                web_chunk_score = cosine_similarity(web_embedding,best_chunk['embedding'])
                print(f"web search chunk score : {web_chunk_score}")
                print(f"{web_text}")
                #print(generate_answer(query,best_chunk["text"],grounded=True))
        else:
            print(generate_answer(query,None,grounded=False))
    scene = get_scene_script(best_chunk, best_chunk_score, similarity_threshold)
    if scene:
        save_scene_script(scene)
        