import json
import os
import numpy as np
from .embed import embed_chunks
from pathlib import Path
from pathlib import Path

CACHE_PATH = Path(__file__).resolve().parents[2] / "RAG" / "cache.json"

def new_cache_saver(question, embedding, context, satisfied=True, path=CACHE_PATH):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = []

    cache.append({
        "question": question,
        "embedding": embedding,
        "context": context,
        "satisfied": satisfied
    })

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def replace_unsatisfied_entry(question, embedding,satisfied, new_context, path=CACHE_PATH):

    with open(path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    best_chunk={}
    max_score = 0
    for entry in cache:
        score = cosine_similarity(embedding,entry['embedding'])
        if score > max_score:
            max_score = score
            best_chunk = entry
    best_chunk["question"] = question
    best_chunk["embedding"] = embedding
    best_chunk["context"] = new_context
    best_chunk["satisfied"] = satisfied
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    return    

def cosine_similarity(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a,b)/(np.linalg.norm(a)* np.linalg.norm(b))

#def overwrite_satisfied(best_chunk):
    

def match_question(question):
    path = CACHE_PATH
    question_embedding = embed_chunks(question)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = []

    best_entry = None
    best_score = -1

    for entry in cache:
        score = cosine_similarity(question_embedding, entry["embedding"])
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry and best_score > 0.90:
        if best_entry["satisfied"]:
            return best_entry  # usable cache hit
        else:
            # unsatisfied match found — needs replace, not append
            return {"question": question, "embedding": question_embedding, "context": None, "satisfied": None, "needs_replace": True}

    # genuinely new question, no match at all
    return {"question": question, "embedding": question_embedding, "context": None, "satisfied": None, "needs_replace": False}


def feedback_cache(result, question, embedding, context, needs_replace=False):
    if not result:
        print("Visualization Failed !!!")
        return

    print("visualized successfully")
    feedback = input("( yes / no ) : ").lower()

    if feedback == "no":
        if needs_replace:
            replace_unsatisfied_entry(question, embedding, False, context)
        else:
            new_cache_saver(question, embedding, context, satisfied=False)
    else:
        if needs_replace:
            replace_unsatisfied_entry(question, embedding, True, context)
        else:
            new_cache_saver(question, embedding, context, satisfied=True)