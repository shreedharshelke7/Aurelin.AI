import json
import os
from google import genai
from dotenv import load_dotenv
import numpy as np 
from google.genai import types
from groq import Groq
from url_Crawler import scrape_page
from web_url_scrape import url_fetch
from spellchecker import SpellChecker
from visualize import run_manim,save_LLM2_output

load_dotenv()
spell = SpellChecker()
client = genai.Client(api_key=os.getenv("embed_api_key"))
LLM1_client = Groq(api_key=os.getenv("LLM1_api_key"))
LLM2_client = genai.Client(api_key=os.getenv("LLM2_api_key"))

def query_text(text):
    result = client.models.embed_content(model="gemini-embedding-001",contents=text,
                                         config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"))
    if result.embeddings is None:return None
    return result.embeddings[0].values

def cosine_similarity(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a,b)/(np.linalg.norm(a)* np.linalg.norm(b))


def LLM1_Narration_scene(question, context):
    prompt = """You are a math visualization planner. Your job is to describe an animated explanation
    that a SEPARATE AI (which only writes Manim Python code, and cannot see the source material) will build from your description alone.

    Reference content: {}

    Question: {}

    Output ONLY valid JSON, no other text, in this exact structure:
    {{
        "narration": "Full spoken explanation of the concept, in your own words.",
        "objects": [
            {{
                "name": "<unique_id, e.g. triangle_main>",
                "shape": "<triangle|square|line|label|equation|circle>",
                "persists_until_step": <step number or "end">
            }}
            ],
        "scene_steps": [
            {{
                "step": 1,
                "action": "<one of: draw, label, transform, fade_out, highlight, show_equation>",
                "target": "<object name from 'objects' list>",
                "relation": "<if applicable>",
                "narration_snippet": "<short text spoken during this step>"
            }}
        ]
    }}

    Rules:
    1. Declare every shape in "objects" FIRST, with a unique lowercase_snake_case name.
    2. Every "relation" must be specific enough to compute position/size without guessing.
    3. Break actions into the smallest reasonable unit.
    4. Every object drawn must later have a "fade_out" step OR be marked "persists_until_step": "end".
    5. Use "transform" only when one object visually morphs.
    6. Keep total steps between 5 and 12.
    Output ONLY the JSON object.""".format(context, question)
    
    response = LLM1_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def LLM2_exe_code(LLM1_ouput):
    rules = """You are LLM2 of AURELIN AI.
    Your only task is to convert the provided narration and scene descriptions into valid executable Manim Community Edition Python code.
    RULES:
    1. Return ONLY Python code.
    2. Do not use markdown code fences.
    3. Do not add explanations, comments outside the code, or extra text.
    4. Always start with:
    from manim import *
    5. Generate exactly one Scene class named AurelinScene .
    6. The Scene class must contain:def construct(self):
    7. Follow the narration and scene descriptions in the exact given order.
    8. Every described visual element must appear in the animation.
    9. Use appropriate Manim objects such as:
    - Text
    - MathTex
    - Line
    - Polygon
    - Circle
    - Square
    - Arrow
    - VGroup
    - Axes
    - NumberPlane
    10. Use Manim animations such as:
        - Create
        - Write
        - FadeIn
        - FadeOut
        - Transform
        - ReplacementTransform
    11. Keep all objects inside the visible Manim frame.
    12. Prevent text, equations, labels, and diagrams from overlapping.
    13. Use relative positioning methods whenever possible:
        next_to()
        arrange()
        to_edge()
        shift()
        move_to()
    14. Mathematical expressions must use MathTex whenever appropriate.
    15. Do not use external images, files, internet resources, or APIs.
    16. Do not require any Python package other than Manim and Python standard libraries.
    17. Do not generate narration/audio code.
    18. Do not generate file handling or subprocess code.
    19. Do not generate interactive input.
    20. The final code must be directly executable by Manim without manual modification.
    21. Treat the supplied reference code as a style and implementation reference.
    22. Do not blindly copy the reference code.
    23. Adapt the reference implementation according to the current narration and scenes.
    24. Prefer simple, reliable Manim implementations over unnecessarily complex animations.
    25. Before producing the final answer, internally verify:
        - syntax is valid
        - all variables are defined
        - Manim methods/classes exist
        - scene order matches the input
        - objects remain visible
        - no obvious overlaps occur
    26. after every step components from previous step which will not be used or necessary on further step remove those components.
    OUTPUT:
    Return only the complete executable Manim Python source code.
    """
    prompt = f"""RULES:{rules} NARRATION and scene : {LLM1_ouput} Generate the complete executable Manim Python code.Return only Python code."""
    response = LLM2_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text

def correct_spelling(input_text):
    words = list(input_text.split())
    Question = ""
    for word in words:
        miss_spelled = spell.unknown([word])
        if miss_spelled:
            corrected_word = str(spell.correction(word))
            Question += corrected_word
            Question += " "
            continue
        Question += word
        Question += " "
    return Question 


if __name__ == "__main__":
    input_text = input("Question : ")
    Question =correct_spelling(input_text)       
    print(Question)
    result = " "
    urls = url_fetch(Question)
    for url in urls:
        print(url)
        result +=" | "
        result +=scrape_page(url,Question)
    LLM1_ouput = LLM1_Narration_scene(Question,result)
    code = LLM2_exe_code(LLM1_ouput)
    file_path = save_LLM2_output(code,Question)
    result = run_manim(file_path)
    if result:
        print("visualized successfully")
    else:
        print("Visualization Failed !!!")