from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Rag_op import match_question, feedback_cache
from url_Crawler import scrape_page
from web_url_scrape import url_fetch
from main import call_agents, correct_spelling
from supabase_client import supabase
from storage_manager import upload_video


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    input_query: str


def update_request(request_id, data):
    supabase.table("explanations").update(data).eq(
        "request_ID",
        request_id
    ).execute()


@app.post("/ask")
def ask(payload: Question):

    request_id = None

    try:
        # -------------------------
        # 1. SPELL CHECK
        # -------------------------

        input_query = payload.input_query
        question = correct_spelling(input_query)

        # -------------------------
        # 2. CREATE REQUEST
        # -------------------------

        response = (
            supabase
            .table("explanations")
            .insert({
                "question": question,
                "status": "processing",
                "stage": "started",
                "narration": None,
                "generated_video_path": None,
                "error": None
            })
            .execute()
        )

        request_id = response.data[0]["request_ID"]

        # -------------------------
        # 3. CACHE CHECK
        # -------------------------

        update_request(
            request_id,
            {
                "stage": "cache_check"
            }
        )

        best_chunk = match_question(question)

        # -------------------------
        # 4. RETRIEVAL
        # -------------------------

        if best_chunk["context"] is None:

            update_request(
                request_id,
                {
                    "stage": "retrieval"
                }
            )

            context = ""

            for url in url_fetch(question):
                context += " | " + scrape_page(
                    url,
                    question
                )

        else:
            context = best_chunk["context"]

        # -------------------------
        # 5. AI + MANIM PIPELINE
        # -------------------------

        update_request(
            request_id,
            {
                "stage": "generation"
            }
        )

        success, narration, video_path, error = call_agents(
            question,
            context
        )

        # -------------------------
        # 6. CACHE FEEDBACK
        # -------------------------

        feedback_cache(
            success,
            question,
            best_chunk["embedding"],
            context,
            best_chunk.get("needs_replace", False)
        )

        # -------------------------
        # 7. PIPELINE FAILED
        # -------------------------

        if not success:

            update_request(
                request_id,
                {
                    "status": "failed",
                    "stage": "failed",
                    "narration": narration,
                    "error": error
                }
            )

            return {
                "request_id": request_id,
                "success": False,
                "status": "failed",
                "stage": "failed",
                "narration": narration,
                "video_url": None,
                "error": error
            }

        # -------------------------
        # 8. UPLOAD VIDEO
        # -------------------------

        update_request(
            request_id,
            {
                "stage": "uploading"
            }
        )

        remote_path, signed_url = upload_video(
            video_path,
            request_id
        )

        # -------------------------
        # 9. COMPLETE
        # -------------------------

        update_request(
            request_id,
            {
                "status": "completed",
                "stage": "complete",
                "narration": narration,
                "generated_video_path": remote_path,
                "error": None
            }
        )

        return {
            "request_id": request_id,
            "success": True,
            "status": "completed",
            "stage": "complete",
            "narration": narration,
            "video_url": signed_url,
            "error": None
        }

    except Exception as e:

        error_message = str(e)

        if request_id:
            try:
                update_request(
                    request_id,
                    {
                        "status": "failed",
                        "stage": "failed",
                        "error": error_message
                    }
                )
            except Exception:
                pass

        return {
            "request_id": request_id,
            "success": False,
            "status": "failed",
            "stage": "failed",
            "narration": None,
            "video_url": None,
            "error": error_message
        }