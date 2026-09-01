from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Rag_op import match_question
from url_Crawler import scrape_page
from web_url_scrape import url_fetch
from main import call_agents, correct_spelling

from supabase_client import supabase
from storage_manager import upload_video
from tts_manager import generate_audio
from media_merger import merge_video_audio


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://172.20.10.2:3000"
    ],
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
                "generated_audio_path": None,
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
            {"stage": "cache_check"}
        )

        best_chunk = match_question(question)

        # -------------------------
        # 4. RETRIEVAL
        # -------------------------

        if best_chunk["context"] is None:

            update_request(
                request_id,
                {"stage": "retrieval"}
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
            {"stage": "generation"}
        )

        success, narration, video_path, error = call_agents(
            question,
            context
        )

        # -------------------------
        # 6. PIPELINE FAILED
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
        # 7. GENERATE TTS AUDIO
        # -------------------------

        update_request(
            request_id,
            {"stage": "tts_generation"}
        )

        audio_path = generate_audio(
            narration,
            request_id
        )

        # -------------------------
        # 8. MERGE VIDEO + AUDIO
        # -------------------------

        update_request(
            request_id,
            {"stage": "media_merge"}
        )

        final_video_path = merge_video_audio(
            video_path,
            audio_path,
            request_id
        )

        # -------------------------
        # 9. UPLOAD FINAL VIDEO
        # -------------------------

        update_request(
            request_id,
            {"stage": "uploading"}
        )

        remote_path, signed_url = upload_video(
            final_video_path,
            request_id
        )

        # -------------------------
        # 10. COMPLETE
        # -------------------------

        update_request(
            request_id,
            {
                "status": "completed",
                "stage": "complete",
                "narration": narration,
                "generated_video_path": remote_path,
                "generated_audio_path": None,
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