import os
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from schemas.models import QueryInput, AskResponse
from routing.query_analyzer import requires_external_validation
from rag.retriever import retrieve_transcript
from rag.generator import load_model, generate_answer, AnswerGenerationError
from google.genai.errors import APIError
from search.web_search import search_web
from validation.claim_extractor import extract_claims
from validation.validator import validate_claims
from config import TOP_K_WEB

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

load_model()


@app.exception_handler(APIError)
async def gemini_api_error_handler(request: Request, error: APIError):
    # Keep provider payloads (which may contain credentials) out of the response.
    if error.code == 429:
        status_code = 429
        message = "Gemini quota or rate limit exceeded. Check your Gemini quota/billing or retry later."
    elif error.code == 503:
        status_code = 503
        message = "Gemini is temporarily unavailable after bounded retries. Please try again later."
    elif error.code in (401, 403):
        status_code = 502
        message = "Gemini authentication failed. Check the backend's GEMINI_API_KEY and API access."
    elif error.code == 404:
        status_code = 502
        message = "The configured Gemini model is unavailable. Check GEMINI_MODEL in backend/.env."
    else:
        status_code = 502
        message = "Gemini could not process the request. Check the backend's Gemini configuration."
    return JSONResponse(status_code=status_code, content={"error": message})


@app.exception_handler(AnswerGenerationError)
async def answer_generation_error_handler(request: Request, error: AnswerGenerationError):
    return JSONResponse(status_code=error.status_code, content={"error": str(error)})


def require_video_evidence(video_evidence):
    if not video_evidence:
        raise HTTPException(
            status_code=422,
            detail="Could not retrieve the video's English transcript. "
            "Check subtitle availability, YouTube cookies, and the backend's network access.",
        )

def aggregate_status(validations):
    if not validations:
        return "UNVERIFIED"
    statuses = [v.status for v in validations]
    if "CONTRADICTED" in statuses:
        return "CONTRADICTED"
    if "PARTIALLY_SUPPORTED" in statuses:
        return "PARTIALLY_SUPPORTED"
    if all(s == "SUPPORTED" for s in statuses):
        return "SUPPORTED"
    if "SUPPORTED" in statuses:
        return "PARTIALLY_SUPPORTED"
    return "UNVERIFIED"

def collect_evidence(validations):
    evidence = []
    for v in validations:
        evidence.extend(v.supporting_evidence)
        evidence.extend(v.contradicting_evidence)
    return evidence

@app.post("/ask", response_model=AskResponse)
async def ask_question(data: QueryInput):
    validation_required = requires_external_validation(
        data.query,
        data.validate_externally
    )

    if not validation_required:
        video_evidence = await retrieve_transcript(
            data.query,
            data.video_id
        )
        require_video_evidence(video_evidence)

        answer = await generate_answer(
            query=data.query,
            video_evidence=video_evidence,
            validation=None,
            external_evidence=[]
        )

        return AskResponse(
            answer=answer,
            validation_required=False,
            validation_status=None,
            video_evidence=video_evidence,
            claims=[],
            sources=[]
        )

    # Parallel initial retrieval
    video_task = retrieve_transcript(
        data.query,
        data.video_id
    )

    web_task = search_web(
        data.query,
        top_k=TOP_K_WEB
    )

    video_evidence, web_results = await asyncio.gather(
        video_task,
        web_task
    )
    require_video_evidence(video_evidence)

    # Extract relevant claims
    claims = await extract_claims(
        data.query,
        video_evidence
    )

    # Validate claims
    validation = await validate_claims(
        claims,
        web_results
    )
    
    # Final answer — ONE LLM call
    answer = await generate_answer(
        query=data.query,
        video_evidence=video_evidence,
        validation=validation,
        external_evidence=collect_evidence(validation)
    )

    return AskResponse(
        answer=answer,
        validation_required=True,
        validation_status=aggregate_status(validation),
        video_evidence=video_evidence,
        claims=validation,
        sources=web_results
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
