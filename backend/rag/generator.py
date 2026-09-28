import json
import logging
import os
from config import GEMINI_API_KEY
from schemas.models import TranscriptChunk, ClaimValidation, ExternalEvidence
from typing import List, Optional
from google import genai
from google.genai.errors import APIError

MODEL_ID = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

def load_model():
    """Validate Gemini API access."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    print(f"Gemini API initialized for {MODEL_ID}.")

async def generate_answer(
    query: str, 
    video_evidence: List[TranscriptChunk], 
    validation: Optional[List[ClaimValidation]] = None,
    external_evidence: Optional[List[ExternalEvidence]] = None
) -> str:
    """Query Gemini API for final answer."""
    
    video_context = "\n".join([chunk.text for chunk in video_evidence])
    
    if not validation:
        prompt = f"""Using the following context, answer the question.
If the answer is not in the context, say "I could not find the answer in the transcript."
Context:
{video_context}
Question: {query}
"""
    else:
        from prompts.answer import FINAL_ANSWER_PROMPT
        
        val_str = json.dumps([v.model_dump() for v in validation], indent=2)
        ext_str = json.dumps([e.model_dump() for e in external_evidence], indent=2) if external_evidence else "[]"
        
        prompt = FINAL_ANSWER_PROMPT.format(
            query=query,
            video_evidence=video_context,
            validation=val_str,
            external_evidence=ext_str
        )

    client = genai.Client(api_key=GEMINI_API_KEY)
    
    from rag.gemini_client import call_gemini_with_retry

    try:
        response = await call_gemini_with_retry(
            client=client,
            model_id=MODEL_ID,
            prompt=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=200,
                top_p=0.95,
                automatic_function_calling={"disable": True}
            )
        )
        
        if not response or not response.text:
            return "No valid response from the API."
            
        answer = response.text.strip()

        if "Answer:" in answer and not validation:
            answer_start_index = answer.find("Answer:")
            if answer_start_index != -1:
                answer = answer[answer_start_index + len("Answer:"):].strip()
            
        return answer or "No valid response from the API."

    except APIError as e:
        print(f"Gemini API error: {e.message}")
        return "Error generating response from the API."
    except TimeoutError:
        print("Gemini API request timed out.")
        return "Error generating response from the API."
    except Exception as e:
        print(f"Unexpected error calling Gemini API: {e}")
        return "Error generating response from the API."
