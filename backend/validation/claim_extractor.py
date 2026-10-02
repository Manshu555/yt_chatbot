import json
from typing import List
from pydantic import BaseModel, Field, ValidationError
from config import GEMINI_API_KEY, MAX_CLAIMS
from schemas.models import Claim, TranscriptChunk
from prompts.claims import CLAIM_EXTRACTION_PROMPT
from google import genai
from google.genai.errors import APIError
import os
from rag.gemini_client import call_gemini_with_retry

MODEL_ID = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

class ExtractedClaim(BaseModel):
    id: str = Field(description="Unique identifier for the claim (e.g., 'claim_1')")
    text: str = Field(description="The factual claim text")

class ClaimExtractionResponse(BaseModel):
    claims: List[ExtractedClaim]

def _is_daily_quota_error(error: Exception) -> bool:
    normalized = str(error).lower().replace(" ", "").replace("_", "").replace("-", "")
    return "perday" in normalized or "requestsperday" in normalized or "daily" in normalized

def _fallback_claim(query: str, reason: str) -> List[Claim]:
    print(f"Claim extraction failed: {reason}. Falling back to UNVERIFIED-safe claim.")
    return [Claim(id="claim_extraction_failed", text=query)]

def _response_to_claim_extraction(response) -> ClaimExtractionResponse:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, ClaimExtractionResponse):
        return parsed
    if isinstance(parsed, dict):
        return ClaimExtractionResponse.model_validate(parsed)

    text = getattr(response, "text", None)
    if not text:
        raise ValueError("empty response")
    return ClaimExtractionResponse.model_validate_json(text)

def _normalize_claims(extracted: ClaimExtractionResponse) -> List[Claim]:
    claims = []
    seen_text = set()
    seen_ids = set()

    for index, item in enumerate(extracted.claims[:MAX_CLAIMS], start=1):
        claim_text = (item.text or "").strip()
        if not claim_text:
            continue
        normalized_text = " ".join(claim_text.lower().split())
        if normalized_text in seen_text:
            continue
        seen_text.add(normalized_text)

        claim_id = (item.id or f"claim_{index}").strip()
        if not claim_id or claim_id in seen_ids:
            claim_id = f"claim_{index}"
        seen_ids.add(claim_id)
        claims.append(Claim(id=claim_id, text=claim_text))

    return claims

async def extract_claims(query: str, video_evidence: List[TranscriptChunk]) -> List[Claim]:
    context = "\n".join([chunk.text for chunk in video_evidence])
    
    # We can simplify the prompt since we use structured output now,
    # but to preserve existing logic, we'll keep the prompt string and let Gemini map it to the schema.
    prompt = CLAIM_EXTRACTION_PROMPT.format(query=query, context=context)
    
    if not GEMINI_API_KEY:
        return _fallback_claim(query, "GEMINI_API_KEY missing")

    client = genai.Client(api_key=GEMINI_API_KEY)
    
    try:
        response = await call_gemini_with_retry(
            client=client,
            model_id=MODEL_ID,
            prompt=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=800,
                top_p=0.95,
                response_mime_type="application/json",
                response_schema=ClaimExtractionResponse,
                automatic_function_calling={"disable": True},
                thinking_config={"thinking_level": "low"}
            )
        )
        
        if not response:
            return _fallback_claim(query, "empty Gemini response object")

        extracted = _response_to_claim_extraction(response)
        claims = _normalize_claims(extracted)
        if claims:
            return claims
        return _fallback_claim(query, "schema contained no usable claims")
            
    except APIError as e:
        print(f"Gemini API error extracting claims: {getattr(e, 'message', str(e))}")
        if _is_daily_quota_error(e):
            raise
        return _fallback_claim(query, "Gemini API error")
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        return _fallback_claim(query, str(e))
    except TimeoutError:
        return _fallback_claim(query, "Gemini API request timed out")
    except Exception as e:
        return _fallback_claim(query, str(e))
