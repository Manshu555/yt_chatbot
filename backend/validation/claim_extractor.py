import json
import uuid
import logging
from typing import List
from config import GEMINI_API_KEY, MAX_CLAIMS
from schemas.models import Claim, TranscriptChunk
from prompts.claims import CLAIM_EXTRACTION_PROMPT
from google import genai
from google.genai.errors import APIError
import os

MODEL_ID = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

async def extract_claims(query: str, video_evidence: List[TranscriptChunk]) -> List[Claim]:
    context = "\n".join([chunk.text for chunk in video_evidence])
    prompt = CLAIM_EXTRACTION_PROMPT.format(query=query, context=context)
    
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY missing. Falling back to single claim.")
        return [Claim(id="claim_1", text=query)]

    client = genai.Client(api_key=GEMINI_API_KEY)
    
    from rag.gemini_client import call_gemini_with_retry
    
    try:
        response = await call_gemini_with_retry(
            client=client,
            model_id=MODEL_ID,
            prompt=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=300,
                top_p=0.95,
                automatic_function_calling={"disable": True}
            )
        )
        
        if not response or not response.text:
            print("Empty response from Gemini claim extraction.")
            return [Claim(id="claim_1", text=query)]
            
        answer = response.text.strip()
            
        start_idx = answer.find('[')
        end_idx = answer.rfind(']')
        if start_idx != -1 and end_idx != -1:
            json_str = answer[start_idx:end_idx+1]
            data = json.loads(json_str)
            claims = []
            for item in data[:MAX_CLAIMS]:
                claims.append(Claim(id=item.get("id", str(uuid.uuid4())), text=item.get("text", "")))
            if claims:
                return claims
            
    except APIError as e:
        print(f"Gemini API error extracting claims: {e.message}")
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from claim extraction: {e}")
    except TimeoutError:
        print("Gemini API request timed out extracting claims.")
    except Exception as e:
        print(f"Error extracting claims: {e}")
        
    return [Claim(id="claim_1", text=query)]
