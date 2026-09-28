import asyncio
import re
from google import genai
from google.genai.errors import APIError

# Global semaphore to limit concurrent requests across the entire process
GEMINI_SEMAPHORE = asyncio.Semaphore(2)
MAX_RETRIES = 4
BASE_BACKOFF = 4.0  # seconds

def _parse_retry_delay(error_message: str) -> float:
    """Extract 'Please retry in X.Xs.' from the error message."""
    match = re.search(r"retry in ([\d\.]+)s", str(error_message))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return -1.0

async def call_gemini_with_retry(
    client: genai.Client, 
    model_id: str, 
    prompt: str, 
    config: genai.types.GenerateContentConfig
):
    """
    Calls the Gemini API with centralized concurrency limits and exponential backoff.
    Handles 429 (Quota/Rate Limit) and 503 (High Demand) specifically.
    """
    for attempt in range(MAX_RETRIES):
        delay = -1.0
        
        async with GEMINI_SEMAPHORE:
            try:
                response = await client.aio.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=config
                )
                return response
            except APIError as e:
                error_str = str(e)
                # Differentiate 429 vs 503
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < MAX_RETRIES - 1:
                        delay = _parse_retry_delay(error_str)
                        if delay <= 0:
                            delay = BASE_BACKOFF * (2 ** attempt)
                        print(f"Gemini API rate limit (429). Retrying in {delay:.2f}s... (Attempt {attempt+1}/{MAX_RETRIES})")
                    else:
                        print("Max retries exceeded for Gemini API rate limit (429).")
                        raise e
                elif "503" in error_str or "UNAVAILABLE" in error_str:
                    if attempt < MAX_RETRIES - 1:
                        delay = BASE_BACKOFF * (2 ** attempt)
                        print(f"Gemini API high demand (503). Retrying in {delay:.2f}s... (Attempt {attempt+1}/{MAX_RETRIES})")
                    else:
                        print("Max retries exceeded for Gemini API high demand (503).")
                        raise e
                else:
                    raise e
        
        # Sleep outside the semaphore so we don't block other tasks while waiting
        if delay > 0:
            await asyncio.sleep(delay)
            
    raise Exception("Max retries exceeded.")
