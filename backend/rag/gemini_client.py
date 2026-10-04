import asyncio
import re
from google import genai
from google.genai.errors import APIError
from config import LLM_TIMEOUT_SECONDS, GEMINI_CALL_TIMEOUT_SECONDS

class GeminiMetrics:
    api_calls = 0
    retries_429 = 0
    retries_503 = 0
    fail_fast_daily = 0

# Global semaphore to limit concurrent requests across the entire process.
# It is created lazily per running event loop so tests and ASGI workers do not
# reuse an asyncio primitive bound to a closed or different loop.
_GEMINI_SEMAPHORE_BY_LOOP = {}
GEMINI_SEMAPHORE = None
MAX_RETRIES = 4
BASE_BACKOFF = 4.0  # seconds

def _get_gemini_semaphore() -> asyncio.Semaphore:
    global GEMINI_SEMAPHORE
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    semaphore = _GEMINI_SEMAPHORE_BY_LOOP.get(loop_id)
    if semaphore is None:
        semaphore = asyncio.Semaphore(2)
        _GEMINI_SEMAPHORE_BY_LOOP[loop_id] = semaphore
    GEMINI_SEMAPHORE = semaphore
    return semaphore

def _parse_retry_delay(error_message: str) -> float:
    """Parse the seconds, minutes, or hours in a provider retry hint."""
    match = re.search(r"retry in ((?:\d+(?:\.\d+)?[hms])+)", str(error_message), re.IGNORECASE)
    if match:
        units = {"h": 3600, "m": 60, "s": 1}
        return sum(float(value) * units[unit.lower()] for value, unit in re.findall(r"(\d+(?:\.\d+)?)([hms])", match.group(1), re.IGNORECASE))
    return -1.0

def is_daily_quota_error(error: Exception) -> bool:
    normalized = str(error).lower().replace(" ", "").replace("_", "").replace("-", "")
    return getattr(error, "code", None) == 429 and ("perday" in normalized or "daily" in normalized)

async def call_gemini_with_retry(
    client: genai.Client, 
    model_id: str, 
    prompt: str, 
    config: genai.types.GenerateContentConfig
):
    # Keep SDK retries disabled so each measured attempt is one HTTP attempt.
    config = config or genai.types.GenerateContentConfig()
    options = config.http_options or genai.types.HttpOptions()
    config = config.model_copy(update={"http_options": options.model_copy(update={
        "timeout": int(LLM_TIMEOUT_SECONDS * 1000),
        "retry_options": genai.types.HttpRetryOptions(attempts=1),
    })})
    async with asyncio.timeout(GEMINI_CALL_TIMEOUT_SECONDS):
        return await _call_with_backoff(client, model_id, prompt, config)

async def _call_with_backoff(client, model_id, prompt, config):
    retry_code = None
    deadline = asyncio.get_running_loop().time() + GEMINI_CALL_TIMEOUT_SECONDS
    for attempt in range(MAX_RETRIES):
        delay = -1.0
        
        async with _get_gemini_semaphore():
            try:
                GeminiMetrics.api_calls += 1
                if retry_code == 429:
                    GeminiMetrics.retries_429 += 1
                elif retry_code == 503:
                    GeminiMetrics.retries_503 += 1
                async with asyncio.timeout(LLM_TIMEOUT_SECONDS):
                    response = await client.aio.models.generate_content(
                        model=model_id,
                        contents=prompt,
                        config=config
                    )
                return response
            except APIError as e:
                error_str = str(e)
                # Differentiate 429 vs 503
                if e.code == 429:
                    # Fail fast on daily quota exhaustion
                    if is_daily_quota_error(e):
                        GeminiMetrics.fail_fast_daily += 1
                        print("Gemini API daily quota exhausted. Failing fast.")
                        raise e
                        
                    if attempt < MAX_RETRIES - 1:
                        retry_code = 429
                        delay = _parse_retry_delay(error_str)
                        if delay <= 0:
                            delay = BASE_BACKOFF * (2 ** attempt)
                        print(f"Gemini API rate limit (429). Retrying in {delay:.2f}s... (Attempt {attempt+1}/{MAX_RETRIES})")
                    else:
                        print("Max retries exceeded for Gemini API rate limit (429).")
                        raise e
                elif e.code == 503:
                    if attempt < MAX_RETRIES - 1:
                        retry_code = 503
                        delay = BASE_BACKOFF * (2 ** attempt)
                        print(f"Gemini API high demand (503). Retrying in {delay:.2f}s... (Attempt {attempt+1}/{MAX_RETRIES})")
                    else:
                        print("Max retries exceeded for Gemini API high demand (503).")
                        raise e
                else:
                    raise e
                if delay >= deadline - asyncio.get_running_loop().time():
                    raise
        
        # Sleep outside the semaphore so we don't block other tasks while waiting
        if delay > 0:
            await asyncio.sleep(delay)
            
    raise Exception("Max retries exceeded.")
