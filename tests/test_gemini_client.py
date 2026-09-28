import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from google.genai.errors import APIError

from backend.rag.gemini_client import call_gemini_with_retry, MAX_RETRIES

@pytest.mark.asyncio
async def test_call_gemini_success():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "SUCCESS"
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    
    response = await call_gemini_with_retry(mock_client, "model", "prompt", None)
    assert response.text == "SUCCESS"
    assert mock_client.aio.models.generate_content.call_count == 1

@pytest.mark.asyncio
async def test_call_gemini_retry_on_429():
    mock_client = MagicMock()
    
    # Create an APIError that looks like a 429
    error_429 = APIError(429, {"error": {"message": "RESOURCE_EXHAUSTED"}})
    
    mock_response = MagicMock()
    mock_response.text = "SUCCESS AFTER RETRY"
    
    # Fail first time, succeed second time
    mock_client.aio.models.generate_content = AsyncMock(side_effect=[error_429, mock_response])
    
    # We don't want tests to actually sleep for 4 seconds
    with pytest.MonkeyPatch.context() as m:
        m.setattr(asyncio, "sleep", AsyncMock())
        response = await call_gemini_with_retry(mock_client, "model", "prompt", None)
        
    assert response.text == "SUCCESS AFTER RETRY"
    assert mock_client.aio.models.generate_content.call_count == 2
    
@pytest.mark.asyncio
async def test_call_gemini_retry_on_503():
    mock_client = MagicMock()
    
    error_503 = APIError(503, {"error": {"message": "UNAVAILABLE"}})
    mock_response = MagicMock()
    mock_response.text = "SUCCESS AFTER 503"
    
    # Fail twice, succeed third time
    mock_client.aio.models.generate_content = AsyncMock(side_effect=[error_503, error_503, mock_response])
    
    with pytest.MonkeyPatch.context() as m:
        m.setattr(asyncio, "sleep", AsyncMock())
        response = await call_gemini_with_retry(mock_client, "model", "prompt", None)
        
    assert response.text == "SUCCESS AFTER 503"
    assert mock_client.aio.models.generate_content.call_count == 3

@pytest.mark.asyncio
async def test_call_gemini_retry_exhaustion():
    mock_client = MagicMock()
    error_429 = APIError(429, {"error": {"message": "RESOURCE_EXHAUSTED"}})
    
    # Always fail
    mock_client.aio.models.generate_content = AsyncMock(side_effect=error_429)
    
    with pytest.MonkeyPatch.context() as m:
        m.setattr(asyncio, "sleep", AsyncMock())
        with pytest.raises(APIError):
            await call_gemini_with_retry(mock_client, "model", "prompt", None)
            
    assert mock_client.aio.models.generate_content.call_count == MAX_RETRIES

@pytest.mark.asyncio
async def test_call_gemini_semaphore_concurrency():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "SUCCESS"
    
    # Let's track active requests to ensure semaphore limits concurrency
    active_requests = 0
    max_active = 0
    
    async def mock_generate_content(*args, **kwargs):
        nonlocal active_requests, max_active
        active_requests += 1
        max_active = max(max_active, active_requests)
        await asyncio.sleep(0.01)  # Simulate some work
        active_requests -= 1
        return mock_response
        
    mock_client.aio.models.generate_content = AsyncMock(side_effect=mock_generate_content)
    
    # Fire 5 concurrent requests
    tasks = [call_gemini_with_retry(mock_client, "model", "prompt", None) for _ in range(5)]
    await asyncio.gather(*tasks)
    
    # Our semaphore is configured to 2
    assert max_active <= 2
    assert mock_client.aio.models.generate_content.call_count == 5
