import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from google.genai.errors import APIError

import rag.gemini_client as gemini_client
from rag.gemini_client import call_gemini_with_retry, MAX_RETRIES, GeminiMetrics

@pytest.fixture(autouse=True)
def reset_metrics():
    GeminiMetrics.api_calls = 0
    GeminiMetrics.retries_429 = 0
    GeminiMetrics.retries_503 = 0
    GeminiMetrics.fail_fast_daily = 0
    gemini_client._GEMINI_SEMAPHORE_BY_LOOP.clear()

@pytest.mark.asyncio
async def test_call_gemini_success():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "SUCCESS"
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    
    response = await call_gemini_with_retry(mock_client, "model", "prompt", None)
    assert response.text == "SUCCESS"
    assert mock_client.aio.models.generate_content.call_count == 1
    assert GeminiMetrics.api_calls == 1
    assert GeminiMetrics.retries_429 == 0
    assert GeminiMetrics.retries_503 == 0

@pytest.mark.asyncio
async def test_call_gemini_retry_on_429():
    mock_client = MagicMock()
    
    # Create an APIError that looks like a 429
    error_429 = APIError(429, {"error": {"message": "RESOURCE_EXHAUSTED. Please retry in 59.8s."}})
    
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
    assert GeminiMetrics.api_calls == 2
    assert GeminiMetrics.retries_429 == 1

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
    assert GeminiMetrics.api_calls == 3
    assert GeminiMetrics.retries_503 == 2

@pytest.mark.asyncio
async def test_call_gemini_fail_fast_daily_quota():
    mock_client = MagicMock()
    
    # This string contains GenerateRequestsPerDayPerProjectPerModel-FreeTier
    error_str = "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests... quotaId: 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'"
    error_429_daily = APIError(429, {"error": {"message": error_str}})
    
    mock_client.aio.models.generate_content = AsyncMock(side_effect=error_429_daily)
    
    with pytest.MonkeyPatch.context() as m:
        m.setattr(asyncio, "sleep", AsyncMock())
        with pytest.raises(APIError) as exc_info:
            await call_gemini_with_retry(mock_client, "model", "prompt", None)
            
    assert "GenerateRequestsPerDay" in str(exc_info.value)
    # Should only call once, no retries
    assert mock_client.aio.models.generate_content.call_count == 1
    assert GeminiMetrics.api_calls == 1
    assert GeminiMetrics.fail_fast_daily == 1
    assert GeminiMetrics.retries_429 == 0

@pytest.mark.asyncio
async def test_call_gemini_fail_fast_daily_quota_requests_per_day_variant():
    mock_client = MagicMock()
    error_429_daily = APIError(429, {"error": {"message": "Quota exceeded: RequestsPerDay limit hit"}})
    mock_client.aio.models.generate_content = AsyncMock(side_effect=error_429_daily)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(asyncio, "sleep", AsyncMock())
        with pytest.raises(APIError):
            await call_gemini_with_retry(mock_client, "model", "prompt", None)

    assert mock_client.aio.models.generate_content.call_count == 1
    assert GeminiMetrics.api_calls == 1
    assert GeminiMetrics.fail_fast_daily == 1
    assert GeminiMetrics.retries_429 == 0


def test_gemini_metrics_single_import_instance():
    import importlib

    imported = importlib.import_module("rag.gemini_client")
    assert imported.GeminiMetrics is GeminiMetrics

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
    assert GeminiMetrics.api_calls == MAX_RETRIES
    assert GeminiMetrics.retries_429 == MAX_RETRIES - 1

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
