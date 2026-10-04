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
        m.setattr(gemini_client, "GEMINI_CALL_TIMEOUT_SECONDS", 120)
        m.setattr(asyncio, "sleep", AsyncMock())
        response = await call_gemini_with_retry(mock_client, "model", "prompt", None)
        
    assert response.text == "SUCCESS AFTER RETRY"
    assert mock_client.aio.models.generate_content.call_count == 2
    assert GeminiMetrics.api_calls == 2
    assert GeminiMetrics.retries_429 == 1

@pytest.mark.asyncio
async def test_503_exhaustion_has_exact_attempts_and_backoff(monkeypatch):
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(side_effect=APIError(503, {"error": {"message": "UNAVAILABLE"}}))
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    with pytest.raises(APIError):
        await call_gemini_with_retry(client, "model", "prompt", None)
    assert [call.args[0] for call in sleep.call_args_list] == [4, 8, 16]
    assert GeminiMetrics.api_calls == 4
    assert GeminiMetrics.retries_503 == 3
    assert client.aio.models.generate_content.await_count == 4

@pytest.mark.asyncio
async def test_http_options_keep_one_sdk_attempt_and_original_schema():
    from google.genai import types
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value="response")
    original = types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json", http_options=types.HttpOptions(headers={"x-test": "original"}))
    await call_gemini_with_retry(client, "model", "prompt", original)
    options = client.aio.models.generate_content.call_args.kwargs["config"]
    assert options.http_options.retry_options.attempts == 1
    assert options.http_options.timeout == gemini_client.LLM_TIMEOUT_SECONDS * 1000
    assert options.http_options.headers == {"x-test": "original"}
    assert options.response_mime_type == "application/json"
    assert original.http_options.timeout is None

@pytest.mark.asyncio
async def test_hung_attempt_is_cancelled_and_does_not_retry(monkeypatch):
    client = MagicMock()
    cancelled = asyncio.Event()
    async def hang(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    client.aio.models.generate_content = AsyncMock(side_effect=hang)
    monkeypatch.setattr(gemini_client, "LLM_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(TimeoutError):
        await call_gemini_with_retry(client, "model", "prompt", None)
    assert cancelled.is_set()
    assert GeminiMetrics.api_calls == 1
    assert GeminiMetrics.retries_503 == 0

@pytest.mark.asyncio
async def test_semaphore_wait_obeys_whole_call_deadline(monkeypatch):
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock()
    monkeypatch.setattr(gemini_client, "GEMINI_CALL_TIMEOUT_SECONDS", 0.01)
    semaphore = gemini_client._get_gemini_semaphore()
    await semaphore.acquire()
    await semaphore.acquire()
    try:
        with pytest.raises(TimeoutError):
            await call_gemini_with_retry(client, "model", "prompt", None)
    finally:
        semaphore.release()
        semaphore.release()
    client.aio.models.generate_content.assert_not_awaited()
    assert GeminiMetrics.api_calls == 0

@pytest.mark.asyncio
async def test_cancelled_backoff_does_not_count_an_unmade_retry(monkeypatch):
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(side_effect=APIError(503, {"error": {"message": "UNAVAILABLE"}}))
    async def blocked_sleep(delay):
        await asyncio.Event().wait()
    monkeypatch.setattr(gemini_client, "GEMINI_CALL_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(gemini_client, "BASE_BACKOFF", 0.001)
    monkeypatch.setattr(asyncio, "sleep", blocked_sleep)
    with pytest.raises(TimeoutError):
        await call_gemini_with_retry(client, "model", "prompt", None)
    assert GeminiMetrics.api_calls == 1
    assert GeminiMetrics.retries_503 == 0

@pytest.mark.asyncio
async def test_retry_hint_longer_than_budget_does_not_sleep(monkeypatch):
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(side_effect=APIError(429, {"error": {"message": "Please retry in 2m10s."}}))
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    with pytest.raises(APIError):
        await call_gemini_with_retry(client, "model", "prompt", None)
    sleep.assert_not_awaited()
    assert GeminiMetrics.api_calls == 1
    assert GeminiMetrics.retries_429 == 0

@pytest.mark.asyncio
async def test_nonretryable_status_is_not_misclassified_by_message(monkeypatch):
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(side_effect=APIError(403, {"error": {"message": "Permission failed for resource 429503."}}))
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    with pytest.raises(APIError):
        await call_gemini_with_retry(client, "model", "prompt", None)
    sleep.assert_not_awaited()
    assert GeminiMetrics.api_calls == 1

@pytest.mark.parametrize(("message", "expected"), [("Please retry in 59.8s.", 59.8), ("Please retry in 1h2m3.5s.", 3723.5), ("No hint", -1)])
def test_retry_hint_units(message, expected):
    assert gemini_client._parse_retry_delay(message) == expected

@pytest.mark.asyncio
@pytest.mark.parametrize("daily_quota", [False, True])
async def test_sdk_http_attempt_count_matches_metrics_without_network(monkeypatch, daily_quota):
    import httpx
    from google import genai
    from google.genai import types
    requests = []
    def provider(request):
        requests.append(request)
        code = 429 if daily_quota else 503
        message = "GenerateRequestsPerDayPerProjectPerModel-FreeTier" if daily_quota else "UNAVAILABLE"
        return httpx.Response(code, json={"error": {"code": code, "message": message}})
    transport = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    client = genai.Client(api_key="unit-test-key", http_options=types.HttpOptions(httpx_async_client=transport, retry_options=types.HttpRetryOptions(attempts=5)))
    monkeypatch.setattr(gemini_client, "BASE_BACKOFF", 0.001)
    try:
        with pytest.raises(APIError):
            await call_gemini_with_retry(client, "test-model", "test-prompt", None)
    finally:
        await client.aio.aclose()
        client.close()
        await transport.aclose()
    assert len(requests) == (1 if daily_quota else 4)
    assert all(request.headers["X-Server-Timeout"] == str(gemini_client.LLM_TIMEOUT_SECONDS) for request in requests)
    assert GeminiMetrics.api_calls == len(requests)
    assert GeminiMetrics.retries_503 == (0 if daily_quota else 3)
    assert GeminiMetrics.fail_fast_daily == int(daily_quota)

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
