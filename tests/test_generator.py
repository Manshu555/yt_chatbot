from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai.errors import APIError
import rag.generator as generator
from schemas.models import TranscriptChunk


@pytest.fixture
def gemini_call(monkeypatch):
    monkeypatch.setattr(generator.genai, "Client", MagicMock())
    call = AsyncMock(return_value=SimpleNamespace(text="A network connects neurons."))
    monkeypatch.setattr(generator, "call_gemini_with_retry", call)
    return call


async def generate():
    return await generator.generate_answer("What is a neural network?", [TranscriptChunk(id="chunk_1", text="A network connects neurons.")])


@pytest.mark.asyncio
async def test_answer_success(gemini_call):
    assert await generate() == "A network connects neurons."
    gemini_call.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [None, SimpleNamespace(text=None), SimpleNamespace(text=""), SimpleNamespace(text="   ")])
async def test_empty_answer_raises_error(gemini_call, response):
    gemini_call.return_value = response
    with pytest.raises(generator.AnswerGenerationError, match="empty answer"):
        await generate()


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [429, 503, 404])
async def test_provider_error_propagates(gemini_call, code):
    error = APIError(code, {"error": {"message": "Provider unavailable"}})
    gemini_call.side_effect = error
    with pytest.raises(APIError) as caught:
        await generate()
    assert caught.value is error
    gemini_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_timeout_is_not_an_answer(gemini_call):
    gemini_call.side_effect = TimeoutError()
    with pytest.raises(generator.AnswerGenerationError) as caught:
        await generate()
    assert caught.value.status_code == 504


@pytest.mark.asyncio
async def test_connection_error_does_not_expose_provider_details(gemini_call):
    gemini_call.side_effect = OSError("secret-api-key-in-url")
    with pytest.raises(generator.AnswerGenerationError) as caught:
        await generate()
    assert caught.value.status_code == 502
    assert "secret-api-key" not in str(caught.value)
