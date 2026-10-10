import pytest
import asyncio
from fastapi.testclient import TestClient
from google.genai.errors import APIError
from unittest.mock import AsyncMock
import backend.app as api
from rag.generator import AnswerGenerationError
from schemas.models import Claim, ClaimValidation, TranscriptChunk

client = TestClient(api.app)


@pytest.fixture
def pipeline(monkeypatch):
    mocks = {
        "retrieve_transcript": AsyncMock(return_value=[TranscriptChunk(id="chunk_1", text="Neurons have weights and biases.")]),
        "search_web": AsyncMock(return_value=[]),
        "extract_claims": AsyncMock(return_value=[Claim(id="claim_1", text="Neurons have weights and biases.")]),
        "validate_claims": AsyncMock(return_value=[ClaimValidation(claim_id="claim_1", claim="Neurons have weights and biases.", status="UNVERIFIED", explanation="No external evidence.")]),
        "generate_answer": AsyncMock(return_value="A neural network connects layers of neurons."),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(api, name, mock)
    return mocks


def ask(validate=True):
    return client.post("/ask", json={"query": "What is a neural network?", "video_id": "aircAruvnKk", "validate_externally": validate})


def test_ranked_sources_serialize_with_available_passage_and_display_limit(pipeline, monkeypatch):
    from schemas.models import SearchResult, ExternalEvidence
    monkeypatch.setattr(api, "TOP_K_WEB_DISPLAY", 2)
    pipeline["search_web"].return_value = [
        SearchResult(title="Unrelated", url="https://other.test"),
        SearchResult(title="Parameters", snippet="Original snippet", url="https://source.test"),
        SearchResult(title="Neural", url="https://partial.test"),
        SearchResult(url="https://empty.test"),
    ]
    evidence = ExternalEvidence(source_title="Parameters", url="https://source.test", domain="source.test", passage="Neural network", relevance_score=0.1)
    pipeline["validate_claims"].return_value[0].supporting_evidence = [evidence]
    response = ask()
    assert response.status_code == 200
    data = response.json()
    assert data["source_display_limit"] == 2
    assert [r["relevance_score"] for r in data["sources"]] == [100, 50, 0, 0]
    assert data["sources"][0]["passage"] == "Neural network"
    assert data["sources"][0]["snippet"] == "Original snippet"
    assert "passage" in data["sources"][0]["relevance_reason"]
    assert data["claims"][0]["supporting_evidence"][0]["relevance_score"] == 0.1
    assert len(pipeline["validate_claims"].call_args.args[1]) == 4


@pytest.mark.parametrize("validate", [False, True])
def test_successful_pipeline_keeps_logical_call_count(pipeline, validate):
    response = ask(validate)
    assert response.status_code == 200
    assert response.json()["answer"] == "A neural network connects layers of neurons."
    assert response.json()["video_evidence"]
    pipeline["generate_answer"].assert_awaited_once()
    assert pipeline["extract_claims"].await_count == int(validate)
    assert pipeline["validate_claims"].await_count == int(validate)


@pytest.mark.parametrize("validate", [False, True])
def test_missing_transcript_stops_before_gemini(pipeline, validate):
    pipeline["retrieve_transcript"].return_value = []
    response = ask(validate)
    assert response.status_code == 422
    assert "transcript" in response.json()["detail"]
    for name in ("extract_claims", "validate_claims", "generate_answer"):
        pipeline[name].assert_not_awaited()


@pytest.mark.parametrize("code, expected", [(429, 429), (503, 503), (504, 504), (401, 502), (403, 502), (404, 502), (500, 502)])
def test_gemini_errors_are_not_successful_answers(pipeline, code, expected):
    pipeline["generate_answer"].side_effect = APIError(code, {"error": {"message": "secret-provider-payload"}})
    response = ask()
    assert response.status_code == expected
    assert response.json()["error"]
    assert "answer" not in response.json()
    assert "secret-provider-payload" not in response.text
    pipeline["generate_answer"].assert_awaited_once()


def test_provider_deadline_is_not_reported_as_configuration_error(pipeline):
    pipeline["generate_answer"].side_effect = APIError(504, {"error": {"message": "secret-provider-payload"}})
    response = ask()
    assert response.status_code == 504
    assert "deadline" in response.json()["error"]
    assert "configuration" not in response.text
    assert "secret-provider-payload" not in response.text


def test_daily_quota_at_extraction_stops_pipeline(pipeline):
    pipeline["extract_claims"].side_effect = APIError(429, {"error": {"message": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}})
    response = ask()
    assert response.status_code == 429
    assert "quota" in response.json()["error"]
    pipeline["validate_claims"].assert_not_awaited()
    pipeline["generate_answer"].assert_not_awaited()


@pytest.mark.parametrize("status", [502, 504])
def test_generation_failure_returns_actionable_error(pipeline, status):
    pipeline["generate_answer"].side_effect = AnswerGenerationError("Gemini connection failed.", status)
    response = ask(False)
    assert response.status_code == status
    assert response.json() == {"error": "Gemini connection failed."}


def test_unverified_validation_still_generates_answer_once(pipeline):
    response = ask()
    assert response.status_code == 200
    assert response.json()["validation_status"] == "UNVERIFIED"
    pipeline["validate_claims"].assert_awaited_once()
    pipeline["generate_answer"].assert_awaited_once()

@pytest.mark.parametrize("validate", [False, True])
def test_request_deadline_cancels_retrieval_and_returns_504(pipeline, monkeypatch, validate):
    cancelled = []
    async def hang(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)
    pipeline["retrieve_transcript"].side_effect = hang
    monkeypatch.setattr(api, "REQUEST_TIMEOUT_SECONDS", 0.02)
    response = ask(validate)
    assert response.status_code == 504
    assert "time limit" in response.json()["detail"]
    assert cancelled == [True]
    pipeline["generate_answer"].assert_not_awaited()

def test_daily_quota_in_real_validator_stops_final_generation(pipeline, monkeypatch):
    import validation.validator as validator
    from schemas.models import ExternalEvidence, SearchResult
    monkeypatch.setattr(api, "validate_claims", validator.validate_claims)
    pipeline["search_web"].return_value = [SearchResult(title="Source", url="https://source.test", snippet="Neurons have weights.")]
    monkeypatch.setattr(validator, "extract_evidence", AsyncMock(return_value=ExternalEvidence(source_title="Source", url="https://source.test", domain="source.test", passage="Neurons have weights.", relevance_score=1)))
    call = AsyncMock(side_effect=APIError(429, {"error": {"message": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}}))
    monkeypatch.setattr(validator, "call_gemini_with_retry", call)
    response = ask()
    assert response.status_code == 429
    call.assert_awaited_once()
    pipeline["generate_answer"].assert_not_awaited()

def test_request_deadline_cancels_final_generation(pipeline, monkeypatch):
    cancelled = []
    async def hang(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)
    pipeline["generate_answer"].side_effect = hang
    monkeypatch.setattr(api, "REQUEST_TIMEOUT_SECONDS", 0.02)
    response = ask()
    assert response.status_code == 504
    assert cancelled == [True]
    for name in ("extract_claims", "validate_claims", "generate_answer"):
        pipeline[name].assert_awaited_once()
