import pytest
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


@pytest.mark.parametrize("code, expected", [(429, 429), (503, 503), (401, 502), (403, 502), (404, 502), (500, 502)])
def test_gemini_errors_are_not_successful_answers(pipeline, code, expected):
    pipeline["generate_answer"].side_effect = APIError(code, {"error": {"message": "secret-provider-payload"}})
    response = ask()
    assert response.status_code == expected
    assert response.json()["error"]
    assert "answer" not in response.json()
    assert "secret-provider-payload" not in response.text
    pipeline["generate_answer"].assert_awaited_once()


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
