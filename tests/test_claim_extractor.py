import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from google.genai.errors import APIError

from schemas.models import TranscriptChunk
from validation.claim_extractor import ClaimExtractionResponse, ExtractedClaim, extract_claims


@pytest.fixture
def video_evidence():
    return [
        TranscriptChunk(id="chunk_1", text="Adam optimizer was introduced in 2014."),
        TranscriptChunk(id="chunk_2", text="It combines momentum and adaptive learning rates."),
    ]


@pytest.mark.asyncio
async def test_extract_claims_valid_structured_parsed_response(video_evidence):
    response = SimpleNamespace(
        parsed=ClaimExtractionResponse(
            claims=[
                ExtractedClaim(id="claim_1", text="Adam optimizer was introduced in 2014."),
                ExtractedClaim(id="claim_2", text="Adam combines momentum and adaptive learning rates."),
            ]
        ),
        text=None,
    )

    with patch("validation.claim_extractor.call_gemini_with_retry", AsyncMock(return_value=response)):
        claims = await extract_claims("Summarize Adam.", video_evidence)

    assert [claim.id for claim in claims] == ["claim_1", "claim_2"]
    assert len(claims) == 2


@pytest.mark.asyncio
async def test_extract_claims_valid_json_text_response(video_evidence):
    response = SimpleNamespace(
        parsed=None,
        text=json.dumps(
            {
                "claims": [
                    {"id": "claim_1", "text": "Adam optimizer was introduced in 2014."},
                    {"id": "claim_2", "text": "Adam combines momentum and adaptive learning rates."},
                ]
            }
        ),
    )

    with patch("validation.claim_extractor.call_gemini_with_retry", AsyncMock(return_value=response)):
        claims = await extract_claims("Summarize Adam.", video_evidence)

    assert len(claims) == 2
    assert claims[0].text == "Adam optimizer was introduced in 2014."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(parsed=None, text="NOT JSON"),
        SimpleNamespace(parsed=None, text='{"claims": [{"id": "claim_1", "text": "unterminated}'),
        SimpleNamespace(parsed=None, text=json.dumps({"items": []})),
        SimpleNamespace(parsed=None, text=""),
        None,
    ],
)
async def test_extract_claims_malformed_or_empty_response_falls_back(response, video_evidence):
    with patch("validation.claim_extractor.call_gemini_with_retry", AsyncMock(return_value=response)):
        claims = await extract_claims("What did the video say?", video_evidence)

    assert len(claims) == 1
    assert claims[0].id == "claim_extraction_failed"
    assert claims[0].text == "What did the video say?"


@pytest.mark.asyncio
async def test_extract_claims_deduplicates_claim_text(video_evidence):
    response = SimpleNamespace(
        parsed=ClaimExtractionResponse(
            claims=[
                ExtractedClaim(id="claim_1", text="Adam optimizer was introduced in 2014."),
                ExtractedClaim(id="claim_2", text="Adam optimizer was introduced in 2014."),
            ]
        ),
        text=None,
    )

    with patch("validation.claim_extractor.call_gemini_with_retry", AsyncMock(return_value=response)):
        claims = await extract_claims("Summarize Adam.", video_evidence)

    assert len(claims) == 1


@pytest.mark.asyncio
async def test_extract_claims_api_error_falls_back(video_evidence):
    error = APIError(500, {"error": {"message": "temporary Gemini error"}})

    with patch("validation.claim_extractor.call_gemini_with_retry", AsyncMock(side_effect=error)):
        claims = await extract_claims("What did the video say?", video_evidence)

    assert claims[0].id == "claim_extraction_failed"


@pytest.mark.asyncio
async def test_extract_claims_daily_quota_error_propagates(video_evidence):
    error = APIError(
        429,
        {
            "error": {
                "message": "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
            }
        },
    )

    with patch("validation.claim_extractor.call_gemini_with_retry", AsyncMock(side_effect=error)):
        with pytest.raises(APIError):
            await extract_claims("What did the video say?", video_evidence)
