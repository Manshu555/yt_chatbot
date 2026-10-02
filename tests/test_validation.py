import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from validation.validator import (
    BatchValidationResponse,
    BatchValidationResult,
    _determine_status,
    validate_claims,
)
from schemas.models import Claim, SearchResult, ExternalEvidence

def test_determine_status():
    assert _determine_status(1, 0) == "SUPPORTED"
    assert _determine_status(1, 1) == "PARTIALLY_SUPPORTED"
    assert _determine_status(0, 1) == "CONTRADICTED"
    assert _determine_status(0, 0) == "UNVERIFIED"

def test_claim_prompt_format_regression():
    from backend.prompts.claims import CLAIM_EXTRACTION_PROMPT
    try:
        formatted = CLAIM_EXTRACTION_PROMPT.format(query="test", context="test")
        assert "claim_1" in formatted
    except KeyError as e:
        pytest.fail(f"KeyError during formatting. Likely unescaped curly braces in JSON example: {e}")

@pytest.fixture
def mock_evidence_extractor():
    async def _extract(*args, **kwargs):
        return ExternalEvidence(
            source_title="Mock Source",
            url="http://mock.url",
            domain="mock.url",
            passage="Mock passage for testing deduplication.",
            relevance_score=0.9
        )
    return _extract

@pytest.mark.asyncio
async def test_batch_validation_success(mock_evidence_extractor):
    claims = [Claim(id="claim_1", text="Test claim 1"), Claim(id="claim_2", text="Test claim 2")]
    web_results = [SearchResult(title="T", url="U", snippet="S")]
    
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "results": [
            {"claim_id": "claim_1", "status": "SUPPORTED", "evidence_ids": ["ev_1"], "reason": "Reason 1"},
            {"claim_id": "claim_2", "status": "CONTRADICTED", "evidence_ids": ["ev_1"], "reason": "Reason 2"}
        ]
    })
    
    with patch("validation.validator.extract_evidence", side_effect=mock_evidence_extractor):
        with patch("validation.validator.call_gemini_with_retry", AsyncMock(return_value=mock_response)):
            validations = await validate_claims(claims, web_results)
            
            assert len(validations) == 2
            
            assert validations[0].claim_id == "claim_1"
            assert validations[0].status == "SUPPORTED"
            assert len(validations[0].supporting_evidence) == 1
            assert len(validations[0].contradicting_evidence) == 0
            
            assert validations[1].claim_id == "claim_2"
            assert validations[1].status == "CONTRADICTED"
            assert len(validations[1].supporting_evidence) == 0
            assert len(validations[1].contradicting_evidence) == 1

@pytest.mark.asyncio
async def test_batch_validation_multiple_claims_in_one_call(mock_evidence_extractor):
    claims = [
        Claim(id="claim_1", text="Test claim 1"),
        Claim(id="claim_2", text="Test claim 2"),
        Claim(id="claim_3", text="Test claim 3"),
    ]
    web_results = [SearchResult(title="T", url="U", snippet="S")]
    mock_response = MagicMock()
    mock_response.parsed = BatchValidationResponse(
        results=[
            BatchValidationResult(claim_id="claim_1", status="SUPPORTED", evidence_ids=["ev_1"], reason="R1"),
            BatchValidationResult(claim_id="claim_2", status="PARTIALLY_SUPPORTED", evidence_ids=["ev_1"], reason="R2"),
            BatchValidationResult(claim_id="claim_3", status="UNVERIFIED", evidence_ids=[], reason="R3"),
        ]
    )

    call = AsyncMock(return_value=mock_response)
    with patch("validation.validator.extract_evidence", side_effect=mock_evidence_extractor):
        with patch("validation.validator.call_gemini_with_retry", call):
            validations = await validate_claims(claims, web_results)

    assert call.call_count == 1
    assert [v.status for v in validations] == ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNVERIFIED"]


@pytest.mark.asyncio
async def test_batch_validation_missing_result_is_unverified(mock_evidence_extractor):
    claims = [Claim(id="claim_1", text="Test claim 1"), Claim(id="claim_2", text="Test claim 2")]
    web_results = [SearchResult(title="T", url="U", snippet="S")]
    mock_response = MagicMock()
    mock_response.parsed = BatchValidationResponse(
        results=[
            BatchValidationResult(claim_id="claim_1", status="SUPPORTED", evidence_ids=["ev_1"], reason="R1"),
        ]
    )

    with patch("validation.validator.extract_evidence", side_effect=mock_evidence_extractor):
        with patch("validation.validator.call_gemini_with_retry", AsyncMock(return_value=mock_response)):
            validations = await validate_claims(claims, web_results)

    assert validations[0].status == "SUPPORTED"
    assert validations[1].status == "UNVERIFIED"


@pytest.mark.asyncio
async def test_batch_validation_deduplicates_evidence():
    claims = [Claim(id="claim_1", text="Test claim")]
    web_results = [
        SearchResult(title="T1", url="U1", snippet="S1"),
        SearchResult(title="T2", url="U2", snippet="S2"),
    ]
    duplicate_evidence = ExternalEvidence(
        source_title="Mock Source",
        url="http://mock.url",
        domain="mock.url",
        passage="Same passage.",
        relevance_score=0.9,
    )
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "results": [
            {"claim_id": "claim_1", "status": "SUPPORTED", "evidence_ids": ["ev_1", "ev_2"], "reason": "Reason"}
        ]
    })

    with patch("validation.validator.extract_evidence", AsyncMock(return_value=duplicate_evidence)):
        with patch("validation.validator.call_gemini_with_retry", AsyncMock(return_value=mock_response)) as call:
            validations = await validate_claims(claims, web_results)

    prompt = call.call_args.kwargs["prompt"]
    assert "[ev_1]" in prompt
    assert "[ev_2]" not in prompt
    assert len(validations[0].supporting_evidence) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_supporting", "expected_contradicting"),
    [
        ("SUPPORTED", 1, 0),
        ("PARTIALLY_SUPPORTED", 1, 0),
        ("CONTRADICTED", 0, 1),
        ("UNVERIFIED", 0, 0),
    ],
)
async def test_batch_validation_status_mapping(mock_evidence_extractor, status, expected_supporting, expected_contradicting):
    claims = [Claim(id="claim_1", text="Test claim")]
    web_results = [SearchResult(title="T", url="U", snippet="S")]
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "results": [
            {"claim_id": "claim_1", "status": status, "evidence_ids": ["ev_1"], "reason": "Reason"}
        ]
    })

    with patch("validation.validator.extract_evidence", side_effect=mock_evidence_extractor):
        with patch("validation.validator.call_gemini_with_retry", AsyncMock(return_value=mock_response)):
            validations = await validate_claims(claims, web_results)

    assert validations[0].status == status
    assert len(validations[0].supporting_evidence) == expected_supporting
    assert len(validations[0].contradicting_evidence) == expected_contradicting

@pytest.mark.asyncio
async def test_batch_validation_malformed(mock_evidence_extractor):
    claims = [Claim(id="claim_1", text="Test claim")]
    web_results = [SearchResult(title="T", url="U", snippet="S")]
    
    mock_response = MagicMock()
    # Malformed JSON
    mock_response.text = "NOT JSON"
    
    with patch("validation.validator.extract_evidence", side_effect=mock_evidence_extractor):
        with patch("validation.validator.call_gemini_with_retry", AsyncMock(return_value=mock_response)):
            validations = await validate_claims(claims, web_results)
            
            assert len(validations) == 1
            assert validations[0].status == "UNVERIFIED"

@pytest.mark.asyncio
async def test_batch_validation_api_failure(mock_evidence_extractor):
    claims = [Claim(id="claim_1", text="Test claim")]
    web_results = [SearchResult(title="T", url="U", snippet="S")]
    
    with patch("validation.validator.extract_evidence", side_effect=mock_evidence_extractor):
        with patch("validation.validator.call_gemini_with_retry", AsyncMock(side_effect=Exception("API Error"))):
            validations = await validate_claims(claims, web_results)
            
            assert len(validations) == 1
            assert validations[0].status == "UNVERIFIED"


@pytest.mark.asyncio
async def test_extraction_failure_claim_does_not_become_supported(mock_evidence_extractor):
    claims = [Claim(id="claim_extraction_failed", text="Original user query")]
    web_results = [SearchResult(title="T", url="U", snippet="S")]
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "results": [
            {
                "claim_id": "claim_extraction_failed",
                "status": "UNVERIFIED",
                "evidence_ids": [],
                "reason": "Extraction failed, no factual claim to validate.",
            }
        ]
    })

    with patch("validation.validator.extract_evidence", side_effect=mock_evidence_extractor):
        with patch("validation.validator.call_gemini_with_retry", AsyncMock(return_value=mock_response)):
            validations = await validate_claims(claims, web_results)

    assert validations[0].status == "UNVERIFIED"
