import pytest
from backend.validation.validator import _determine_status

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

@pytest.mark.asyncio
async def test_classify_evidence_support():
    from backend.validation.validator import _classify_evidence
    from unittest.mock import AsyncMock, MagicMock, patch
    from backend.schemas.models import ExternalEvidence
    
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "SUPPORT"
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    
    ev = ExternalEvidence(source_title="T", url="U", domain="D", passage="P", relevance_score=0.9)
    
    with patch("backend.validation.validator.genai.Client", return_value=mock_client):
        res = await _classify_evidence("Claim", ev)
        assert res == "SUPPORT"

@pytest.mark.asyncio
async def test_classify_evidence_contradict():
    from backend.validation.validator import _classify_evidence
    from unittest.mock import AsyncMock, MagicMock, patch
    from backend.schemas.models import ExternalEvidence
    
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "CONTRADICT"
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    
    ev = ExternalEvidence(source_title="T", url="U", domain="D", passage="P", relevance_score=0.9)
    
    with patch("backend.validation.validator.genai.Client", return_value=mock_client):
        res = await _classify_evidence("Claim", ev)
        assert res == "CONTRADICT"

@pytest.mark.asyncio
async def test_classify_evidence_unverified():
    from backend.validation.validator import _classify_evidence
    from unittest.mock import AsyncMock, MagicMock, patch
    from backend.schemas.models import ExternalEvidence
    
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "UNVERIFIED"
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    
    ev = ExternalEvidence(source_title="T", url="U", domain="D", passage="P", relevance_score=0.9)
    
    with patch("backend.validation.validator.genai.Client", return_value=mock_client):
        res = await _classify_evidence("Claim", ev)
        assert res == "UNVERIFIED"
