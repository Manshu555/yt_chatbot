from schemas.models import SearchResult, ExternalEvidence
from search.source_fetcher import fetch_page_text

async def extract_evidence(claim_text: str, result: SearchResult) -> ExternalEvidence:
    """Extracts relevant evidence from a search result for a given claim."""
    page_text = await fetch_page_text(result)
    
    passage = page_text[:500] if page_text else result.snippet
    
    return ExternalEvidence(
        source_title=result.title,
        url=result.url,
        domain=result.domain or "unknown",
        passage=passage,
        relevance_score=0.8
    )
