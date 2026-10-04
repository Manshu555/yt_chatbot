from schemas.models import SearchResult, ExternalEvidence
from search.source_fetcher import fetch_page_text
import re
import asyncio

_STOPWORDS = set("a an the is are was were be been being it its to of and or in on for by with from that this as at can has have how what which does do".split())

def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _STOPWORDS

def _select_passage(claim: str, page_text: str, snippet: str) -> tuple[str, float]:
    terms = _terms(claim)
    candidates = []
    for paragraph in page_text.splitlines():
        paragraph = " ".join(paragraph.split())
        for start in range(0, len(paragraph), 250):
            candidates.append(paragraph[start:start + 500])
    if snippet:
        candidates.append(snippet[:500])
    if not candidates:
        return "", 0.0
    passage = max(candidates, key=lambda candidate: len(terms & _terms(candidate)))
    score = len(terms & _terms(passage)) / max(len(terms), 1)
    if not score and snippet:
        passage = snippet[:500]
    return passage, score

async def extract_evidence(claim_text: str, result: SearchResult, page_cache: dict | None = None) -> ExternalEvidence:
    """Extracts relevant evidence from a search result for a given claim."""
    if page_cache is None:
        page_text = await fetch_page_text(result)
    else:
        if result.url not in page_cache:
            page_cache[result.url] = asyncio.create_task(fetch_page_text(result))
        page_text = await page_cache[result.url]
    
    passage, score = _select_passage(claim_text, page_text, result.snippet)
    
    return ExternalEvidence(
        source_title=result.title,
        url=result.url,
        domain=result.domain or "unknown",
        passage=passage,
        relevance_score=score
    )
