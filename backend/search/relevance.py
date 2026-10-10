"""Deterministic topical coverage; deliberately independent of claim verdicts."""

from schemas.models import ExternalEvidence, SearchResult
from search.evidence_extractor import _select_passage, _terms


def score_result(question: str, result: SearchResult) -> SearchResult:
    """Score unique question-term coverage across title, snippet and passage."""
    terms = _terms(question or "")
    fields = [("title", result.title), ("snippet", result.snippet), ("passage", result.passage)]
    matches = terms & _terms(" ".join(text for _, text in fields))
    score = round(100 * len(matches) / len(terms), 2) if terms else 0.0
    if not terms:
        reason = "No searchable question terms to compare."
    elif not matches:
        reason = "No question terms matched the available title, snippet or passage."
    else:
        locations = ", ".join(label for label, text in fields if terms & _terms(text))
        examples = ", ".join(sorted(matches)[:5])
        reason = f"Matched {len(matches)} of {len(terms)} question terms in {locations}: {examples}."
    return result.model_copy(update={"relevance_score": score, "relevance_reason": reason})


def rank_results(question: str, results: list[SearchResult], evidence: list[ExternalEvidence] | None = None) -> list[SearchResult]:
    """Use available extracted passages without extra page fetches or LLM calls.

    Stable sorting retains provider order for ties, including zero scores.
    Claim-specific evidence scores are never combined with result relevance.
    """
    passages_by_url: dict[str, list[str]] = {}
    for item in evidence or []:
        if item.passage:
            passages_by_url.setdefault(item.url, []).append(item.passage)
    scored = []
    for result in results:
        passages = passages_by_url.get(result.url, [])
        if passages:
            passage, _ = _select_passage(question, "\n".join([result.passage, *passages]), "")
            result = result.model_copy(update={"passage": passage})
        scored.append(score_result(question, result))
    return sorted(scored, key=lambda result: result.relevance_score, reverse=True)
