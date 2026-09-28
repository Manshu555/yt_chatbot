from typing import List
from schemas.models import SearchResult

def rerank_results(results: List[SearchResult]) -> List[SearchResult]:
    """Reranks search results based on heuristics."""
    return results
