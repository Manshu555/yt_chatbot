from unittest.mock import AsyncMock
import asyncio

import pytest
from schemas.models import SearchResult
import search.evidence_extractor as extractor
import search.source_fetcher as fetcher


@pytest.mark.asyncio
async def test_evidence_is_selected_from_relevant_body_not_page_prefix(monkeypatch):
    body = "Menu subscriptions advertising. " * 40 + "\nWeights and biases are learned from examples in a neural network."
    monkeypatch.setattr(extractor, "fetch_page_text", AsyncMock(return_value=body))
    source = SearchResult(title="Source", url="https://source.test/article", domain="source.test", snippet="Menu")
    evidence = await extractor.extract_evidence("A neural network learns weights and biases from examples.", source)
    assert "Weights and biases are learned" in evidence.passage
    assert evidence.relevance_score > 0
    assert evidence.url == source.url


@pytest.mark.asyncio
async def test_unavailable_page_uses_actual_search_snippet(monkeypatch):
    monkeypatch.setattr(extractor, "fetch_page_text", AsyncMock(return_value=""))
    source = SearchResult(title="Source", url="https://source.test/article", snippet="Weights tune the strength of connections.")
    evidence = await extractor.extract_evidence("Weights tune connections.", source)
    assert evidence.passage == source.snippet

@pytest.mark.asyncio
async def test_same_source_is_fetched_once_for_multiple_claims(monkeypatch):
    fetch = AsyncMock(return_value="Weights control connections. Biases set activation thresholds.")
    monkeypatch.setattr(extractor, "fetch_page_text", fetch)
    source = SearchResult(title="Source", url="https://source.test", snippet="Network parameters")
    cache = {}
    results = await asyncio.gather(extractor.extract_evidence("Weights control connections.", source, cache), extractor.extract_evidence("Biases set thresholds.", source, cache))
    fetch.assert_awaited_once()
    assert len(results) == 2


@pytest.mark.asyncio
async def test_source_html_removes_boilerplate_and_decodes_entities(monkeypatch):
    class Response:
        status = 200
        headers = {"Content-Type": "text/html; charset=utf-8"}
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def text(self):
            return '<html><nav>Navigation noise</nav><SCRIPT>secretScript()</SCRIPT><main><p>Weights &amp; biases learn.</p></main><footer>Footer noise</footer></html>'
    class Session:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        def get(self, *args, **kwargs):
            return Response()
    monkeypatch.setattr(fetcher.aiohttp, "ClientSession", lambda **kwargs: Session())
    result = await fetcher.fetch_page_text(SearchResult(title="Source", url="https://example.com/article", snippet="Snippet"))
    assert result == "Weights & biases learn."
    assert "mock page" not in result
