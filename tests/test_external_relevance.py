"""Regression cases for question-focused external discovery and passage ranking."""
import math
from unittest.mock import AsyncMock

import pytest

import search.reranker as web
from schemas.models import SearchResult
from search.query_planner import discovery_queries, ranking_query


def test_specific_question_is_not_diluted_by_other_success_rules():
    question = "How does the environment shape success?"
    claims = ["Wake up early every day.", "Your environment and peers influence your success.",
              "Invest money wisely and build discipline."]
    assert ranking_query(question, claims) == question.rstrip("?")
    queries = discovery_queries(question, claims)
    assert len(queries) == 2
    assert all("Wake" not in query and "money" not in query for query in queries)
    assert all("-site:facebook.com" in query for query in queries)
    assert "peers" in queries[1]


def test_vague_verification_is_grounded_and_social_requests_are_allowed():
    claim = "Neurons have weights and biases."
    assert ranking_query("Verify this with external evidence", [claim]) == claim
    assert claim in discovery_queries("Verify this with external evidence", [claim])[0]
    assert "-site:facebook.com" not in discovery_queries("What do Facebook posts say about success?")[0]


@pytest.mark.asyncio
async def test_fetch_long_snippets_and_rank_question_without_unrelated_claims(monkeypatch):
    page = "Your environment shapes success through supportive peers and habits."
    fetch = AsyncMock(return_value=page)
    monkeypatch.setattr(web, "fetch_page_text", fetch)
    batches = []

    def embed(texts):
        batches.append(texts)
        return [[1.0, 0.0] if "environment" in text.lower() else [0.0, 1.0] for text in texts]

    monkeypatch.setattr(web, "embed_texts", embed)
    question = "How does the environment shape success?"
    result = SearchResult(title="Environment", url="https://article.test",
                          snippet="Employee task performance and workplace commitment " * 8)
    ranked = await web.rerank_results(question, [result], context="Wake early. Work hard. Invest money.")
    fetch.assert_awaited_once_with(result)
    assert batches[0][0] == question.rstrip("?")
    assert len(ranked) == 1
    assert ranked[0].passage == page


@pytest.mark.asyncio
async def test_environment_success_filters_social_and_weak_adjacent_topic(monkeypatch):
    monkeypatch.setattr(web, "fetch_page_text", AsyncMock(return_value=""))

    def embed(texts):
        def vector(text):
            similarity = 0.82 if text.startswith("Supportive peers") else 0.51
            return [similarity, math.sqrt(1 - similarity ** 2)]
        return [[1.0, 0.0], *[vector(text) for text in texts[1:]]]

    monkeypatch.setattr(web, "embed_texts", embed)
    results = [
        SearchResult(title="Success post", snippet="Your environment shapes success.", url="https://m.facebook.com/posts/123"),
        SearchResult(title="Environment and success", snippet="Supportive peers and surroundings build habits that help people succeed.", url="https://article.test"),
        SearchResult(title="Workplace task performance", snippet="Employee task performance correlates with workplace commitment.", url="https://study.test"),
    ]
    ranked = await web.rerank_results("How does environment shape success?", results, top_k=3)
    assert [result.url for result in ranked] == ["https://article.test"]
    assert "MiniLM cosine=0.820" in ranked[0].relevance_reason
    # The score still uses the existing absolute-cosine formula, not top-result normalization.
    assert 70 < ranked[0].relevance_score < 90
    assert web.fetch_page_text.await_count == 2


@pytest.mark.asyncio
async def test_social_sources_remain_available_when_requested(monkeypatch):
    monkeypatch.setattr(web, "fetch_page_text", AsyncMock(return_value=""))
    monkeypatch.setattr(web, "embed_texts", lambda texts: [[1.0, 0.0] for _ in texts])
    result = SearchResult(title="Facebook discussion", snippet="Environment and success discussion.", url="https://facebook.com/posts/123")
    assert await web.rerank_results("Find Facebook posts about environment and success", [result])


def test_misleading_title_cannot_supply_semantic_relevance(monkeypatch):
    monkeypatch.setattr(web, "embed_texts", lambda texts: [
        [1.0, 0.0] if "environment" in text.lower() else [0.0, 1.0] for text in texts])
    assert web._rank("environment success", [SearchResult(
        title="Environment success", snippet="Buy watches and shoes in our shop.", url="https://shop.test")], 3) == []


def test_weak_pool_is_not_normalized_to_high_scores(monkeypatch):
    monkeypatch.setattr(web, "embed_texts", lambda texts: [[1.0, 0.0], *[
        [0.4, math.sqrt(0.84)] for _ in texts[1:]]])
    assert web._rank("environment success", [SearchResult(
        title="Adjacent topic", snippet="Workplace task performance.", url="https://study.test")], 3) == []


def test_relaxed_ranking_preserves_low_scores_and_labels_them(monkeypatch):
    monkeypatch.setattr(web, "embed_texts", lambda texts: [[1.0, 0.0], *[
        [0.3, math.sqrt(0.91)] for _ in texts[1:]]])
    candidate = SearchResult(title="Adjacent topic", snippet="Workplace task performance.", url="https://study.test")
    ranked = web._rank("environment success", [candidate], 3, allow_weak=True)
    assert len(ranked) == 1
    assert ranked[0].relevance_score < 50
    assert "low topical match" in ranked[0].relevance_reason
    assert "cosine=0.300" in ranked[0].relevance_reason


@pytest.mark.asyncio
async def test_last_resort_real_social_link_without_page_content(monkeypatch):
    monkeypatch.setattr(web, "fetch_page_text", AsyncMock(return_value=""))
    monkeypatch.setattr(web, "embed_texts", lambda texts: [[1.0, 0.0] for _ in texts])
    candidate = SearchResult(title="Environment and success", url="https://facebook.com/posts/123")
    assert await web.rerank_results("environment success", [candidate]) == []
    ranked = await web.rerank_results("environment success", [candidate], allow_weak=True)
    assert len(ranked) == 1
    assert ranked[0].url == candidate.url
    assert ranked[0].passage == ""
    assert "Only the search-result title" in ranked[0].relevance_reason


@pytest.mark.asyncio
async def test_relaxed_ranking_prefers_articles_over_social_posts(monkeypatch):
    monkeypatch.setattr(web, "fetch_page_text", AsyncMock(return_value=""))
    monkeypatch.setattr(web, "embed_texts", lambda texts: [[1.0, 0.0] for _ in texts])
    social = SearchResult(title="Post", snippet="Environment success.", url="https://facebook.com/posts/123")
    article = SearchResult(title="Article", snippet="Environment success.", url="https://article.test")
    ranked = await web.rerank_results("environment success", [social, article], allow_weak=True)
    assert [result.url for result in ranked] == [article.url]


def test_relaxed_ranking_retains_lexical_matches_when_embeddings_fail(monkeypatch):
    monkeypatch.setattr(web, "embed_texts", lambda texts: (_ for _ in ()).throw(RuntimeError()))
    candidate = SearchResult(title="Learning", snippet="Learning through practice.", url="https://article.test")
    ranked = web._rank("environment success learning", [candidate], 3, allow_weak=True)
    assert len(ranked) == 1
    assert "BM25 fallback" in ranked[0].relevance_reason
    assert "low topical match" in ranked[0].relevance_reason


@pytest.mark.asyncio
async def test_fallback_reuses_page_fetches_even_when_fetch_failed(monkeypatch):
    fetch = AsyncMock(return_value="")
    monkeypatch.setattr(web, "fetch_page_text", fetch)
    monkeypatch.setattr(web, "embed_texts", lambda texts: [[1.0, 0.0], *[
        [0.3, math.sqrt(0.91)] for _ in texts[1:]]])
    candidate = SearchResult(title="Article", snippet="Workplace task performance.", url="https://article.test")
    cache = {}
    assert await web.rerank_results("environment success", [candidate], page_cache=cache) == []
    assert await web.rerank_results("environment success", [candidate], page_cache=cache, allow_weak=True)
    fetch.assert_awaited_once()


@pytest.mark.asyncio
async def test_relaxed_ranking_never_uses_invalid_urls(monkeypatch):
    fetch = AsyncMock(return_value="")
    monkeypatch.setattr(web, "fetch_page_text", fetch)
    assert await web.rerank_results("success", [SearchResult(title="Invalid", url="javascript:bad")], allow_weak=True) == []
    fetch.assert_not_awaited()
