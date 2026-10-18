"""Offline deterministic vectors test ranking; real MiniLM smoke is opt-in."""
import os
import uuid
from unittest.mock import AsyncMock

import pytest
from chromadb import EphemeralClient
import rag.retriever as transcript
import search.reranker as web
from rag.ranking import bm25, fuse
from schemas.models import SearchResult
from routing.query_analyzer import requires_external_validation


def vector(text):
    text = text.lower()
    if any(term in text for term in ("automobile", "car", "vehicle")):
        return [1.0, 0.0] + [0.0] * 382
    return [0.0, 1.0] + [0.0] * 382


@pytest.fixture
def corpus(monkeypatch):
    monkeypatch.setattr(transcript, "chroma_client", EphemeralClient())
    monkeypatch.setattr(transcript, "_raw_documents", {})
    monkeypatch.setattr(transcript, "_collection_locks", {})
    monkeypatch.setattr(transcript, "embed_texts", lambda texts: [vector(text) for text in texts])
    monkeypatch.setattr(transcript, "get_transcript", lambda video: "The automobile uses battery propulsion. Gardening grows roses.")
    monkeypatch.setattr(transcript, "chunk_transcript", lambda text: text.split(". "))
    return uuid.uuid4().hex[:11]


def test_paraphrase_uses_actual_chroma_query_vectors(corpus):
    query = "How is the car powered?"
    assert bm25(query, ["The automobile uses battery propulsion."])[0] == 0
    results = transcript._retrieve_sync(query, corpus, 2)
    assert len(results) == 1
    assert "automobile" in results[0].text
    collection = transcript.get_or_create_collection(corpus)
    assert collection.metadata["hnsw:space"] == "cosine"
    assert collection.get(include=["metadatas"])["metadatas"][0]["video_id"] == corpus


def test_exact_identifier_and_query_embedding_failure(corpus, monkeypatch, caplog):
    transcript.get_or_create_collection(corpus)
    def fail(texts):
        raise RuntimeError("offline")
    monkeypatch.setattr(transcript, "embed_texts", fail)
    assert "Gardening" in transcript._retrieve_sync("roses", corpus, 1)[0].text
    assert "fallback=BM25" in caplog.text


def test_document_embedding_failure_returns_lexical(corpus, monkeypatch, caplog):
    monkeypatch.setattr(transcript, "embed_texts", lambda texts: (_ for _ in ()).throw(RuntimeError()))
    assert transcript._retrieve_sync("automobile", corpus, 1)
    assert "fallback=BM25" in caplog.text


def test_summary_covers_beginning_middle_end(corpus, monkeypatch):
    documents = ["Battery propulsion uses electrical energy", "Neurons adapt connection strengths",
                 "Gardens depend on adequate water", "Astronomy studies distant galaxies",
                 "Economic inflation affects household budgets", "Ocean currents distribute heat",
                 "Language evolves through cultural exchange", "Algorithms organize complex information",
                 "Exercise improves cardiovascular endurance"]
    monkeypatch.setattr(transcript, "chunk_transcript", lambda text: documents)
    result = transcript._retrieve_sync("Summarize this video", corpus, 3)
    assert [chunk.id for chunk in result] == ["chunk_0", "chunk_4", "chunk_8"]


def test_incompatible_collection_is_not_reused(corpus):
    transcript.chroma_client.create_collection(f"yt_semantic_v2_{corpus}", metadata={"embedding_model": "different"})
    with pytest.raises(ValueError, match="Incompatible"):
        transcript.get_or_create_collection(corpus)


@pytest.fixture
def external(monkeypatch):
    monkeypatch.setattr(web, "embed_texts", lambda texts: [vector(text) for text in texts])
    monkeypatch.setattr(web, "fetch_page_text", AsyncMock(return_value=""))


@pytest.mark.asyncio
async def test_tools_duplicates_and_semantic_paraphrase(external):
    candidates = [
        SearchResult(title="Car powered YouTube Summarizer", snippet="Summarize this video with our AI tool", url="https://tool.test"),
        SearchResult(title="Battery propulsion", snippet="The automobile uses battery propulsion.", url="https://source.test/p?utm_source=x#part"),
        SearchResult(title="Duplicate", snippet="The automobile uses battery propulsion.", url="https://www.source.test/p"),
        SearchResult(title="Gardening", snippet="Roses bloom in spring.", url="https://garden.test"),
    ]
    from search.relevance import score_result
    # The old term-coverage ranker favors the unrelated product over the paraphrase.
    assert score_result("How is the car powered?", candidates[0]).relevance_score > score_result("How is the car powered?", candidates[1]).relevance_score
    results = await web.rerank_results("How is the car powered?", candidates)
    assert len(results) == 1
    assert results[0].url == candidates[1].url
    assert "MiniLM cosine=1.000" in results[0].relevance_reason
    assert "provider rank=2" in results[0].relevance_reason
    assert await web.rerank_results("car", []) == []
    assert await web.rerank_results("car", candidates[-1:]) == []


@pytest.mark.asyncio
async def test_external_missing_embeddings_explicit_fallback(external, monkeypatch, caplog):
    monkeypatch.setattr(web, "embed_texts", lambda texts: (_ for _ in ()).throw(RuntimeError()))
    results = await web.rerank_results("battery propulsion", [SearchResult(title="Battery", snippet="Battery propulsion drives vehicles.", url="https://source.test")])
    assert results and "BM25 fallback" in results[0].relevance_reason
    assert "fallback=BM25" in caplog.text


@pytest.mark.asyncio
async def test_content_can_beat_title_and_near_duplicates(external):
    results = await web.rerank_results("car", [
        SearchResult(title="Reference", snippet="The automobile uses battery propulsion.", url="https://a.test"),
        SearchResult(title="Reference copy", snippet="The automobile uses battery propulsion.", url="https://b.test"),
        SearchResult(title="car", snippet="", url="https://empty.test"),
    ])
    assert len(results) == 1
    assert results[0].title == "Reference"


def test_routing_and_fusion():
    assert requires_external_validation("Summarize this video", True)
    assert requires_external_validation("Summarize and verify this video", True)
    assert requires_external_validation("Give additional context", False)
    assert requires_external_validation("What is the latest evidence?", False)
    scores = fuse([0.9, 0.3], [3, 5])
    assert scores[0] == pytest.approx(0.75 + 0.25 * 61 / 62)
    assert scores[0] > scores[1]


@pytest.mark.skipif(os.getenv("RUN_LOCAL_EMBEDDING_TESTS") != "1", reason="Opt-in MiniLM weights required")
def test_real_minilm_paraphrase(corpus, monkeypatch):
    from rag.embeddings import embed_texts, cosine
    vectors = embed_texts(["How is the car powered?", "The automobile runs on electricity stored in batteries.", "Roses require fertile soil and regular watering."])
    assert cosine(vectors[0], vectors[1]) > cosine(vectors[0], vectors[2]) + 0.15
    monkeypatch.setattr(transcript, "embed_texts", embed_texts)
    assert "automobile" in transcript._retrieve_sync("How is the car powered?", corpus, 1)[0].text
    monkeypatch.setattr(web, "embed_texts", embed_texts)
    ranked = web._rank("How is the car powered?", [
        SearchResult(title="Electric propulsion", snippet="The automobile runs on electricity stored in batteries.", url="https://motor.test"),
        SearchResult(title="Gardening", snippet="Roses require fertile soil and regular watering.", url="https://garden.test"),
    ], 3)
    assert [result.url for result in ranked] == ["https://motor.test"]


def test_timestamp_metadata_survives_chunking(corpus, monkeypatch):
    from rag.chunker import chunk_transcript
    from rag.transcript import TranscriptText
    text = TranscriptText("An automobile runs on batteries.", [
        {"text": "An automobile runs on batteries.", "start": 12.0, "end": 18.0},
    ])
    monkeypatch.setattr(transcript, "get_transcript", lambda video: text)
    monkeypatch.setattr(transcript, "chunk_transcript", chunk_transcript)
    collection = transcript.get_or_create_collection(corpus)
    meta = collection.get(include=["metadatas"])["metadatas"][0]
    assert meta["start_seconds"] == 12.0
    assert meta["end_seconds"] == 18.0
    assert meta["chunk_index"] == 0
    assert meta["source_url"].endswith(corpus)


def test_embedding_cache_batches_and_reuses(monkeypatch):
    import rag.embeddings as embeddings
    from collections import OrderedDict
    from unittest.mock import Mock
    encoder = Mock(side_effect=lambda texts: [vector(text) for text in texts])
    monkeypatch.setattr(embeddings, "_encoder", encoder)
    monkeypatch.setattr(embeddings, "_vectors", OrderedDict())
    embeddings.embed_texts(["car", "roses", "car"])
    embeddings.embed_texts(["roses", "car"])
    encoder.assert_called_once_with(["car", "roses"])
    monkeypatch.setattr(embeddings, "_encoder", Mock(return_value=[]))
    with pytest.raises(ValueError, match="Incomplete"):
        embeddings.embed_texts(["new text"])
