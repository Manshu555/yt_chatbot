import asyncio
import threading
from unittest.mock import Mock

import pytest
import rag.retriever as retriever


@pytest.mark.asyncio
async def test_blocking_retrieval_runs_off_the_event_loop(monkeypatch):
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    def blocking(*args):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(2)
        return []
    monkeypatch.setattr(retriever, "_retrieve_sync", blocking)
    task = asyncio.create_task(retriever.retrieve_transcript("query", "aircAruvnKk"))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        assert not task.done()
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_concurrent_same_video_has_one_download_and_collection(monkeypatch):
    from chromadb import EphemeralClient
    import uuid
    chroma = EphemeralClient()
    create = Mock(wraps=chroma.create_collection)
    monkeypatch.setattr(chroma, "create_collection", create)
    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[1.0] + [0.0] * 383 for _ in texts])
    download = Mock(return_value="Neural networks learn.")
    monkeypatch.setattr(retriever, "chroma_client", chroma)
    monkeypatch.setattr(retriever, "get_transcript", download)
    monkeypatch.setattr(retriever, "_collection_locks", {})
    monkeypatch.setattr(retriever, "_raw_documents", {})
    video = uuid.uuid4().hex[:11]
    results = await asyncio.gather(retriever.retrieve_transcript("query", video), retriever.retrieve_transcript("query", video))
    download.assert_called_once()
    chroma.create_collection.assert_called_once()
    assert all(result[0].text == "Neural networks learn." for result in results)


def test_empty_chunks_never_create_an_empty_cached_collection(monkeypatch):
    chroma = Mock()
    chroma.get_collection.side_effect = KeyError()
    monkeypatch.setattr(retriever, "chroma_client", chroma)
    monkeypatch.setattr(retriever, "get_transcript", Mock(return_value="..."))
    monkeypatch.setattr(retriever, "_collection_locks", {})
    assert retriever.get_or_create_collection("aircAruvnKk") is None
    chroma.create_collection.assert_not_called()
