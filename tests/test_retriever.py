import asyncio
import threading
from types import SimpleNamespace
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
    collection = SimpleNamespace(query=lambda **kwargs: {"documents": [["Neural networks learn."]]})
    collections = {}
    def get(name):
        if name not in collections:
            raise KeyError(name)
        return collections[name]
    def create(name):
        assert name not in collections
        collection.add = Mock()
        collections[name] = collection
        return collection
    chroma = SimpleNamespace(get_collection=get, create_collection=Mock(side_effect=create))
    download = Mock(return_value="Neural networks learn.")
    monkeypatch.setattr(retriever, "chroma_client", chroma)
    monkeypatch.setattr(retriever, "get_transcript", download)
    monkeypatch.setattr(retriever, "_collection_locks", {})
    results = await asyncio.gather(retriever.retrieve_transcript("query", "aircAruvnKk"), retriever.retrieve_transcript("query", "aircAruvnKk"))
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
