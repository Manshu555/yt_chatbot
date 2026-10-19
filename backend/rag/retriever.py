import asyncio
import logging
from threading import Lock

from chromadb import Client
from chromadb.errors import NotFoundError
from config import TOP_K_TRANSCRIPT, TRANSCRIPT_MIN_SIMILARITY
from schemas.models import TranscriptChunk
from rag.transcript import get_transcript
from rag.chunker import chunk_transcript
from rag.embeddings import MODEL_ID, embed_texts
from rag.ranking import bm25, fuse, near_duplicate, is_summary

logger = logging.getLogger(__name__)
chroma_client = Client()
_collection_locks = {}
_raw_documents = {}
_guard = Lock()
CONFIG = {"embedding_model": MODEL_ID, "hnsw:space": "cosine", "chunk_version": 2}


def get_or_create_collection(video_id):
    with _guard:
        lock = _collection_locks.setdefault(video_id, Lock())
    with lock:
        return _load_collection(video_id)


def _load_collection(video_id):
    # A fresh namespace leaves legacy collections intact. Fail closed if the
    # new namespace has incompatible metadata rather than mix vector spaces.
    name = f"yt_semantic_v2_{video_id}"
    try:
        collection = chroma_client.get_collection(name, embedding_function=None)
    except (NotFoundError, KeyError):
        collection = None
    if collection is not None:
        if any((collection.metadata or {}).get(key) != value for key, value in CONFIG.items()):
            raise ValueError("Incompatible transcript collection; use a fresh versioned namespace")
        if collection.count():
            return collection

    if video_id not in _raw_documents:
        text = get_transcript(video_id)
        if not text:
            return None
        documents = chunk_transcript(text)
        if not documents:
            return None
        _raw_documents[video_id] = documents
    documents = _raw_documents[video_id]
    try:
        vectors = embed_texts(documents)
    except Exception as error:
        logger.warning("Transcript embedding fallback=BM25 error=%s", type(error).__name__)
        return None
    if collection is None:
        collection = chroma_client.create_collection(name, metadata=CONFIG, embedding_function=None)
    collection.add(
        documents=documents, embeddings=vectors,
        ids=[f"chunk_{i}" for i in range(len(documents))],
        metadatas=[{"video_id": video_id, "chunk_index": i,
                    "source_url": f"https://www.youtube.com/watch?v={video_id}",
                    **getattr(doc, "metadata", {})} for i, doc in enumerate(documents)],
    )
    logger.info("Transcript indexed method=MiniLM/cosine chunks=%d", len(documents))
    return collection


async def retrieve_transcript(query, video_id, n_results=TOP_K_TRANSCRIPT):
    return await asyncio.to_thread(_retrieve_sync, query, video_id, n_results)


def _retrieve_sync(query, video_id, n_results):
    if n_results <= 0:
        return []
    collection = get_or_create_collection(video_id)
    stored = collection.get(include=["documents", "metadatas"]) if collection is not None else None
    documents = stored["documents"] if stored else _raw_documents.get(video_id, [])
    if not documents:
        return []
    ids = stored["ids"] if stored else [f"chunk_{i}" for i in range(len(documents))]
    metadata = stored["metadatas"] if stored else [{} for _ in documents]
    lexical = bm25(query, documents)
    dense = [0.0] * len(documents)
    semantic = False
    if collection is not None and not is_summary(query):
        try:
            result = collection.query(query_embeddings=embed_texts([query]),
                                      n_results=min(len(documents), max(n_results * 4, 20)),
                                      include=["distances"])
            positions = {id_: i for i, id_ in enumerate(ids)}
            for id_, distance in zip(result["ids"][0], result["distances"][0]):
                dense[positions[id_]] = max(-1.0, min(1.0, 1 - distance))
            semantic = True
        except Exception as error:
            logger.warning("Transcript query fallback=BM25 error=%s", type(error).__name__)
    scores = fuse(dense, lexical) if semantic else lexical
    if is_summary(query):
        order = sorted(range(len(documents)), key=lambda i: (metadata[i] or {}).get("chunk_index", i))
        count = min(n_results, len(order))
        order = [order[round(i * (len(order) - 1) / max(count - 1, 1))] for i in range(count)]
        method = "summary-coverage"
    else:
        order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
        order = [i for i in order if (dense[i] >= TRANSCRIPT_MIN_SIMILARITY or lexical[i] > 0)]
        method = "dense+BM25/RRF" if semantic else "BM25-fallback"
    selected = []
    for i in order:
        if any(near_duplicate(documents[i], chunk.text) for chunk in selected):
            continue
        selected.append(TranscriptChunk(id=ids[i], text=documents[i],
                                        score=None if is_summary(query) else scores[i]))
        if len(selected) >= n_results:
            break
    logger.info("Transcript retrieval method=%s candidates=%d returned=%d", method, len(documents), len(selected))
    return selected
