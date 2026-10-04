from chromadb import Client
import asyncio
from threading import Lock
from schemas.models import TranscriptChunk
from rag.transcript import get_transcript
from rag.chunker import chunk_transcript

chroma_client = Client()
_collection_locks = {}

def get_or_create_collection(video_id: str):
    with _collection_locks.setdefault(video_id, Lock()):
        return _load_collection(video_id)

def _load_collection(video_id: str):
    collection_name = f"yt_transcript_{video_id}"

    try:
        collection = chroma_client.get_collection(collection_name)
        print(f"Using existing collection: {collection_name}")
        return collection
    except Exception:
        pass

    transcript_text = get_transcript(video_id)
    if not transcript_text:
        return None

    documents = chunk_transcript(transcript_text)
    ids = [f"chunk_{i}" for i in range(len(documents))]

    if documents:
        collection = chroma_client.create_collection(collection_name)
        print(f"Created new collection: {collection_name}")
        print(f"Adding {len(documents)} chunks to ChromaDB for video {video_id}")
        collection.add(documents=documents, ids=ids)
    else:
        print(f"No valid documents to add to ChromaDB for video {video_id}")
        return None

    return collection

async def retrieve_transcript(query: str, video_id: str, n_results: int = 5) -> list[TranscriptChunk]:
    return await asyncio.to_thread(_retrieve_sync, query, video_id, n_results)

def _retrieve_sync(query: str, video_id: str, n_results: int) -> list[TranscriptChunk]:
    collection = get_or_create_collection(video_id)
    if not collection:
        return []

    results = collection.query(query_texts=[query], n_results=n_results)
    if not results['documents'] or len(results['documents'][0]) == 0:
        return []
        
    retrieved_chunks = results['documents'][0]
    
    chunks = []
    for i, text in enumerate(retrieved_chunks):
        chunks.append(TranscriptChunk(id=f"retrieved_{i}", text=text))
        
    return chunks
