import asyncio
import os
import sys

# add current dir to path to import backend
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from backend.rag.retriever import retrieve_transcript

async def test_real_transcript():
    print("=== VERIFY REAL TRANSCRIPT RETRIEVAL ===")
    video_id = "ukzFI9rgwfU" 
    print(f"Testing real transcript for video {video_id}")
    chunks = await retrieve_transcript("machine learning", video_id)
    if not chunks:
        print("Failed to retrieve chunks or no transcript available.")
    else:
        print(f"Successfully retrieved {len(chunks)} chunks.")
        for chunk in chunks:
            print(f"- {chunk.text[:50]}...")

if __name__ == "__main__":
    asyncio.run(test_real_transcript())
