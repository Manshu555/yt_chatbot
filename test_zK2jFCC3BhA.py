import asyncio
import sys
import os

# Ensure backend is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from test_integration_real import run_pipeline

async def run_test():
    video_id = "zK2jFCC3BhA"
    query = "Summarize this video in simple, concise bullet points. Include the main topic, key points, and important conclusions. Use only information supported by the video transcript/evidence."
    
    print("==================================================")
    print("TEST: YT_CHATBOT E2E Pipeline (Summary)")
    print(f"Video ID: {video_id}")
    print(f"Query: {query}")
    print("==================================================")
    
    try:
        res = await run_pipeline(query, video_id, validate_externally=False)
        print("Result Status:", res['status'])
        print("Sources:", res.get('sources', []))
        print("Metrics:", res['metrics'])
        print("Claims Extracted:", len(res.get('claims', [])))
        for c in res.get('claims', []):
            print(f" - {c.text if hasattr(c, 'text') else str(c)}")
        print("Answer:", res.get('answer', ''))
    except Exception as e:
        print("Failed with Exception:", e)

if __name__ == "__main__":
    asyncio.run(run_test())
