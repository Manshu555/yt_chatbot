import asyncio
import time
import sys
import os

# Ensure backend is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from backend.routing.query_analyzer import requires_external_validation
from backend.rag.retriever import retrieve_transcript
from backend.search.web_search import search_web
from backend.validation.claim_extractor import extract_claims
from backend.validation.validator import validate_claims
from backend.rag.generator import generate_answer
from backend.app import aggregate_status, collect_evidence

async def run_pipeline(query, video_id, validate_externally=False, mock_llm_responses=False):
    metrics = {}
    
    t0 = time.time()
    validation_required = requires_external_validation(query, validate_externally)
    t1 = time.time()
    metrics['query_analysis_ms'] = (t1 - t0) * 1000

    if not validation_required:
        t0 = time.time()
        video_evidence = await retrieve_transcript(query, video_id)
        t1 = time.time()
        metrics['youtube_retrieval_ms'] = (t1 - t0) * 1000
        
        t0 = time.time()
        answer = await generate_answer(query, video_evidence, None, [])
        t1 = time.time()
        metrics['final_llm_ms'] = (t1 - t0) * 1000
        
        metrics['total_ms'] = sum(metrics.values())
        return {"status": "SUCCESS (Normal RAG)", "metrics": metrics, "answer": answer}

    # Parallel retrieval
    t_start_parallel = time.time()
    
    video_task = retrieve_transcript(query, video_id)
    web_task = search_web(query, top_k=3)
    
    video_evidence, web_results = await asyncio.gather(video_task, web_task)
    
    t_end_parallel = time.time()
    
    # We can't perfectly separate the parallel times using just gather, but we can assume they overlap.
    metrics['parallel_retrieval_ms'] = (t_end_parallel - t_start_parallel) * 1000
    metrics['web_search_ms'] = metrics['parallel_retrieval_ms'] # Upper bound
    metrics['youtube_retrieval_ms'] = metrics['parallel_retrieval_ms'] # Upper bound

    # Claim extraction
    t0 = time.time()
    if mock_llm_responses:
        claims = ["Mocked claim 1", "Mocked claim 2"]
    else:
        claims = await extract_claims(query, video_evidence)
    t1 = time.time()
    metrics['claim_extraction_ms'] = (t1 - t0) * 1000

    # Validation
    t0 = time.time()
    validation = await validate_claims(claims, web_results)
    t1 = time.time()
    metrics['validation_ms'] = (t1 - t0) * 1000
    
    agg_status = aggregate_status(validation)

    # Final LLM
    t0 = time.time()
    answer = await generate_answer(
        query=query,
        video_evidence=video_evidence,
        validation=validation,
        external_evidence=collect_evidence(validation)
    )
    t1 = time.time()
    metrics['final_llm_ms'] = (t1 - t0) * 1000

    metrics['total_ms'] = sum(v for k,v in metrics.items() if not k.endswith('_upper_bound') and k != 'total_ms')
    
    return {
        "status": agg_status, 
        "metrics": metrics, 
        "answer": answer, 
        "sources": [r.url for r in web_results],
        "claims": claims
    }

async def run_all_tests():
    video_id = "ukzFI9rgwfU" # example video about machine learning
    
    print("==================================================")
    print("TEST 1 — NORMAL RAG")
    print("==================================================")
    try:
        res1 = await run_pipeline("What is the main topic discussed in this video?", video_id, False)
        print("Result:", res1['status'])
        print("Metrics:", res1['metrics'])
    except Exception as e:
        print("Failed:", e)

    print("\n==================================================")
    print("TEST 2 — VALIDATED QUERY")
    print("==================================================")
    try:
        res2 = await run_pipeline("Is the claim made in the video about Adam optimizer correct according to external sources?", video_id, True)
        print("Result:", res2['status'])
        print("Sources:", res2['sources'])
        print("Metrics:", res2['metrics'])
    except Exception as e:
        print("Failed:", e)

    print("\n==================================================")
    print("TEST 3 — CONTRADICTION")
    print("==================================================")
    try:
        # Mocking the claim extraction to force a contradiction for testing logic
        res3 = await run_pipeline("Is the sky green according to the video?", video_id, True)
        print("Result:", res3['status'])
    except Exception as e:
        print("Failed:", e)
        
    print("\n==================================================")
    print("TEST 4 — UNVERIFIED")
    print("==================================================")
    try:
        res4 = await run_pipeline("Did the video mention the fictitious Xylophone-9 algorithm?", video_id, True)
        print("Result:", res4['status'])
    except Exception as e:
        print("Failed:", e)

if __name__ == "__main__":
    asyncio.run(run_all_tests())
