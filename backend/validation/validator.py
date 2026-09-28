import asyncio
import os
from typing import List
from schemas.models import Claim, ClaimValidation, ExternalEvidence, SearchResult
from search.evidence_extractor import extract_evidence
from google import genai
from google.genai.errors import APIError

def _determine_status(supports: int, contradicts: int) -> str:
    if supports > 0 and contradicts == 0:
        return "SUPPORTED"
    elif supports > 0 and contradicts > 0:
        return "PARTIALLY_SUPPORTED"
    elif supports == 0 and contradicts > 0:
        return "CONTRADICTED"
    else:
        return "UNVERIFIED"

async def _classify_evidence(claim_text: str, evidence: ExternalEvidence) -> str:
    """Uses LLM to classify if evidence supports or contradicts the claim."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key or gemini_key == "dummy_key":
        return "UNVERIFIED"
        
    model_id = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    
    try:
        client = genai.Client(api_key=gemini_key)
        
        from rag.gemini_client import call_gemini_with_retry
        
        prompt = f"""You are a strict fact-checking assistant.
Claim: "{claim_text}"
Evidence: "{evidence.passage}"

Does the evidence support the claim, contradict the claim, or provide no verifiable information about the claim?
Respond with exactly ONE word: SUPPORT, CONTRADICT, or UNVERIFIED.
"""
        
        response = await call_gemini_with_retry(
            client=client,
            model_id=model_id,
            prompt=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=10,
                top_p=0.95,
                automatic_function_calling={"disable": True}
            )
        )
        
        if not response or not response.text:
            return "UNVERIFIED"
            
        answer = response.text.strip().upper()
        if "SUPPORT" in answer and "CONTRADICT" not in answer:
            return "SUPPORT"
        elif "CONTRADICT" in answer:
            return "CONTRADICT"
        return "UNVERIFIED"
    except Exception as e:
        print(f"Classification error: {e}")
        return "UNVERIFIED"

async def validate_claims(claims: List[Claim], web_results_prefetched: List[SearchResult]) -> List[ClaimValidation]:
    validations = []
    
    for claim in claims:
        # Just use prefetched for now to minimize latency
        evidence_tasks = [extract_evidence(claim.text, result) for result in web_results_prefetched]
        evidence_list = await asyncio.gather(*evidence_tasks)
        
        supporting = []
        contradicting = []
        
        if evidence_list:
            # Run classifications in parallel
            classifications = await asyncio.gather(*[_classify_evidence(claim.text, ev) for ev in evidence_list])
            
            for ev, classification in zip(evidence_list, classifications):
                if classification == "SUPPORT":
                    supporting.append(ev)
                elif classification == "CONTRADICT":
                    contradicting.append(ev)
                
        status = _determine_status(len(supporting), len(contradicting))
        
        validations.append(
            ClaimValidation(
                claim_id=claim.id,
                claim=claim.text,
                status=status,
                supporting_evidence=supporting,
                contradicting_evidence=contradicting,
                explanation=f"Found {len(supporting)} supporting and {len(contradicting)} contradicting sources."
            )
        )
        
    return validations
