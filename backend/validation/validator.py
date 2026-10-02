import asyncio
import json
import os
from typing import List, Dict
from pydantic import BaseModel, Field, ValidationError
from schemas.models import Claim, ClaimValidation, ExternalEvidence, SearchResult
from search.evidence_extractor import extract_evidence
from google import genai
from rag.gemini_client import call_gemini_with_retry

class BatchValidationResult(BaseModel):
    claim_id: str
    status: str = Field(description="Must be exactly: SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, or UNVERIFIED")
    evidence_ids: List[str] = Field(description="List of evidence IDs that support or contradict the claim")
    reason: str

class BatchValidationResponse(BaseModel):
    results: List[BatchValidationResult]

def _determine_status(supports: int, contradicts: int) -> str:
    if supports > 0 and contradicts == 0:
        return "SUPPORTED"
    elif supports > 0 and contradicts > 0:
        return "PARTIALLY_SUPPORTED"
    elif supports == 0 and contradicts > 0:
        return "CONTRADICTED"
    else:
        return "UNVERIFIED"

def _safe_unverified(claims: List[Claim], explanation: str) -> List[ClaimValidation]:
    return [
        ClaimValidation(
            claim_id=c.id,
            claim=c.text,
            status="UNVERIFIED",
            explanation=explanation,
        )
        for c in claims
    ]

def _response_to_batch_validation(response) -> BatchValidationResponse:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, BatchValidationResponse):
        return parsed
    if isinstance(parsed, dict):
        return BatchValidationResponse.model_validate(parsed)

    text = getattr(response, "text", None)
    if not text:
        raise ValueError("empty response")
    return BatchValidationResponse.model_validate_json(text)

async def validate_claims(claims: List[Claim], web_results_prefetched: List[SearchResult]) -> List[ClaimValidation]:
    if not claims:
        return []

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key or gemini_key == "dummy_key":
        return _safe_unverified(claims, "No API key.")
        
    model_id = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    
    # 1. Gather all evidence for all claims
    all_evidence_tasks = []
    
    for claim in claims:
        for result in web_results_prefetched:
            all_evidence_tasks.append(extract_evidence(claim.text, result))
            
    all_evidence = await asyncio.gather(*all_evidence_tasks)
    
    # 2. Deduplicate evidence and assign IDs
    evidence_map: Dict[str, ExternalEvidence] = {}
    passage_to_id: Dict[str, str] = {}
    
    for ev in all_evidence:
        if ev and ev.passage:
            # Simple deduplication by passage text
            if ev.passage not in passage_to_id:
                ev_id = f"ev_{len(evidence_map) + 1}"
                passage_to_id[ev.passage] = ev_id
                evidence_map[ev_id] = ev
    
    if not evidence_map:
        return _safe_unverified(claims, "No evidence found.")
        
    # 3. Construct the batch prompt
    prompt = "You are a strict fact-checking assistant. Evaluate each claim against the provided evidence.\n\n"
    
    prompt += "CLAIMS:\n"
    for claim in claims:
        prompt += f"[{claim.id}] {claim.text}\n"
        
    prompt += "\nEVIDENCE:\n"
    for ev_id, ev in evidence_map.items():
        prompt += f"[{ev_id}] Source: {ev.source_title} - Passage: {ev.passage}\n"
        
    prompt += "\nFor each claim, determine if the evidence as a whole supports it, partially supports it, contradicts it, or provides no verifiable information.\n"
    prompt += "Use exactly one of these statuses for each claim: SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, UNVERIFIED.\n"
    prompt += "Use UNVERIFIED when evidence is insufficient. Return one result for every claim in the requested JSON schema."

    # 4. Call Gemini
    client = genai.Client(api_key=gemini_key)
    
    try:
        response = await call_gemini_with_retry(
            client=client,
            model_id=model_id,
            prompt=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1000,
                top_p=0.95,
                response_mime_type="application/json",
                response_schema=BatchValidationResponse,
                automatic_function_calling={"disable": True},
                thinking_config={"thinking_level": "low"}
            )
        )
        
        if not response:
            raise ValueError("Empty response from batch validation.")
        parsed_response = _response_to_batch_validation(response)
        results_data = [r.model_dump() for r in parsed_response.results]
        
        # 5. Map results back to claims
        validations = []
        result_dict = {r.get("claim_id"): r for r in results_data}
        
        for claim in claims:
            res = result_dict.get(claim.id)
            if not res:
                validations.append(ClaimValidation(
                    claim_id=claim.id,
                    claim=claim.text,
                    status="UNVERIFIED",
                    explanation="Gemini failed to return a result for this claim."
                ))
                continue
                
            status = str(res.get("status", "UNVERIFIED")).upper()
            if status not in ["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "UNVERIFIED"]:
                status = "UNVERIFIED"
                
            ev_ids = res.get("evidence_ids", [])
            matched_evidence = [evidence_map[eid] for eid in ev_ids if eid in evidence_map]
            
            supporting = matched_evidence if status in ["SUPPORTED", "PARTIALLY_SUPPORTED"] else []
            contradicting = matched_evidence if status == "CONTRADICTED" else []
            
            validations.append(ClaimValidation(
                claim_id=claim.id,
                claim=claim.text,
                status=status,
                supporting_evidence=supporting,
                contradicting_evidence=contradicting,
                explanation=res.get("reason", "No reason provided.")
            ))
            
        return validations
        
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        print(f"Batch classification parse error: {e}")
        return _safe_unverified(claims, "Validation failed due to malformed Gemini response.")
    except Exception as e:
        print(f"Batch classification error: {e}")
        return _safe_unverified(claims, "Validation failed due to API error.")
