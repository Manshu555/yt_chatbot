from pydantic import BaseModel
from typing import List, Optional

class QueryInput(BaseModel):
    query: str
    video_id: str
    validate_externally: bool = True

class TranscriptChunk(BaseModel):
    id: str
    text: str
    score: Optional[float] = None

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    domain: Optional[str] = None

class ExternalEvidence(BaseModel):
    source_title: str
    url: str
    domain: str
    passage: str
    relevance_score: float

class Claim(BaseModel):
    id: str
    text: str

class ClaimValidation(BaseModel):
    claim_id: str
    claim: str
    status: str
    supporting_evidence: List[ExternalEvidence] = []
    contradicting_evidence: List[ExternalEvidence] = []
    explanation: str

class AskResponse(BaseModel):
    answer: str
    validation_required: bool
    validation_status: Optional[str]
    video_evidence: List[TranscriptChunk]
    claims: List[ClaimValidation]
    sources: List[SearchResult]
