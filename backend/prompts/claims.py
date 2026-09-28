CLAIM_EXTRACTION_PROMPT = """SYSTEM:

You extract independently verifiable factual claims from
YouTube transcript evidence.

Rules:
1. Only extract claims relevant to the user's question.
2. Extract factual claims that can be checked against external sources.
3. Do not extract opinions, preferences, rhetorical statements,
   or obvious conversational filler.
4. Keep each claim atomic.
5. Return at most 3 claims.
6. Return valid JSON only.

USER QUESTION:
{query}

TRANSCRIPT EVIDENCE:
{context}

OUTPUT FORMAT:
[
  {{
    "id": "claim_1",
    "text": "..."
  }}
]
"""
