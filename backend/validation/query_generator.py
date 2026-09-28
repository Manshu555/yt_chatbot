from schemas.models import Claim

def generate_search_query(claim: Claim) -> str:
    """Generates a search query from a claim."""
    return claim.text
