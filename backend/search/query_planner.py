"""Keep discovery and ranking focused on the user's information need."""
import re

from rag.ranking import terms


SOCIAL_HOSTS = ("facebook.com", "instagram.com", "tiktok.com", "pinterest.com")


def wants_social_sources(question):
    return bool(re.search(r"\b(facebook|instagram|tiktok|pinterest|social media|social posts?)\b", question, re.I))


def focused_question(question):
    text = re.sub(r"\b(?:in|from|according to) (?:this |the )?video\b", "", question, flags=re.I)
    text = re.sub(r"\b(?:please |can you |could you )", "", text, flags=re.I)
    return " ".join(text.split()).strip(" ,:?")


def ranking_query(question, claims=()):
    """Use claims to resolve vague requests, never to dilute a specific question."""
    focused = focused_question(question)
    generic = {"verify", "check", "fact", "factcheck", "true", "correct", "accurate", "accuracy",
               "claims", "claim", "external", "evidence", "research", "sources", "source",
               "give", "additional", "context", "it", "its", "this", "these", "that", "me"}
    generic.update({"overview", "key", "points", "takeaways", "main", "ideas"})
    if set(terms(focused)) - generic:
        return focused
    grounded = " ".join(claims).strip()
    return grounded[:600] or focused


def discovery_queries(question, claims=()):
    claims = [claim for claim in claims if claim.strip()]
    primary = ranking_query(question, claims)
    queries = [primary]
    # A second, short transcript-grounded query helps paraphrases without
    # concatenating every claim into an over-constrained search.
    topic = set(terms(primary))
    relevant = [claim for claim in claims if topic & set(terms(claim))]
    if relevant and primary == focused_question(question):
        best = max(relevant, key=lambda claim: len(topic & set(terms(claim))) / max(1, len(set(terms(claim)))))
        secondary = " ".join(dict.fromkeys(terms(primary) + terms(best)))
        if len(secondary) <= 240 and set(terms(secondary)) != topic:
            queries.append(secondary)
    if not wants_social_sources(question):
        exclusions = " ".join(f"-site:{host}" for host in SOCIAL_HOSTS)
        queries = [f"{query} {exclusions}" for query in queries]
    return queries


def fallback_queries(question, claims=()):
    """Retry without domain exclusions and simplify long conversational queries."""
    primary = ranking_query(question, claims)
    queries = [primary]
    compact = " ".join(dict.fromkeys(terms(primary)))
    compact = " ".join(compact.split()[:12])
    if compact and compact.lower() != primary.lower():
        queries.append(compact)
    return queries
