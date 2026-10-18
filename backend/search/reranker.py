"""Candidate discovery stays with Serper/DDG; rank content before validation."""
import asyncio
import logging
import re
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from config import WEB_MIN_SIMILARITY, WEB_MIN_LEXICAL_COVERAGE, WEB_MAX_SIMILARITY_GAP
from rag.embeddings import embed_texts, cosine
from rag.ranking import bm25, fuse, near_duplicate, terms
from search.source_fetcher import fetch_page_text
from search.query_planner import SOCIAL_HOSTS, wants_social_sources, ranking_query

logger = logging.getLogger(__name__)


def canonical_url(url):
    try:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return ""
        query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                 if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}]
        host = parts.netloc.lower().removeprefix("www.")
        return urlunsplit(("https", host, parts.path.rstrip("/"), urlencode(sorted(query)), ""))
    except ValueError:
        return ""


def _tool_page(result, query):
    # Intent gate, not a domain blacklist: tools are valid for tool questions.
    if re.search(r"\b(tool|app|software|service|product|summarizer|summariser)\b", query.lower()):
        return False
    return bool(re.search(r"(?:youtube|video|ai).{0,35}summari[sz](?:er|ation)|summari[sz]e.{0,30}(?:youtube|videos)",
                          (result.title + " " + result.snippet).lower()))


async def rerank_results(question, results, top_k=3, context="", allow_weak=False, page_cache=None):
    query = ranking_query(question, [context] if context else [])
    seen, candidates, provider_ranks, deferred = set(), [], [], []
    for rank, result in enumerate(results, 1):
        url = canonical_url(result.url)
        if not url or url in seen:
            continue
        seen.add(url)
        host = (urlsplit(url).hostname or "").removeprefix("www.")
        social = not wants_social_sources(question) and any(host == social or host.endswith("." + social) for social in SOCIAL_HOSTS)
        if _tool_page(result, question) or social:
            deferred.append((rank, result))
            continue
        candidates.append(result)
        provider_ranks.append(rank)
    if allow_weak and not candidates:
        provider_ranks = [rank for rank, _ in deferred]
        candidates = [result for _, result in deferred]
    if not candidates or top_k <= 0:
        return []
    page_cache = {} if page_cache is None else page_cache

    async def enrich(result):
        if result.passage:
            return result
        key = canonical_url(result.url)
        if key not in page_cache:
            page_cache[key] = await fetch_page_text(result)
        page = page_cache[key]
        return result.model_copy(update={"passage": page[:10000]})

    candidates = await asyncio.gather(*(enrich(result) for result in candidates))
    return await asyncio.to_thread(_rank, query, candidates, top_k, provider_ranks, allow_weak)


def _rank(query, candidates, top_k, provider_ranks=None, allow_weak=False):
    # Split available content so a late relevant paragraph survives truncation.
    provider_ranks = provider_ranks or list(range(1, len(candidates) + 1))
    passages, owners, content_windows, origins = [], [], [], []
    for i, result in enumerate(candidates):
        if not (result.snippet.strip() or result.passage.strip()):
            continue  # title alone is insufficient evidence
        # Keep the snippet separate from the article. Word-aligned overlapping
        # windows avoid burying a useful paragraph in navigation or unrelated text.
        for origin, content in (("snippet", result.snippet), ("passage", result.passage[:10000])):
            words = content.split()
            for start in range(0, len(words), 75):
                window = " ".join(words[start:start + 100])
                passages.append(window)
                owners.append(i)
                content_windows.append(window)
                origins.append(origin)
                if start + 100 >= len(words):
                    break
    texts = [" ".join([r.title, r.snippet, r.passage]) for r in candidates]
    lexical = bm25(query, texts)
    dense = [0.0] * len(candidates)
    best = [r.passage for r in candidates]
    best_source = ["snippet"] * len(candidates)
    semantic = False
    try:
        vectors = embed_texts([query, *passages])
        for passage, owner, origin, vector in zip(content_windows, owners, origins, vectors[1:]):
            similarity = cosine(vectors[0], vector)
            if similarity > dense[owner]:
                dense[owner], best[owner] = similarity, passage
                best_source[owner] = origin
        semantic = True
    except Exception as error:
        logger.warning("External reranking fallback=BM25 error=%s", type(error).__name__)

    fused = fuse(dense, lexical)
    best_similarity = max(dense, default=0)
    query_terms = set(terms(query))
    ranked = []
    for i, result in enumerate(candidates):
        content_terms = set(terms(result.snippet + " " + result.passage))
        coverage = len(query_terms & content_terms) / max(1, len(query_terms))
        available = bool(result.snippet.strip() or result.passage.strip())
        if not available and not (allow_weak and result.title.strip()):
            continue
        passes = available and (dense[i] >= WEB_MIN_SIMILARITY and best_similarity - dense[i] <= WEB_MAX_SIMILARITY_GAP
                                if semantic else coverage >= WEB_MIN_LEXICAL_COVERAGE)
        if not passes and not allow_weak:
            continue
        quality = 1.0 if (urlsplit(result.url).hostname or "").endswith((".gov", ".edu")) else 0.5
        # Absolute cosine dominates; RRF preserves lexical identifiers. Provider
        # position, content and source-type priors are deliberately small.
        score = (0.7 * max(0, dense[i]) + 0.2 * fused[i] if semantic else 0.9 * coverage)
        score += 0.05 / provider_ranks[i] + 0.03 * quality + 0.02 * (best_source[i] == "passage")
        reason = (f"MiniLM cosine={dense[i]:.3f}; dense/BM25 RRF={fused[i]:.3f}; "
                  if semantic else f"BM25 fallback; term coverage={coverage:.3f}; ")
        reason += f"provider rank={provider_ranks[i]}; content={best_source[i] if semantic else 'lexical'}. Relevance is not factual support."
        if not passes:
            reason = "Closest available source; low topical match. " + reason
        if not available:
            reason += " Only the search-result title is available; page content could not be checked."
        ranked.append(result.model_copy(update={"passage": best[i][:1200],
                                                "relevance_score": round(min(100, 100 * score), 2),
                                                "relevance_reason": reason}))
        logger.info("External candidate rank=%d cosine=%.3f BM25=%.3f score=%.2f", provider_ranks[i], dense[i], lexical[i], 100 * score)
    ranked.sort(key=lambda r: r.relevance_score, reverse=True)
    selected = []
    for result in ranked:
        if any(near_duplicate(result.snippet + " " + result.passage, old.snippet + " " + old.passage) for old in selected):
            continue
        selected.append(result)
        if len(selected) >= top_k:
            break
    logger.info("External retrieval method=%s candidates=%d returned=%d filtered=%d",
                "MiniLM+BM25/RRF" if semantic else "BM25-fallback", len(candidates), len(selected), len(candidates) - len(selected))
    return selected
