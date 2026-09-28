from typing import List
import aiohttp
import asyncio
from schemas.models import SearchResult
from config import SEARCH_API_KEY, WEB_TIMEOUT_SECONDS
from urllib.parse import urlparse

class SearchProvider:
    async def search(self, query: str, top_k: int) -> List[SearchResult]:
        raise NotImplementedError

class MockSearchProvider(SearchProvider):
    async def search(self, query: str, top_k: int) -> List[SearchResult]:
        print(f"Mocking search for: {query}")
        return [
            SearchResult(
                title=f"Mock Result 1 for {query}",
                url="https://example.com/mock1",
                snippet="Adam was introduced in 2014.",
                domain="example.com"
            ),
            SearchResult(
                title=f"Mock Result 2 for {query}",
                url="https://example.org/mock2",
                snippet="Another source confirming 2014.",
                domain="example.org"
            )
        ][:top_k]

class SerperSearchProvider(SearchProvider):
    async def search(self, query: str, top_k: int) -> List[SearchResult]:
        if not SEARCH_API_KEY:
            print("SEARCH_API_KEY not configured. Search aborted.")
            return []
            
        url = "https://google.serper.dev/search"
        payload = {
            "q": query,
            "num": top_k
        }
        headers = {
            'X-API-KEY': SEARCH_API_KEY,
            'Content-Type': 'application/json'
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=WEB_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 401 or response.status == 403:
                        print("Search API authentication failed (401/403).")
                        return []
                    if response.status == 429:
                        print("Search API rate limit exceeded (429).")
                        return []
                    if response.status != 200:
                        print(f"Search API returned error status: {response.status}")
                        return []
                        
                    data = await response.json()
                    
                    if not data or "organic" not in data:
                        print("Empty or malformed search response.")
                        return []
                        
                    results = []
                    for item in data["organic"][:top_k]:
                        link = item.get("link", "")
                        domain = urlparse(link).netloc if link else None
                        results.append(SearchResult(
                            title=item.get("title", "No Title"),
                            url=link,
                            snippet=item.get("snippet", ""),
                            domain=domain
                        ))
                    return results
        except asyncio.TimeoutError:
            print("Search API request timed out.")
            return []
        except Exception as e:
            print(f"Search API request failed: {e}")
            return []

class DDGSearchProvider(SearchProvider):
    async def search(self, query: str, top_k: int) -> List[SearchResult]:
        try:
            # Run the synchronous duckduckgo_search in a thread
            from duckduckgo_search import DDGS
            def _do_search():
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=top_k))
                    
            results_data = await asyncio.to_thread(_do_search)
            
            results = []
            for item in results_data:
                link = item.get("href", "")
                domain = urlparse(link).netloc if link else None
                results.append(SearchResult(
                    title=item.get("title", "No Title"),
                    url=link,
                    snippet=item.get("body", ""),
                    domain=domain
                ))
            return results
        except Exception as e:
            print(f"DDG Search failed: {e}")
            return []

def get_search_provider() -> SearchProvider:
    if SEARCH_API_KEY and SEARCH_API_KEY.strip():
        return SerperSearchProvider()
    return DDGSearchProvider()

async def search_web(query: str, top_k: int) -> List[SearchResult]:
    provider = get_search_provider()
    try:
        return await provider.search(query, top_k)
    except Exception as e:
        print(f"Web search failed: {e}")
        return []
