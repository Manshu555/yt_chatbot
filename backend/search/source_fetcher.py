import aiohttp
import re
from schemas.models import SearchResult
from config import PAGE_FETCH_TIMEOUT_SECONDS

async def fetch_page_text(result: SearchResult) -> str:
    """Fetches a webpage and extracts text."""
    try:
        # Quick mock bypass for mock search results
        if result.url.startswith("https://example.com") or result.url.startswith("https://example.org"):
            return f"This is mock page content for {result.url}. " + result.snippet
            
        timeout = aiohttp.ClientTimeout(total=PAGE_FETCH_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(result.url, allow_redirects=True) as response:
                if response.status != 200:
                    return ""
                
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type and 'text/plain' not in content_type:
                    return ""
                    
                text = await response.text()
                # Basic HTML stripping
                text = re.sub(r'<style.*?>.*?</style>', '', text, flags=re.DOTALL)
                text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL)
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text[:10000] # Limit size
    except Exception as e:
        print(f"Error fetching {result.url}: {e}")
        return ""
