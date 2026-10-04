import aiohttp
from bs4 import BeautifulSoup
from schemas.models import SearchResult
from config import PAGE_FETCH_TIMEOUT_SECONDS

async def fetch_page_text(result: SearchResult) -> str:
    """Fetches a webpage and extracts text."""
    try:
        timeout = aiohttp.ClientTimeout(total=PAGE_FETCH_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(result.url, allow_redirects=True) as response:
                if response.status != 200:
                    return ""
                
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type and 'text/plain' not in content_type:
                    return ""
                    
                text = await response.text()
                if 'text/html' in content_type:
                    soup = BeautifulSoup(text, "html.parser")
                    for element in soup(["script", "style", "nav", "header", "footer", "noscript", "form"]):
                        element.decompose()
                    content = soup.find("main") or soup.find("article") or soup
                    text = content.get_text(separator="\n", strip=True)
                return text[:10000]
    except Exception as e:
        print(f"Error fetching source ({type(e).__name__}).")
        return ""
