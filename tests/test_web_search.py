import pytest
import asyncio
from unittest.mock import patch
from backend.search.web_search import SerperSearchProvider
from schemas.models import SearchResult

@pytest.fixture
def serper_provider(monkeypatch):
    monkeypatch.setattr("backend.search.web_search.SEARCH_API_KEY", "dummy_key")
    return SerperSearchProvider()

class FakeResponse:
    def __init__(self, status, json_data=None):
        self.status = status
        self._json_data = json_data
    
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
    async def json(self):
        return self._json_data

class FakeSession:
    def __init__(self, response=None, exception=None):
        self.response = response
        self.exception = exception
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def post(self, url, **kwargs):
        if self.exception:
            raise self.exception
        return self.response

@pytest.mark.asyncio
async def test_serper_successful_search(serper_provider):
    mock_response_data = {
        "organic": [
            {"title": "Test Title 1", "link": "https://example.com", "snippet": "Snippet 1"},
            {"title": "Test Title 2", "link": "https://example.org", "snippet": "Snippet 2"}
        ]
    }
    
    fake_session = FakeSession(response=FakeResponse(200, mock_response_data))
    
    with patch('aiohttp.ClientSession', return_value=fake_session):
        results = await serper_provider.search("test query", top_k=2)
        
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].title == "Test Title 1"
        assert results[0].domain == "example.com"
        assert results[1].domain == "example.org"

@pytest.mark.asyncio
async def test_serper_empty_search(serper_provider):
    mock_response_data = {"organic": []}
    fake_session = FakeSession(response=FakeResponse(200, mock_response_data))
    
    with patch('aiohttp.ClientSession', return_value=fake_session):
        results = await serper_provider.search("test query", top_k=2)
        assert len(results) == 0

@pytest.mark.asyncio
async def test_serper_401_403(serper_provider):
    fake_session = FakeSession(response=FakeResponse(401))
    with patch('aiohttp.ClientSession', return_value=fake_session):
        results = await serper_provider.search("test query", top_k=2)
        assert len(results) == 0

@pytest.mark.asyncio
async def test_serper_429(serper_provider):
    fake_session = FakeSession(response=FakeResponse(429))
    with patch('aiohttp.ClientSession', return_value=fake_session):
        results = await serper_provider.search("test query", top_k=2)
        assert len(results) == 0

@pytest.mark.asyncio
async def test_serper_timeout(serper_provider):
    fake_session = FakeSession(exception=asyncio.TimeoutError())
    with patch('aiohttp.ClientSession', return_value=fake_session):
        results = await serper_provider.search("test query", top_k=2)
        assert len(results) == 0

@pytest.mark.asyncio
async def test_serper_malformed_response(serper_provider):
    fake_session = FakeSession(response=FakeResponse(200, {"error": "some error"}))
    with patch('aiohttp.ClientSession', return_value=fake_session):
        results = await serper_provider.search("test query", top_k=2)
        assert len(results) == 0

@pytest.mark.asyncio
async def test_serper_top_k_behavior(serper_provider):
    mock_response_data = {
        "organic": [
            {"title": f"Test Title {i}", "link": "https://example.com", "snippet": "Snippet"}
            for i in range(5)
        ]
    }
    fake_session = FakeSession(response=FakeResponse(200, mock_response_data))
    with patch('aiohttp.ClientSession', return_value=fake_session):
        results = await serper_provider.search("test query", top_k=3)
        assert len(results) == 3
