import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

TOP_K_TRANSCRIPT = int(os.getenv("TOP_K_TRANSCRIPT", "5"))
TOP_K_WEB = int(os.getenv("TOP_K_WEB", "3"))
TOP_K_EVIDENCE = int(os.getenv("TOP_K_EVIDENCE", "5"))

WEB_TIMEOUT_SECONDS = int(os.getenv("WEB_TIMEOUT_SECONDS", "3"))
PAGE_FETCH_TIMEOUT_SECONDS = int(os.getenv("PAGE_FETCH_TIMEOUT_SECONDS", "3"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "15"))

VALIDATION_ENABLED_BY_DEFAULT = os.getenv("VALIDATION_ENABLED_BY_DEFAULT", "true").lower() == "true"
MAX_CLAIMS = int(os.getenv("MAX_CLAIMS", "3"))
