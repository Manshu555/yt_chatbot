import os
from dotenv import load_dotenv

# Explicitly load .env from the same directory as config.py
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_BATCH_MODEL = os.getenv("OPENAI_BATCH_MODEL", OPENAI_MODEL)
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

TOP_K_TRANSCRIPT = int(os.getenv("TOP_K_TRANSCRIPT", "5"))
TOP_K_WEB = int(os.getenv("TOP_K_WEB", "3"))
TOP_K_EVIDENCE = int(os.getenv("TOP_K_EVIDENCE", "5"))

WEB_TIMEOUT_SECONDS = int(os.getenv("WEB_TIMEOUT_SECONDS", "3"))
PAGE_FETCH_TIMEOUT_SECONDS = int(os.getenv("PAGE_FETCH_TIMEOUT_SECONDS", "3"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
GEMINI_CALL_TIMEOUT_SECONDS = float(os.getenv("GEMINI_CALL_TIMEOUT_SECONDS", "40"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "150"))
TRANSCRIPT_TIMEOUT_SECONDS = float(os.getenv("TRANSCRIPT_TIMEOUT_SECONDS", "20"))

VALIDATION_ENABLED_BY_DEFAULT = os.getenv("VALIDATION_ENABLED_BY_DEFAULT", "true").lower() == "true"
MAX_CLAIMS = int(os.getenv("MAX_CLAIMS", "3"))
