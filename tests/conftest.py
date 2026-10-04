import os

# Unit tests mock Gemini; never initialize them with the developer's real key.
os.environ["GEMINI_API_KEY"] = "unit-test-key"
