# YT Chatbot --- Multi-Source Validated RAG

This project is a YouTube Transcript RAG Chrome Extension and backend that supports Multi-Source Validated RAG. It extracts transcripts, analyzes user queries, optionally cross-references claims with external web sources, and returns an evidence-backed answer.

## Architecture
- **Chrome Extension**: Simple UI for asking questions with an option to "Validate externally".
- **FastAPI Backend**: Orchestrates the RAG flow.
- **YouTube Retrieval**: Uses `yt-dlp` and `webvtt` to fetch and parse transcripts, chunks them, and stores in ChromaDB.
- **Web Search**: Fetches external pages and extracts evidence.
- **LLM**: Uses Hugging Face Inference API (`mistralai/Mixtral-8x7B-Instruct-v0.1`) for claim extraction and final answer synthesis.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- `pip` package manager
- `yt-dlp` installed and accessible

### Installation & Usage

1. **Install Python dependencies:**

  ```bash
  cd backend
  pip install -r requirements.txt
  ```

2. **Set up Environment Variables:**
  Copy `.env.example` to `.env` and configure `HUGGINGFACE_API_KEY`.

3. **Run the backend server:**

  ```bash
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
  ```

  The backend API will be available at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

4. **Install the Chrome Extension in developer mode:**

  - Open Chrome and go to `chrome://extensions/`
  - Enable "Developer mode" (toggle in the top right)
  - Click "Load unpacked" and select the `extension` folder from this repository.

### API Usage

Send a POST request to `/ask` endpoint with a JSON payload containing the `query`, `video_id`, and `validate_externally` boolean.

**Example using curl:**

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "When was Adam optimizer introduced?", "video_id": "ukzFI9rgwfU", "validate_externally": true}'
```

**Troubleshooting**
- Ensure `yt-dlp` is in your PATH.
- Verify that your Hugging Face API key is correct and has access.
- For local development, CORS is configured to allow `*`. In production, restrict this.
