## Getting Started

### Prerequisites

- Python 3.9 or higher
- `pip` package manager

### Installation & Usage

1. **Install Python dependencies:**

  ```bash
  pip install -r requirements.txt
  ```

2. **Run the backend server:**

  ```bash
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
  ```

  The backend API will be available at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

3. **Check the backend using a curl command:**

  ```bash
  curl -X POST http://127.0.0.1:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"query": "what is machine learning?", "video_id": "ukzFI9rgwfU"}'
  ```

  You should receive a JSON response with the answer.

4. **Install the Chrome Extension in developer mode:**

  - Open Chrome and go to `chrome://extensions/`
  - Enable "Developer mode" (toggle in the top right)
  - Click "Load unpacked" and select the extension folder from this repository

---

## API Usage

### Ask a Question

Send a POST request to `/ask` endpoint with a JSON payload containing the `query` and the `video_id` of the YouTube video whose transcript you want to query.

**Example using curl:**

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "what is machine learning?", "video_id": "ukzFI9rgwfU"}'
```

**Sample Response:**

```json
{
  "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
  "source": "Transcript snippet or reference"
}
```
