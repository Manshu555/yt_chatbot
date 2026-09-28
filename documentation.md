# YouTube Transcript Chatbot — Detailed Technical Documentation

## 1. Project Overview

YouTube Transcript Chatbot is a Chrome-extension + FastAPI + RAG application that allows a user to ask questions about the currently open YouTube video.

The current implementation has two major layers:

- **Chrome Extension** — identifies the active YouTube video, accepts a question, calls the backend, and displays the answer.
- **Python Backend** — obtains the English transcript, indexes it in ChromaDB, retrieves relevant transcript chunks, builds a RAG prompt, and calls `mistralai/Mixtral-8x7B-Instruct-v0.1` through the Hugging Face Inference API.

### High-level architecture

```text
                    +----------------------+
                    |       YouTube        |
                    |    Current Video     |
                    +----------+-----------+
                               |
                               | URL / video ID
                               v
                    +----------------------+
                    |   Chrome Extension   |
                    |----------------------|
                    | popup.html            |
                    | popup.js              |
                    | style.css             |
                    +----------+-----------+
                               |
                               | POST /ask
                               | query + video_id
                               v
                    +----------------------+
                    |       FastAPI        |
                    |       app.py         |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |      model.py        |
                    |----------------------|
                    | Collection lookup    |
                    | Transcript retrieval |
                    | VTT parsing          |
                    | Chunking             |
                    | Chroma retrieval     |
                    | Prompt construction  |
                    | LLM API call         |
                    +----+-------------+---+
                         |             |
                         v             v
                 +-------------+ +-------------+
                 |  ChromaDB   | | HuggingFace |
                 | Transcript  | | Inference   |
                 | Collections | | Mixtral     |
                 +-------------+ +------+------+ 
                                        |
                                        v
                                  Generated Answer
                                        |
                                        v
                                  Chrome Popup
```

---

## 2. Repository Structure

```text
yt_chatbot/
│
├── README.md
├── .gitignore
│
├── backend/
│   ├── app.py
│   ├── model.py
│   ├── requirements.txt
│   ├── Procfile
│   └── test.py
│
└── extension/
    ├── manifest.json
    ├── popup.html
    ├── popup.js
    ├── style.css
    └── icon.png
```

The repository also contains generated `__pycache__` and `.DS_Store` files; these are not part of the application's logical architecture.

---

# 3. End-to-End Request Flow

Suppose the user opens a YouTube video and asks:

> What is machine learning?

The complete flow is:

```text
1. User opens YouTube video
        |
2. Chrome extension reads active tab
        |
3. Extension validates YouTube URL
        |
4. Extension extracts video ID
        |
5. User enters question
        |
6. Extension sends POST /ask
        |  {query, video_id}
        v
7. FastAPI validates request
        |
8. query_transcript(query, video_id)
        |
9. Check Chroma collection for video
        |
   +----+-------------------+
   |                        |
 exists                  missing
   |                        |
   |                        v
   |                     yt-dlp
   |                        |
   |                        v
   |                  English VTT
   |                        |
   |                        v
   |                     webvtt
   |                        |
   |                        v
   |                 transcript text
   |                        |
   |                        v
   |                 sentence chunks
   |                        |
   |                        v
   |                    ChromaDB
   |                        |
   +-----------+------------+
               |
               v
10. Retrieve top 5 relevant chunks
               |
11. Build context
               |
12. Build RAG prompt
               |
13. Call Hugging Face
               |
14. Mixtral generates answer
               |
15. FastAPI returns {answer: ...}
               |
16. Extension displays answer
```

---

# 4. Backend Architecture

## 4.1 `backend/app.py`

`app.py` is the backend entry point.

### Imports

```python
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from model import load_model, query_transcript
from dotenv import load_dotenv
```

Responsibilities of the imports:

| Import | Purpose |
|---|---|
| `os` | Reads environment variables such as deployment port |
| `FastAPI` | Creates REST API |
| `BaseModel` | Validates request JSON |
| `CORSMiddleware` | Enables browser-originated requests |
| `load_model` | Validates Hugging Face configuration |
| `query_transcript` | Runs the RAG pipeline |
| `load_dotenv` | Loads `.env` values |

---

## 4.2 Environment Configuration

The backend executes:

```python
load_dotenv()
```

The important environment variable is:

```text
HUGGINGFACE_API_KEY
```

A local `.env` can contain:

```env
HUGGINGFACE_API_KEY=your_huggingface_token
```

The repository's `.gitignore` contains `.env`, which prevents the secret from being committed through normal Git tracking.

---

## 4.3 Creating the FastAPI Application

```python
app = FastAPI()
```

This creates the application object that Uvicorn serves.

---

## 4.4 CORS Configuration

The project uses:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Why CORS is needed

The Chrome extension calls the backend at:

```text
http://localhost:8000/ask
```

CORS middleware allows the browser-originated request to reach the FastAPI application.

### Current behavior

The implementation allows:

- every origin;
- every method;
- every header.

This is convenient for development. A production version should normally restrict allowed origins.

---

# 5. Request Validation

The backend defines:

```python
class QueryInput(BaseModel):
    query: str
    video_id: str
```

Therefore `/ask` expects:

```json
{
  "query": "what is machine learning?",
  "video_id": "ukzFI9rgwfU"
}
```

### Field meanings

| Field | Type | Meaning |
|---|---|---|
| `query` | `str` | User's question |
| `video_id` | `str` | YouTube video identifier |

Pydantic performs request validation before the endpoint function receives the data.

---

# 6. Model Initialization

At application startup:

```python
load_model()
```

The current `load_model()` implementation is a configuration check rather than local neural-network loading.

```python
def load_model():
    """Validate Hugging Face API access."""
    global api_key
    if not api_key:
        raise ValueError("HUGGINGFACE_API_KEY environment variable not set.")
    print("Hugging Face API initialized for mistralai/Mixtral-8x7B-Instruct-v0.1.")
```

Important: the project does **not** instantiate Mixtral locally in this function. The LLM is called remotely through Hugging Face.

---

# 7. `/ask` Endpoint

The main API route is:

```python
@app.post("/ask")
async def ask_question(data: QueryInput):
    answer = query_transcript(data.query, data.video_id)
    return {"answer": answer}
```

### Execution sequence

1. HTTP POST arrives.
2. FastAPI parses JSON.
3. Pydantic validates `query` and `video_id`.
4. `query_transcript()` is called.
5. The generated answer is returned.

Response:

```json
{
  "answer": "..."
}
```

---

# 8. Running the Backend

The README uses:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Interpretation:

- `app` before `:` = `app.py` module.
- `app` after `:` = FastAPI object.
- `0.0.0.0` = listen on all interfaces.
- `8000` = local port.
- `--reload` = development auto-reload.

The repository's deployment `Procfile` is:

```text
web: uvicorn app:app --host 0.0.0.0 --port $PORT
```

This lets a hosting platform provide the port through `$PORT`.

---

# 9. Core AI/RAG Module: `backend/model.py`

`model.py` contains the complete retrieval and generation pipeline.

Its logical stages are:

```text
Configuration
    ↓
Video collection lookup
    ↓
Transcript download if necessary
    ↓
VTT parsing
    ↓
Chunking
    ↓
ChromaDB indexing
    ↓
Similarity retrieval
    ↓
Context construction
    ↓
Prompt construction
    ↓
Hugging Face inference
    ↓
Answer parsing
```

---

# 10. ChromaDB Initialization

The code imports:

```python
from chromadb import Client
```

and creates:

```python
chroma_client = Client()
```

This client is used throughout the module.

The application uses Chroma as the transcript retrieval layer.

---

# 11. Hugging Face API Key

The key is read using:

```python
api_key = os.getenv("HUGGINGFACE_API_KEY")
```

It is later used in an HTTP Bearer authorization header.

---

# 12. Per-Video Collection Architecture

The collection name is generated with:

```python
collection_name = f"yt_transcript_{video_id}"
```

Example:

```text
Video ID: ukzFI9rgwfU
Collection: yt_transcript_ukzFI9rgwfU
```

Therefore the system conceptually has:

```text
Video A -> yt_transcript_A
Video B -> yt_transcript_B
Video C -> yt_transcript_C
```

This isolates transcript data between videos.

---

# 13. `get_or_create_collection(video_id)`

This function is responsible for finding an already-indexed transcript or building one.

## Step 1: Look for existing collection

```python
collection = chroma_client.get_collection(collection_name)
```

If successful:

```python
print(f"Using existing collection: {collection_name}")
return collection
```

This is effectively a cache at the collection level.

## Step 2: If collection does not exist

The function proceeds to download and process the transcript.

---

# 14. YouTube Subtitle Retrieval

The code creates:

```python
video_url = f"https://www.youtube.com/watch?v={video_id}"
subtitle_file = f"{video_id}.en.vtt"
```

It then runs:

```python
subprocess.run([
    "yt-dlp",
    "--write-auto-sub",
    "--sub-lang", "en",
    "--skip-download",
    "--sub-format", "vtt",
    "-o", video_id,
    video_url
], check=True, capture_output=True, text=True)
```

### Command breakdown

| Argument | Meaning |
|---|---|
| `yt-dlp` | YouTube extractor/downloader |
| `--write-auto-sub` | Request automatic subtitles |
| `--sub-lang en` | English subtitles |
| `--skip-download` | Do not download video itself |
| `--sub-format vtt` | WebVTT output |
| `-o video_id` | Output naming prefix |
| `video_url` | Target video |

The system therefore processes text rather than downloading the full video.

---

# 15. Subtitle Validation

After the command completes:

```python
if not os.path.exists(subtitle_file):
    print(f"No English subtitles found for video {video_id}")
    return None
```

If the English VTT file was not created, the transcript pipeline stops.

---

# 16. VTT Parsing

The code uses the `webvtt` package:

```python
transcript_text = ""
for caption in webvtt.read(subtitle_file):
    transcript_text += caption.text + " "
```

Each caption contains subtitle text and timing metadata. The implementation keeps the text and ignores the timestamps.

Conceptually:

```text
Caption 1 ─┐
Caption 2 ─┼──> Combined transcript text
Caption 3 ─┘
```

---

# 17. Temporary File Cleanup

After parsing:

```python
os.remove(subtitle_file)
```

The downloaded VTT file is deleted because the transcript text has already been extracted and indexed.

---

# 18. Creating the Chroma Collection

A new collection is created using:

```python
collection = chroma_client.create_collection(collection_name)
```

This collection belongs to one YouTube video.

---

# 19. Transcript Chunking

The current implementation performs very simple chunking:

```python
sentences = transcript_text.split(".")
documents = [s.strip() for s in sentences if s.strip()]
```

### Example

Input:

```text
Machine learning is a field. It learns from data. It is used in AI.
```

Approximate chunks:

```text
chunk 0 -> Machine learning is a field
chunk 1 -> It learns from data
chunk 2 -> It is used in AI
```

### Important implementation detail

The project does **not** currently use a recursive text splitter, token-based splitter, or overlapping chunk strategy. The period character is the boundary.

This makes the implementation simple, but chunks can have uneven lengths.

---

# 20. Chunk IDs

IDs are created using:

```python
ids = [f"chunk_{i}" for i in range(len(documents))]
```

Example:

```text
chunk_0
chunk_1
chunk_2
chunk_3
...
```

---

# 21. Indexing in ChromaDB

The documents are added with:

```python
collection.add(documents=documents, ids=ids)
```

The indexing pipeline is:

```text
VTT subtitles
     ↓
Transcript text
     ↓
Split by '.'
     ↓
Document chunks
     ↓
Chunk IDs
     ↓
Chroma collection
```

---

# 22. `query_transcript()`

The central question-answering function is:

```python
def query_transcript(query_text, video_id, n_results=5):
```

The parameters are:

| Parameter | Meaning |
|---|---|
| `query_text` | User question |
| `video_id` | YouTube video ID |
| `n_results` | Number of retrieved chunks |

The default is `5`.

---

# 23. Collection Retrieval

The first operation is:

```python
collection = get_or_create_collection(video_id)
```

If transcript retrieval fails:

```python
if not collection:
    return "Could not fetch transcript for the given video ID."
```

Thus the question-answering pipeline cannot continue without a transcript collection.

---

# 24. Similarity Retrieval

The user question is sent to Chroma:

```python
results = collection.query(
    query_texts=[query_text],
    n_results=n_results
)
```

The default retrieves five documents.

The returned documents are extracted with:

```python
retrieved_chunks = results['documents'][0]
```

Conceptually:

```text
User question
      ↓
Chroma similarity search
      ↓
Top 5 transcript chunks
```

Chroma handles the embedding/retrieval operation internally.

---

# 25. Empty Retrieval Handling

The code checks:

```python
if not retrieved_chunks:
    return "Could not find relevant information in the transcript."
```

This prevents an LLM request from being constructed from an empty context.

---

# 26. Context Construction

Retrieved chunks are joined with:

```python
context = "\n".join(retrieved_chunks)
```

This creates one context block for the language model.

---

# 27. RAG Prompt Construction

The prompt is:

```python
prompt = f"""Using the following context, answer the question.
If the answer is not in the context, say "I could not find the answer in the transcript."
Context:
{context}
Question: {query_text}

Answer:
"""
```

The prompt has four logical parts:

```text
1. Instruction
2. Retrieved transcript context
3. User question
4. Answer marker
```

The instruction explicitly tells the model to report that it could not find the answer if the answer is absent from the supplied context.

---

# 28. Hugging Face Generation

Headers:

```python
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
```

Payload:

```python
payload = {
    "inputs": prompt,
    "parameters": {
        "max_new_tokens": 200,
        "top_p": 0.95,
        "top_k": 50,
        "temperature": 0.7,
        "return_full_text": False
    }
}
```

Target model:

```text
mistralai/Mixtral-8x7B-Instruct-v0.1
```

Endpoint:

```text
https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1
```

---

# 29. Generation Parameters

| Parameter | Value | Purpose |
|---|---:|---|
| `max_new_tokens` | 200 | Limits generated output |
| `top_p` | 0.95 | Nucleus sampling threshold |
| `top_k` | 50 | Limits candidate tokens |
| `temperature` | 0.7 | Controls generation randomness |
| `return_full_text` | false | Returns generated continuation rather than the entire prompt |

---

# 30. HTTP Inference Call

The backend executes:

```python
response = requests.post(
    "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
    headers=headers,
    json=payload
)
response.raise_for_status()
result = response.json()
```

Therefore Mixtral inference is remote rather than local.

---

# 31. Reading `generated_text`

The code handles a list response:

```python
if isinstance(result, list) and len(result) > 0:
    answer = result[0].get('generated_text', '').strip()
```

and otherwise:

```python
answer = result.get('generated_text', '').strip()
```

The expected model output field is `generated_text`.

---

# 32. Answer Cleanup

The code searches for `Answer:`:

```python
answer_start_index = answer.find("Answer:")
if answer_start_index != -1:
    answer = answer[answer_start_index + len("Answer:"):].strip()
```

This removes the `Answer:` marker and any generated text before it when the model repeats the prompt structure.

---

# 33. Hugging Face Error Handling

The code catches request errors:

```python
except requests.exceptions.RequestException as e:
    print(f"Error calling Hugging Face API: {e}")
    return "Error generating response from the API."
```

---

# 34. Complete RAG Pipeline

The project implements:

```text
                  RETRIEVAL-AUGMENTED GENERATION

                  YouTube video
                       |
                       v
                English subtitles
                       |
                       v
                   VTT parser
                       |
                       v
                Transcript text
                       |
                       v
                Sentence chunks
                       |
                       v
                   ChromaDB
                       |
                 similarity search
                       ^
                       |
                 User question
                       |
                       v
                  Top 5 chunks
                       |
                       v
                    Context
                       |
                       v
                     Prompt
                       |
                       v
             Hugging Face Inference
                       |
                       v
                    Mixtral
                       |
                       v
                 Final answer
```

The project therefore follows:

```text
RAG = Retrieval + Augmentation + Generation
```

---

# 35. Chrome Extension Architecture

The extension consists of:

```text
extension/
├── manifest.json
├── popup.html
├── popup.js
├── style.css
└── icon.png
```

---

# 36. `manifest.json`

The extension uses Manifest V3:

```json
"manifest_version": 3
```

Permissions:

```json
"permissions": ["activeTab", "scripting"]
```

YouTube host permission:

```json
"host_permissions": ["https://*.youtube.com/*"]
```

Popup configuration:

```json
"action": {
  "default_popup": "popup.html",
  "default_icon": "icon.png"
}
```

The manifest also associates `popup.js` with YouTube watch URLs as a content script.

---

# 37. `popup.html`

The popup UI is intentionally simple:

```html
<h3>Ask about this video</h3>
<input id="queryInput" type="text" placeholder="Enter your question..." />
<button id="askBtn">Ask</button>
<div id="responseBox"></div>
```

The user interaction is:

```text
Question input
     ↓
Ask button
     ↓
Thinking...
     ↓
Generated answer
```

---

# 38. `popup.js`: Active Tab Detection

The script waits for the popup DOM:

```javascript
document.addEventListener("DOMContentLoaded", async () => {
```

Then gets the active tab:

```javascript
const [tab] = await chrome.tabs.query({
  active: true,
  currentWindow: true
});
```

The tab URL is used to identify the current video.

---

# 39. YouTube URL Validation

The code parses the URL:

```javascript
const url = new URL(tab.url);
```

Then checks:

```javascript
if (!url.hostname.includes("youtube.com") && !url.hostname.includes("youtu.be")) {
  document.body.innerHTML = "<p>This extension only works on YouTube video pages.</p>";
  return;
}
```

Non-YouTube pages are rejected.

---

# 40. Video ID Extraction

### Standard YouTube URL

For:

```text
https://www.youtube.com/watch?v=ukzFI9rgwfU
```

the code executes:

```javascript
videoId = new URLSearchParams(url.search).get("v");
```

### Short YouTube URL

For:

```text
https://youtu.be/ukzFI9rgwfU
```

the code executes:

```javascript
videoId = url.pathname.split("/")[1];
```

If no ID is found:

```javascript
if (!videoId) {
  document.body.innerHTML = "<p>Could not extract video ID.</p>";
  return;
}
```

---

# 41. User Query Collection

The Ask button reads:

```javascript
const query = document.getElementById("queryInput").value;
```

An empty query is rejected:

```javascript
if (!query) {
  responseBox.innerText = "Please enter a question.";
  return;
}
```

---

# 42. Loading State

Before sending the request:

```javascript
responseBox.innerText = "Thinking...";
document.getElementById("askBtn").disabled = true;
```

This prevents repeated clicks while the current request is running.

---

# 43. Frontend -> Backend Request

The extension uses:

```javascript
const res = await fetch("http://localhost:8000/ask", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    query: query,
    video_id: videoId
  }),
  signal: controller.signal
});
```

The data sent is exactly:

```json
{
  "query": "user question",
  "video_id": "current video ID"
}
```

---

# 44. Client Timeout

The extension uses an `AbortController`:

```javascript
const controller = new AbortController();
const timeoutId = setTimeout(() => controller.abort(), 10000);
```

The timeout is 10 seconds.

After a successful response:

```javascript
clearTimeout(timeoutId);
```

---

# 45. Response Handling

The response is parsed as JSON:

```javascript
const data = await res.json();
```

The extension then displays:

```javascript
responseBox.innerText = data.answer;
```

The expected response is:

```json
{
  "answer": "Generated response..."
}
```

---

# 46. Frontend Error Handling

If the backend response contains an `error` property, the extension displays it.

Network errors are caught by:

```javascript
catch (err) {
  responseBox.innerText = "Error: Ascertain that the server is running and accessible.";
  console.error("Fetch error:", err);
}
```

The button is always re-enabled:

```javascript
finally {
  document.getElementById("askBtn").disabled = false;
}
```

---

# 47. `style.css`

The project uses minimal CSS:

```css
body {
  font-family: sans-serif;
  padding: 10px;
}

input {
  width: 90%;
  padding: 5px;
}

button {
  margin-top: 10px;
  padding: 5px 10px;
}
```

The goal is a compact popup rather than a large standalone UI.

---

# 48. `backend/test.py`

`test.py` is a transcript inspection utility using `youtube-transcript-api`.

It calls:

```python
transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
```

and prints the available transcript languages and whether each transcript is generated.

It first tries a manually-created English transcript:

```python
transcripts.find_manually_created_transcript(['en'])
```

and falls back to an auto-generated transcript:

```python
transcripts.find_generated_transcript(['en'])
```

### Important distinction

The main production transcript path in `model.py` uses **`yt-dlp` + `webvtt`**, while `test.py` uses **`youtube-transcript-api`** for testing/inspection.

---

# 49. Dependencies

Current `requirements.txt`:

```text
transformers
accelerate
bitsandbytes
torch
youtube-transcript-api
chromadb
fastapi
uvicorn
requests
python-dotenv
yt-dlp
```

### Role mapping

| Dependency | Role |
|---|---|
| `fastapi` | REST backend |
| `uvicorn` | ASGI server |
| `chromadb` | Vector/retrieval store |
| `requests` | Hugging Face HTTP call |
| `python-dotenv` | Environment variables |
| `yt-dlp` | YouTube subtitle extraction |
| `youtube-transcript-api` | Transcript testing utility |
| `transformers` | Transformer ecosystem |
| `accelerate` | Transformer/inference ecosystem |
| `bitsandbytes` | Quantization ecosystem |
| `torch` | PyTorch ecosystem |
| `webvtt` | Directly imported by `model.py` for VTT parsing |

### Dependency consistency note

`model.py` directly imports `webvtt`, but the current `requirements.txt` does not explicitly list the corresponding WebVTT package. A fresh setup should ensure that dependency is installed.

Some other packages are present for the broader ML environment but are not directly imported by the current `app.py`/`model.py` implementation.

---

# 50. Installation and Local Setup

## Step 1 — Clone

```bash
git clone https://github.com/Manshu555/yt_chatbot.git
cd yt_chatbot
```

## Step 2 — Create a virtual environment

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

## Step 3 — Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

Also install the WebVTT dependency required by `model.py` if it is not included in your environment.

## Step 4 — Configure API key

Create `backend/.env`:

```env
HUGGINGFACE_API_KEY=your_token
```

## Step 5 — Start backend

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Step 6 — Load extension

1. Open `chrome://extensions/`.
2. Enable **Developer mode**.
3. Click **Load unpacked**.
4. Select the repository's `extension` folder.

## Step 7 — Ask questions

Open a YouTube video with an accessible English transcript, click the extension, enter a question, and click **Ask**.

---

# 51. API Documentation

## Endpoint

```text
POST /ask
```

## Request

```json
{
  "query": "what is machine learning?",
  "video_id": "ukzFI9rgwfU"
}
```

## Response

```json
{
  "answer": "..."
}
```

## Curl test

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"what is machine learning?","video_id":"ukzFI9rgwfU"}'
```

---

# 52. First Question vs Subsequent Questions

For a new video:

```text
Question
   ↓
Collection lookup
   ↓
No collection
   ↓
yt-dlp subtitle download
   ↓
VTT parsing
   ↓
Chunking
   ↓
Chroma indexing
   ↓
Retrieval
   ↓
LLM
```

For another question on the same video while the collection is available:

```text
Question
   ↓
Collection lookup
   ↓
Existing collection
   ↓
Retrieval
   ↓
LLM
```

The second path avoids repeating transcript download and indexing.

---

# 53. Data Contracts

### Extension -> API

```json
{
  "query": "user question",
  "video_id": "youtube video id"
}
```

### API -> Extension

```json
{
  "answer": "generated answer"
}
```

### Backend -> Chroma

```text
collection name
+ documents
+ document IDs
```

### Backend -> Hugging Face

```json
{
  "inputs": "RAG prompt",
  "parameters": {
    "max_new_tokens": 200,
    "top_p": 0.95,
    "top_k": 50,
    "temperature": 0.7,
    "return_full_text": false
  }
}
```

---

# 54. Complete Sequence Diagram

```text
User
 |
 | types question
 v
Chrome Popup
 |
 | active tab query
 v
Chrome Tabs API
 |
 | YouTube URL
 v
popup.js
 |
 | extract video_id
 |
 | POST /ask
 v
FastAPI
 |
 | QueryInput validation
 v
query_transcript()
 |
 v
get_or_create_collection()
 |
 +-------------------------------+
 | Collection exists?             |
 +---------------+---------------+
                 |
          +------+------+
          |             |
         YES           NO
          |             |
          |             v
          |           yt-dlp
          |             |
          |             v
          |        English VTT
          |             |
          |             v
          |           webvtt
          |             |
          |             v
          |       transcript text
          |             |
          |             v
          |      sentence chunks
          |             |
          |             v
          |          ChromaDB
          |             |
          +------+------+
                 |
                 v
          similarity search
                 |
                 v
             top 5 chunks
                 |
                 v
              context
                 |
                 v
             RAG prompt
                 |
                 v
          Hugging Face API
                 |
                 v
              Mixtral
                 |
                 v
             answer
                 |
                 v
             FastAPI
                 |
                 | JSON
                 v
           Chrome Popup
                 |
                 v
               User
```

---

# 55. Error Handling

The current implementation covers several failure cases.

### Missing Hugging Face key

```text
HUGGINGFACE_API_KEY environment variable not set.
```

### Missing English transcript

```text
Could not fetch transcript for the given video ID.
```

### No relevant chunks

```text
Could not find relevant information in the transcript.
```

### Hugging Face request failure

```text
Error generating response from the API.
```

### Extension opened on another site

```text
This extension only works on YouTube video pages.
```

### Video ID unavailable

```text
Could not extract video ID.
```

### Backend unavailable

The extension displays a message asking the user to ensure the server is running and accessible.

---

# 56. Security Considerations

## API key

The Hugging Face token should remain server-side. It should be stored in `.env` locally or in protected environment configuration in deployment.

Never put the token in extension JavaScript.

## `.gitignore`

The repository's `.gitignore` includes:

```text
.env
```

## CORS

Current implementation:

```python
allow_origins=["*"]
```

This is broad. A production deployment should restrict it to the expected origins.

## Localhost dependency

The extension currently calls:

```text
http://localhost:8000/ask
```

so the standard local workflow expects the backend to run on the same machine as Chrome.

---

# 57. Performance Characteristics

The main stages contributing to latency are:

1. **Subtitle acquisition** — network request through `yt-dlp` on a first-time video.
2. **Transcript parsing/chunking** — local CPU work.
3. **Chroma retrieval** — similarity search over indexed transcript chunks.
4. **LLM inference** — remote Hugging Face API request.

The first question for a video can therefore be slower than later questions because it includes transcript acquisition and indexing.

The extension also imposes a 10-second client-side timeout on the API request.

---

# 58. Important Current-Implementation Notes

1. The main transcript pipeline uses **`yt-dlp` + `webvtt`**.
2. `youtube-transcript-api` is used in **`test.py`**, not the main `get_or_create_collection()` implementation.
3. Transcript chunks are currently created using **`transcript_text.split(".")`**.
4. The default retrieval count is **5 chunks**.
5. Mixtral is called through the **Hugging Face Inference API**, not loaded locally.
6. The API currently returns only **`answer`**, not source snippets or timestamps.
7. Chroma is initialized with `Client()` in the current code; no explicit persistent storage path is configured in the shown implementation.
8. The extension currently uses a **hard-coded localhost backend URL**.
9. `popup.js` appears both as the popup script and as a manifest content script; its actual logic is primarily written around popup initialization and active-tab access.

These points are important when explaining the project in an interview because they describe what the code actually does rather than what a more advanced RAG implementation might do.

---

# 59. Conceptual RAG Formula

The system can be summarized as:

```text
Video ID
   ↓
Transcript T
   ↓
Chunks C = {c1, c2, ..., cn}
   ↓
R(q) = TopK(C, q), K = 5
   ↓
Context = concat(R(q))
   ↓
Answer = Mixtral(Context, q)
```

Where:

- `T` = transcript text;
- `C` = indexed transcript chunks;
- `q` = user question;
- `K` = 5 in the current implementation;
- `R(q)` = chunks retrieved for the question.

---

# 60. File-by-File Responsibility

| File | Responsibility |
|---|---|
| `backend/app.py` | FastAPI application, CORS, request validation, `/ask` |
| `backend/model.py` | Transcript retrieval, chunking, Chroma indexing, retrieval, prompt creation, Hugging Face generation |
| `backend/requirements.txt` | Python dependencies |
| `backend/Procfile` | Deployment start command |
| `backend/test.py` | Transcript availability/debugging |
| `extension/manifest.json` | Chrome extension metadata, permissions, scripts |
| `extension/popup.html` | Popup UI |
| `extension/popup.js` | Active-tab detection, video ID extraction, API request, response rendering |
| `extension/style.css` | Popup styling |
| `extension/icon.png` | Extension icon |
| `README.md` | Basic setup and API usage |
| `.gitignore` | Prevents `.env` from normal Git tracking |

---

# 61. Interview-Ready Explanation

> This project is a YouTube transcript-based RAG chatbot implemented as a Chrome extension and FastAPI backend. The extension extracts the current video's ID and sends it with the user's question to a `/ask` endpoint. The backend checks whether a ChromaDB collection exists for that video. If it does not, the backend uses `yt-dlp` to retrieve English subtitles, parses the VTT file using `webvtt`, splits the transcript into sentence-level documents, and stores them in ChromaDB. For a question, Chroma retrieves the top five relevant chunks. Those chunks are inserted into a prompt together with the user's question. The prompt is sent to `mistralai/Mixtral-8x7B-Instruct-v0.1` through the Hugging Face Inference API. The generated answer is returned as JSON and displayed in the Chrome extension.

The important architectural idea is that the LLM is not asked to answer from the entire YouTube transcript blindly. The system first retrieves relevant transcript information and then gives only that retrieved context to the model. This is the RAG pattern.

---

# 62. Potential Production Improvements

The following are improvements to the current implementation, not existing features.

## 62.1 Better chunking

Replace period-only splitting with token-aware chunks and overlap. This can preserve context across sentence boundaries and create more consistent retrieval units.

## 62.2 Timestamp-aware chunks

Store the VTT start/end timestamps alongside each chunk. The answer could then link the user to the relevant section of the video.

## 62.3 Persistent Chroma storage

Use an explicitly persistent Chroma configuration if transcript indexes need to survive process restarts.

## 62.4 Source-aware API response

The backend could return:

```json
{
  "answer": "...",
  "sources": [
    {
      "chunk_id": "chunk_12",
      "text": "...",
      "timestamp": "..."
    }
  ]
}
```

This would make the answer traceable to transcript evidence.

## 62.5 Better API errors

Return structured error JSON and appropriate HTTP status codes instead of returning plain error strings for every failure.

## 62.6 Configurable backend URL

Move `http://localhost:8000/ask` into extension configuration so local and deployed backends can be selected without changing the application logic.

## 62.7 Restrict CORS

Replace the wildcard origin with an explicit allow-list in production.

## 62.8 Dependency cleanup

Make the WebVTT dependency explicit and remove packages that are not required by the final runtime after validating the environment.

---

# 63. Final Architecture Summary

```text
                         +-------------------+
                         |      YouTube      |
                         |   Current Video   |
                         +---------+---------+
                                   |
                                   | URL
                                   v
                         +-------------------+
                         | Chrome Extension  |
                         |-------------------|
                         | popup.html        |
                         | popup.js          |
                         | style.css         |
                         +---------+---------+
                                   |
                                   | POST /ask
                                   | query + video_id
                                   v
                         +-------------------+
                         |      FastAPI      |
                         |      app.py       |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         |      model.py     |
                         |-------------------|
                         | collection lookup |
                         | yt-dlp            |
                         | VTT parsing       |
                         | chunking          |
                         | Chroma retrieval  |
                         | prompt generation |
                         | HF API call       |
                         +----+---------+----+
                              |         |
                              |         |
                              v         v
                      +-----------+ +-----------+
                      | ChromaDB  | | Hugging   |
                      | transcript| | Face      |
                      | retrieval | | Mixtral   |
                      +-----------+ +-----+-----+
                                          |
                                          | answer
                                          v
                                   +--------------+
                                   | FastAPI JSON |
                                   +------+-------+
                                          |
                                          v
                                   +--------------+
                                   | Chrome Popup |
                                   +--------------+
```

The project is therefore a practical **Chrome Extension + FastAPI + ChromaDB + Retrieval-Augmented Generation + Hugging Face LLM** application for conversational interaction with YouTube transcripts.

---

## Source Files Analyzed

This documentation was derived from the current repository implementation of:

- `README.md`
- `backend/app.py`
- `backend/model.py`
- `backend/requirements.txt`
- `backend/Procfile`
- `backend/test.py`
- `extension/manifest.json`
- `extension/popup.html`
- `extension/popup.js`
- `extension/style.css`
- `.gitignore`

Repository: `Manshu555/yt_chatbot`
