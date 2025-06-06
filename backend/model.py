import os
import requests
import subprocess
import webvtt
from chromadb import Client

# --- Global variables ---
chroma_client = Client()
api_key = os.getenv("HUGGINGFACE_API_KEY")  # Ensure this is set in your environment

def load_model():
    """Validate Hugging Face API access."""
    global api_key
    if not api_key:
        raise ValueError("HUGGINGFACE_API_KEY environment variable not set.")
    print("Hugging Face API initialized for mistralai/Mixtral-8x7B-Instruct-v0.1.")

def get_or_create_collection(video_id):
    collection_name = f"yt_transcript_{video_id}"

    # If a Chroma collection already exists, reuse it
    try:
        collection = chroma_client.get_collection(collection_name)
        print(f"Using existing collection: {collection_name}")
        return collection
    except Exception:
        # Collection doesn't exist; create it below
        pass

    # Use yt-dlp to fetch subtitles
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    subtitle_file = f"{video_id}.en.vtt"

    try:
        # Run yt-dlp to download subtitles (English, auto-generated or manual)
        subprocess.run([
            "yt-dlp",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--skip-download",
            "--sub-format", "vtt",
            "-o", video_id,
            video_url
        ], check=True, capture_output=True, text=True)

        # Check if subtitle file exists
        if not os.path.exists(subtitle_file):
            print(f"No English subtitles found for video {video_id}")
            return None

        # Parse the .vtt file
        transcript_text = ""
        for caption in webvtt.read(subtitle_file):
            transcript_text += caption.text + " "

        # Clean up the subtitle file
        os.remove(subtitle_file)

        print(f"Fetched transcript for video {video_id}: {len(transcript_text)} characters")

    except subprocess.CalledProcessError as e:
        print(f"Error fetching subtitles for video {video_id} with yt-dlp: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error while fetching subtitles for video {video_id}: {e}")
        return None

    # Create new collection
    collection = chroma_client.create_collection(collection_name)
    print(f"Created new collection: {collection_name}")

    # Split into sentences
    sentences = transcript_text.split(".")
    documents = [s.strip() for s in sentences if s.strip()]
    ids = [f"chunk_{i}" for i in range(len(documents))]

    if documents:
        print(f"Adding {len(documents)} chunks to ChromaDB for video {video_id}")
        collection.add(documents=documents, ids=ids)
    else:
        print(f"No valid documents to add to ChromaDB for video {video_id}")
        return None

    return collection

def query_transcript(query_text, video_id, n_results=5):
    """Query the transcript and use Hugging Face API for mistralai/Mixtral-8x7B-Instruct-v0.1 to generate an answer."""
    global api_key

    print(f"\nQuery: {query_text} | Video ID: {video_id}")
    collection = get_or_create_collection(video_id)
    if not collection:
        return "Could not fetch transcript for the given video ID."

    results = collection.query(query_texts=[query_text], n_results=n_results)
    retrieved_chunks = results['documents'][0]

    if not retrieved_chunks:
        return "Could not find relevant information in the transcript."

    context = "\n".join(retrieved_chunks)
    prompt = f"""Using the following context, answer the question.
If the answer is not in the context, say "I could not find the answer in the transcript."
Context:
{context}
Question: {query_text}

Answer:
"""

    # Call Hugging Face Inference API for mistralai/Mixtral-8x7B-Instruct-v0.1
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
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

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            answer = result[0].get('generated_text', '').strip()
        else:
            answer = result.get('generated_text', '').strip()

        # Extract the answer part after "Answer:" if present
        answer_start_index = answer.find("Answer:")
        if answer_start_index != -1:
            answer = answer[answer_start_index + len("Answer:"):].strip()
        return answer or "No valid response from the API."

    except requests.exceptions.RequestException as e:
        print(f"Error calling Hugging Face API: {e}")
        return "Error generating response from the API."