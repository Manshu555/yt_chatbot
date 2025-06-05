from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from youtube_transcript_api import YouTubeTranscriptApi
from chromadb import Client
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from chromadb import Client
# --- Global model/tokenizer ---
model = None
tokenizer = None
chroma_client = Client()

def load_model():
    global model, tokenizer
    model_name = "distilgpt2"  # Lightweight model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"  # Automatically map to CPU/GPU if available
    )
    # Set pad_token_id if not already set (required for distilgpt2)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Model loaded.")

chroma_client = Client()
def get_or_create_collection(video_id):
    collection_name = f"yt_transcript_{video_id}"

    # If a Chroma collection already exists, reuse it
    try:
        collection = chroma_client.get_collection(collection_name)
        print(f"Using existing collection: {collection_name}")
        return collection
    except Exception:
        # (Collection doesn't exist yet; we will create it below.)
        pass

    # Attempt to fetch the transcript (manual first, then auto)
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            # Prefer a manually created English transcript
            transcript = transcripts.find_manually_created_transcript(["en"])
        except NoTranscriptFound:
            # If no manual English transcript, fall back to generated
            transcript = transcripts.find_generated_transcript(["en"])

        try:
            # This is where ExpatError can occur if raw_data is empty
            transcript_list = transcript.fetch()
        except Exception as fetch_err:
            # Catches the xml.parsers.expat.ExpatError (and any other fetch-time error)
            print(f"Transcript fetch error for video {video_id}: {fetch_err}")
            return None

        # Build a single string from all the fetched snippets
        transcript_text = " ".join([item.text for item in transcript_list])

    except (TranscriptsDisabled, NoTranscriptFound) as e:
        # These errors mean “no transcript available” at the list/lookup stage
        print(f"Transcript lookup error for video {video_id}: {e}")
        return None

    # At this point, we have a valid transcript_text
    collection = chroma_client.create_collection(collection_name)
    print(f"Created new collection: {collection_name}")

    # Split into sentences (or however you prefer to chunk)
    sentences = transcript_text.split(".")
    documents = [s.strip() for s in sentences if s.strip()]
    ids = [f"chunk_{i}" for i in range(len(documents))]

    if documents:
        print(f"Adding {len(documents)} chunks to ChromaDB for video {video_id}")
        collection.add(documents=documents, ids=ids)

    return collection

def query_transcript(query_text, video_id, n_results=5):
    global model, tokenizer

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

    print("Generating response with LLM...")
    max_input_length = 800
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_start_index = response.find("Answer:")
    return response[answer_start_index + len("Answer:"):].strip() if answer_start_index != -1 else response.strip()